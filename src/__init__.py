#!/usr/bin/env python3
import os.path as osp
from yacs.config import CfgNode as CN

from transformers import TrainingArguments, AutoTokenizer

from src.utils import (
    logger,
    ModelArguments,
    find_data_path,
    find_most_recent_path
)

from src.run_finetuning_pos import run_pos
from src.run_finetuning_glue import run_glue
from src.registry import get_model
from src.multitask import run_multitask
# init depends on common
# finetuning depnds on cmmon
# common depends on finetuning

# ! RESOLVE DEPENDENCIES

# The tasks we specify in config.yaml are the keys here
TASK_DICT = {
    "mnli": run_glue,
    "pos": run_pos,
    "sts_b": run_glue,
    "sst_2": run_glue
}

MULTITASK_STRATEGIES = {
    "SEQUENTIAL", # data-proportional sequential training (aka train one task after the other). With multiple epochs, will cycle between the tasks.
    "SAMPLE", # data-proportional sampled training
    "EQUAL_SEQUENTIAL", # equal update sequential training
    "EQUAL_SAMPLE", # equal update sampled training
    "FULL_SEQUENTIAL", # multi-epoch sequential. trains task A for N epochs, then task B
    "MANUAL_SEQUENTIAL" # Legacy
}

def make_training_args(cfg, checkpoint_path=None):
    is_multitasking = len(cfg.TASK.TASKS) > 1 and cfg.TASK.MULTITASK_STRATEGY != "MANUAL_SEQUENTIAL"
    return TrainingArguments(
        output_dir=cfg.MODEL_DIR,
        overwrite_output_dir=checkpoint_path is not None, # ? uncertain about this
        do_train=True,
        do_eval=not is_multitasking and cfg.TRAIN.DO_VAL,
        per_device_train_batch_size=cfg.TRAIN.BATCH_SIZE,
        per_device_eval_batch_size=cfg.EVAL.BATCH_SIZE,
        num_train_epochs=1 if cfg.TASK.MULTITASK_STRATEGY == "FULL_SEQUENTIAL" else cfg.TRAIN.NUM_EPOCHS_PER_TASK,
        max_steps=cfg.TRAIN.NUM_UPDATES_PER_TASK,
        logging_steps=cfg.TRAIN.LOG_INTERVAL,
        logging_first_step=True,
        logging_dir=cfg.TENSORBOARD_DIR,
        save_steps=cfg.TRAIN.CHECKPOINT_INTERVAL,
        evaluate_during_training=not is_multitasking and cfg.TRAIN.DO_VAL,
        learning_rate=cfg.TRAIN.LR_INIT,
        warmup_steps=cfg.TRAIN.LR_WARMUP_STEPS,
        weight_decay=cfg.TRAIN.WEIGHT_DECAY,
        eval_steps=cfg.TRAIN.EVAL_STEPS,
        seed=cfg.SEED
    )

def get_runner_func(
    cfg: CN,
    checkpoint_path: str=None,
    mode: str="train",
):
    r"""
        Return function that orchestrates fine-tuning.
        TODO: bind task config only
    """
    model_args = ModelArguments(
        model_name_or_path=cfg.MODEL.BASE if checkpoint_path is None else checkpoint_path,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.MODEL.BASE,
        use_fast=True,
    )

    assert len(cfg.TASK.TASKS) > 0, "requires positive number of tasks"
    for task in cfg.TASK.TASKS:
        assert task in TASK_DICT, f"unknown task {task}"
    if len(cfg.TASK.TASKS) == 1:
        task = cfg.TASK.TASKS[0]
        training_args = make_training_args(cfg, checkpoint_path=checkpoint_path)
        logger.info(f"{task} training arguments:{training_args}")
        cfg.defrost()
        cfg.DATA.DATAPATH = find_data_path(cfg.DATA.DATAPATH, task)
        cfg.freeze()

        model = get_model(task, cfg, model_args, ckpt_path=checkpoint_path)
        bound_task = lambda *args, **kwargs: \
            TASK_DICT[task](
                task, # Pass the runner the task name so it can pull any information we want to keep flexible from the registry
                cfg,
                model,
                model_args,
                training_args,
                tokenizer,
                mode=mode,
                *args,
                **kwargs
            )
        return bound_task
    assert cfg.TASK.MULTITASK_STRATEGY in MULTITASK_STRATEGIES
    if cfg.TASK.MULTITASK_STRATEGY == "MANUAL_SEQUENTIAL":
        def sequential_evaluation(*args, **kwargs):
            # ! This doesn't support different model types
            task_checkpoint = checkpoint_path
            for i, task in enumerate(cfg.TASK.TASKS):
                # Update configs to use subdirectories, to enable intermediate task analysis.
                task_specific_cfg = cfg.clone()
                task_specific_cfg.defrost()
                # TODO update/clone(?) tensorboard dir as well, once we figure out how that's actually used
                task_specific_cfg.MODEL_DIR = osp.join(task_specific_cfg.MODEL_DIR, f'{i}_{task}')
                task_specific_cfg.DATA.DATAPATH = find_data_path(cfg.DATA.DATAPATH, task)
                task_specific_cfg.freeze()
                training_args = make_training_args(task_specific_cfg, checkpoint_path=task_checkpoint)
                model = get_model(task, task_specific_cfg, model_args, ckpt_path=task_checkpoint)
                logger.info(f"{task} training arguments:{training_args}")
                bound_task = lambda *args, **kwargs: \
                    TASK_DICT[task](
                        task,
                        task_specific_cfg,
                        model,
                        model_args,
                        training_args,
                        tokenizer,
                        mode=mode,
                        *args,
                        **kwargs
                    )
                bound_task(ckpt_path=task_checkpoint, *args, **kwargs)
                # Use previous task's most recent checkpoint as next checkpoint file
                task_checkpoint = find_most_recent_path(task_specific_cfg.MODEL_DIR)

        return sequential_evaluation
    else:
        if "EQUAL" in cfg.TASK.MULTITASK_STRATEGY:
            assert cfg.TRAIN.NUM_UPDATES_PER_TASK > 0, "equal settings require update specification"
            cfg.defrost()
            cfg.TRAIN.NUM_UPDATES_PER_TASK *= len(cfg.TASK.TASKS)
            cfg.freeze()
        else:
            assert cfg.TRAIN.NUM_UPDATES_PER_TASK <= 0, "unsafe to run equal strategies without epoch setting"
        if "FULL_SEQUENTIAL" in cfg.TASK.MULTITASK_STRATEGY:
            assert cfg.TRAIN.NUM_UPDATES_PER_TASK < 0, "Full sequencing uses epochs"
        training_args = make_training_args(cfg, checkpoint_path=checkpoint_path)
        def bound_multitask(*args, **kwargs):
            run_multitask(
                cfg,
                model_args,
                training_args,
                tokenizer,
                mode=mode,
                checkpoint_path=checkpoint_path,
                *args,
                **kwargs)
        return bound_multitask