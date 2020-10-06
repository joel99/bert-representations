#!/usr/bin/env python3
import os.path as osp
from yacs.config import CfgNode as CN

from transformers import TrainingArguments, AutoTokenizer

from src.logger_wrapper import logger

# Import tasks first or we'll go circular

from src.common import (
    ModelArguments,
    get_eval_metrics_func,
)
from src.utils import find_most_recent_path, find_data_path

from src.run_finetuning_mnli import run_mnli
from src.run_finetuning_sst_2 import run_sst_2
from src.run_finetuning_sts_b import run_sts_b
from src.run_finetuning_ner import run_ner
# init depends on common
# finetuning depnds on cmmon
# common depends on finetuning
TASK_DICT = {
    "mnli": run_mnli,
    "ner": run_ner,
    "sts_b": run_sts_b,
    "sst_2": run_sst_2
}

MULTITASK_STRATEGIES = { "SEQUENTIAL" }

def make_training_args(cfg, checkpoint_path=None):
    return TrainingArguments(
        output_dir=cfg.MODEL_DIR,
        overwrite_output_dir=checkpoint_path is not None, # ? uncertain about this
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=cfg.TRAIN.BATCH_SIZE,
        per_device_eval_batch_size=cfg.EVAL.BATCH_SIZE,
        num_train_epochs=cfg.TRAIN.NUM_EPOCHS_PER_TASK,
        max_steps=cfg.TRAIN.NUM_UPDATES_PER_TASK,
        logging_steps=cfg.TRAIN.LOG_INTERVAL,
        logging_first_step=True,
        logging_dir=cfg.TENSORBOARD_DIR,
        save_steps=cfg.TRAIN.CHECKPOINT_INTERVAL,
        evaluate_during_training=cfg.TRAIN.DO_VAL,
        learning_rate=cfg.TRAIN.LR_INIT,
        weight_decay=cfg.TRAIN.WEIGHT_DECAY,
        eval_steps=cfg.TRAIN.EVAL_STEPS
    )

def get_train_func(
    cfg: CN,
    checkpoint_path: str=None
):
    r"""
        Return function that orchestrates fine-tuning.
        TODO: bind task config only
    """
    model_args = ModelArguments(
        model_name_or_path=cfg.MODEL.BASE,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
    )

    assert len(cfg.TASK.TASKS) > 0, "requires positive number of tasks"
    if len(cfg.TASK.TASKS) == 1:
        task = cfg.TASK.TASKS[0]
        training_args = make_training_args(cfg, checkpoint_path=checkpoint_path)
        logger.info(f"{task} training arguments:{training_args}")
        cfg.defrost()
        cfg.DATA.DATAPATH = find_data_path(cfg.DATA.DATAPATH, task)
        cfg.freeze()
        bound_task = lambda *args, **kwargs: \
            TASK_DICT[task](
                cfg,
                model_args,
                training_args,
                tokenizer,
                *args,
                **kwargs
            )
        return bound_task
    assert cfg.TASK.MULTITASK_STRATEGY in MULTITASK_STRATEGIES
    if cfg.TASK.MULTITASK_STRATEGY == "SEQUENTIAL":
        def sequential_evaluation(*args, **kwargs):
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
                logger.info(f"{task} training arguments:{training_args}")
                bound_task = lambda *args, **kwargs: \
                    TASK_DICT[task](
                        task_specific_cfg,
                        model_args,
                        training_args,
                        tokenizer,
                        *args,
                        **kwargs
                    )
                bound_task(ckpt_path=task_checkpoint, *args, **kwargs)
                # Use previous task's most recent checkpoint as next checkpoint file
                task_checkpoint = find_most_recent_path(task_specific_cfg.MODEL_DIR)

        return sequential_evaluation
