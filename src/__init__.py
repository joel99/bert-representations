#!/usr/bin/env python3

from transformers import TrainingArguments, AutoTokenizer

from src.logger_wrapper import logger

# Import tasks first or we'll go circular

from src.common import (
    ModelArguments,
    get_eval_metrics_func,
)

from src.run_finetuning_mnli import run_mnli
from src.run_finetuning_sst_2 import run_sst_2
from src.run_finetuning_sts_b import run_sts_b
from src.run_finetuning_ner import run_ner
# init depends on common
# finetuning depnds on cmmon
# common depends on finetuning
TASK_DICT = {
    "mnli": run_mnli,
    "pos": run_pos,
    "sts_b": run_sts_b,
    "sst_2": run_sst_2
}

def get_train_func(cfg, checkpoint_path=None):
    r"""
        Return function that orchestrates fine-tuning.
        TODO: bind task config only
        TODO: add support for sequential training
    """
    model_args = ModelArguments(
        model_name_or_path=cfg.MODEL.BASE,
    )

    training_args = TrainingArguments(
        output_dir=cfg.MODEL_DIR,
        overwrite_output_dir=checkpoint_path is not None, # ? uncertain about this
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=cfg.TRAIN.BATCH_SIZE,
        per_device_eval_batch_size=cfg.EVAL.BATCH_SIZE,
        num_train_epochs=cfg.TRAIN.NUM_EPOCHS_PER_TASK,
        logging_steps=cfg.TRAIN.LOG_INTERVAL,
        logging_first_step=True,
        logging_dir=cfg.TENSORBOARD_DIR,
        save_steps=cfg.TRAIN.CHECKPOINT_INTERVAL,
        evaluate_during_training=cfg.TRAIN.DO_VAL,
        learning_rate=cfg.TRAIN.LR_INIT,
        weight_decay=cfg.TRAIN.WEIGHT_DECAY,
        eval_steps=cfg.TRAIN.EVAL_STEPS
    )
    print(f"training arguments:{training_args}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
    )

    task = TASK_DICT[cfg.TASK.TASKS[0]]
    def bound_task(*args, **kwargs):
        return task(
            cfg,
            model_args,
            training_args,
            tokenizer,
            *args,
            **kwargs
        )
    return bound_task
