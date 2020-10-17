#!/usr/bin/env python3
from yacs.config import CfgNode as CN

# Src: https://colab.research.google.com/github/huggingface/blog/blob/master/notebooks/trainer/01_text_classification.ipynb
from transformers import AutoConfig, AutoModelForSequenceClassification, GlueDataset
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import (
    Trainer,
    glue_tasks_num_labels,
)

from src.utils import (
    logger,
    get_eval_metrics_func,
    TASK_KEY_TO_NAME,
)
from src.registry import get_model

def run_glue(task_key, cfg, model, model_args, training_args, tokenizer, mode="train"):
    r"""
        cfg: YACS cfg node
        ckpt_path: Unsupported
    """
    task_name = TASK_KEY_TO_NAME[task_key]

    data_args = DataTrainingArguments(
        task_name=task_name,
        data_dir=cfg.DATA.DATAPATH
    )

    train_dataset = GlueDataset(data_args, tokenizer=tokenizer, limit_length=100_000)
    eval_dataset = GlueDataset(data_args, tokenizer=tokenizer, mode='dev')

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=get_eval_metrics_func(task_name),
    )

    if mode == "train":
        trainer.train()
    else:
        trainer.evaluate()
