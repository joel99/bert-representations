#!/usr/bin/env python3

# Src: https://colab.research.google.com/github/huggingface/blog/blob/master/notebooks/trainer/01_text_classification.ipynb
from transformers import AutoConfig, AutoModelForSequenceClassification, GlueDataset
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import (
    Trainer,
    glue_tasks_num_labels,
)

from src import (
    get_eval_metrics_func,
    logger
)

def run_mnli(cfg, model_args, training_args, tokenizer, ckpt_path=None):
    r"""
        cfg: YACS cfg node
        ckpt_path: Unsupported
    """
    task_name = "mnli"

    data_args = DataTrainingArguments(
        task_name=task_name,
        data_dir=cfg.DATA.DATAPATH
    )

    num_labels = glue_tasks_num_labels[data_args.task_name]
    logger.info(f"Num MNLI Labels: \t {num_labels}")

    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        config=config,
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

    trainer.train()