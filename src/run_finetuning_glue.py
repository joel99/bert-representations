#!/usr/bin/env python3
import os.path as osp
from yacs.config import CfgNode as CN

import torch
from torch.utils.data.dataloader import DataLoader

# Src: https://colab.research.google.com/github/huggingface/blog/blob/master/notebooks/trainer/01_text_classification.ipynb
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import (
    DataCollatorWithPadding
)
from src.utils import (
    logger,
    get_eval_metrics_func,
    TASK_KEY_TO_NAME,
    FixedTrainer,
    get_extract_path,
    get_metrics_path
)
from src.registry import load_features_dict

def run_glue(task_key, cfg, model, model_args, training_args, tokenizer, mode="train", extract=False, **kwargs):
    r"""
        cfg: YACS cfg node
        ckpt_path: Unsupported
    """
    task_name = TASK_KEY_TO_NAME[task_key]

    data_args = DataTrainingArguments(
        task_name=task_name,
        data_dir=cfg.DATA.DATAPATH
    )

    glue_dataset = load_features_dict(tokenizer, cfg)
    # print(glue_dataset.keys())
    train_dataset = glue_dataset[task_key]['train']
    split_key = cfg.EVAL.SPLIT
    if task_key == "mnli":
        split_key = f"{split_key}_matched"
    eval_dataset = glue_dataset[task_key][split_key]
    # eval_dataset = glue_dataset[task_key]['validation_mismached']
    # train_dataset = GlueDataset(data_args, tokenizer=tokenizer, limit_length=cfg.TRAIN.TASK_LIMIT)
    # eval_dataset = GlueDataset(data_args, tokenizer=tokenizer, mode='dev')
    collator = DataCollatorWithPadding(tokenizer)
    trainer = FixedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=get_eval_metrics_func(task_key),
        data_collator=collator,
        config=cfg
    )

    if mode == "train":
        trainer.train()
    else:
        extract_path = None
        if extract:
            extract_path = get_extract_path(cfg, model_args)
        metrics = trainer.evaluate(
            extract_path=extract_path,
            cache_path=osp.join(cfg.TASK.EXTRACT_TOKENS_MASK_CACHE, task_key)
        )
        metrics_file = get_metrics_path(cfg, model_args)
        torch.save(metrics, metrics_file)