#!/usr/bin/env python3
from yacs.config import CfgNode as CN

# Src: https://colab.research.google.com/github/huggingface/blog/blob/master/notebooks/trainer/01_text_classification.ipynb
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import (
    DataCollatorWithPadding
)
# transformers.logging.set_verbosity_info()
from src.utils import (
    logger,
    get_eval_metrics_func,
    TASK_KEY_TO_NAME,
    FixedTrainer
)
from src.registry import get_model, load_features_dict

def run_glue(task_key, cfg, model, model_args, training_args, tokenizer, mode="train", **kwargs):
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
    trainer = FixedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=get_eval_metrics_func(task_key),
        data_collator=DataCollatorWithPadding(tokenizer)
    )

    if mode == "train":
        trainer.train()
    else:
        trainer.evaluate()