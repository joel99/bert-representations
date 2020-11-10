import os
import sys
from dataclasses import dataclass, field
from importlib import import_module
from typing import Dict, List, Optional, Tuple, Union
from yacs.config import CfgNode as CN

import numpy as np

from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    set_seed,
    PreTrainedTokenizerBase,
)
from src.registry import load_features_dict
from src.utils import DataCollatorForTokenClassification

from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import (
    Trainer,
    glue_tasks_num_labels,
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

from src.utils import (
    logger,
    get_eval_metrics_func,
)
from src.registry import get_config


def run_pos(task_key: str, cfg: CN, model, model_args, training_args, tokenizer, mode="train"):
    r"""
        cfg: YACS cfg node
        ckpt_path: Unsupported
    """
    task_name = "POS"
    data_args = DataTrainingArguments(
        task_name=task_name,
        data_dir=cfg.DATA.DATAPATH
    )

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config, labels, label_map = get_config(task_key, cfg)

    # Get datasets
    pos_dataset = load_features_dict(tokenizer, cfg)
    train_dataset = pos_dataset["pos"]["train"]
    eval_dataset = pos_dataset["pos"]["validation"]

    # train_dataset = (
    #     TokenClassificationDataset(
    #         token_classification_task=token_classification_task,
    #         data_dir=data_args.data_dir,
    #         tokenizer=tokenizer,
    #         labels=labels,
    #         model_type=config.model_type,
    #         max_seq_length=data_args.max_seq_length,
    #         overwrite_cache=data_args.overwrite_cache,
    #         mode=Split.train,
    #     )
    #     if training_args.do_train
    #     else None
    # )
    # eval_dataset = (
    #     TokenClassificationDataset(
    #         token_classification_task=token_classification_task,
    #         data_dir=data_args.data_dir,
    #         tokenizer=tokenizer,
    #         labels=labels,
    #         model_type=config.model_type,
    #         max_seq_length=data_args.max_seq_length,
    #         overwrite_cache=data_args.overwrite_cache,
    #         mode=Split.dev,
    #     )
    #     if training_args.do_eval
    #     else Nones
    # )

    compute_metrics_raw = get_eval_metrics_func(task_name)
    compute_metrics = lambda p: compute_metrics_raw(label_map, p)

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
    )

    if mode == "train":
        trainer.train()
    else:
        trainer.evaluate()
