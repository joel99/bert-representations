#!/usr/bin/env python3
# Registry e.g. for config
# * We use this abstraction so that multi-task models can be init-ed simply

from yacs.config import CfgNode as CN

import transformers
from transformers import AutoConfig, glue_tasks_num_labels

from src.utils import (
    logger,
    ModelArguments,
    TASK_KEY_TO_NAME
)

from src.run_finetuning_pos import get_pos_config

def get_glue_config(cfg: CN, model_args, name: str):
    num_labels = glue_tasks_num_labels[name]
    logger.info(f"Num {name} Labels: \t {num_labels}")

    return (AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=name
    ),)

CONFIG_MAP = {
    "pos": get_pos_config,
    "mnli": get_glue_config,
    "sts_b": get_glue_config,
    "sst_2": get_glue_config,
    # ! Add DP
}

def get_config(task_key: str, cfg: CN, model_args):
    # Configs returned from this follow a tuple interface, in case we fetch additional info (see POS)
    model_args = ModelArguments(
        model_name_or_path=cfg.MODEL.BASE,
    )
    name = TASK_KEY_TO_NAME[task_key]

    return CONFIG_MAP[task_key](cfg, model_args, name)

MODEL_TYPE_REFERENCE={
    "mnli": transformers.AutoModelForSequenceClassification,
    "sst_2": transformers.AutoModelForSequenceClassification,
    "sts_b": transformers.AutoModelForSequenceClassification,
    "pos": transformers.AutoModelForTokenClassification,
    # ! Add DP
}

def get_model_type(task_key: str, cfg: CN):
    return MODEL_TYPE_REFERENCE[task_key]

def get_model(task_key: str, cfg: CN, model_args, ckpt_path=None):
    # Ruining huggingface's delicately chosen abstractions one function at a time
    model_type = get_model_type(task_key, cfg)
    if ckpt_path is not None:
        model = model_type.from_pretrained(
            ckpt_path
        )
    else:
        model_cfg = get_config(task_key, cfg, model_args)[0]
        model = model_type.from_pretrained(
            model_args.model_name_or_path,
            config=model_cfg, # ! What is this config even for?
            cache_dir=model_args.cache_dir
        )
    return model