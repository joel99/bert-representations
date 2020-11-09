#!/usr/bin/env python3
# Registry e.g. for config
# * We use this abstraction so that multi-task models can be init-ed simply

import numpy as np
from yacs.config import CfgNode as CN

import transformers
from transformers import AutoConfig, glue_tasks_num_labels
import datasets as nlp

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

# ---
# Data and dataloading
# ---

# Taking at face value that we need to encode like so for NLP to work properly
def convert_to_stsb_features(cfg, tokenizer, example_batch):
    inputs = list(zip(example_batch['sentence1'], example_batch['sentence2']))
    features = tokenizer.batch_encode_plus(
        inputs, max_length=cfg.MODEL.MAX_LENGTH, pad_to_max_length=True, truncation=True,
    )
    features["labels"] = example_batch["label"]
    return features

def convert_to_mnli_features(cfg, tokenizer, example_batch):
    inputs = list(zip(example_batch['hypothesis'], example_batch['premise']))
    features = tokenizer.batch_encode_plus(
        inputs, max_length=cfg.MODEL.MAX_LENGTH, pad_to_max_length=True, truncation=True,
    )
    features["labels"] = example_batch["label"]
    return features

def convert_to_sst2_features(cfg, tokenizer, example_batch):
    features = tokenizer.batch_encode_plus(
        example_batch["sentence"], max_length=cfg.MODEL.MAX_LENGTH, pad_to_max_length=True, truncation=True,
    )
    features["labels"] = example_batch["label"]
    return features

def convert_to_pos_features(cfg, tokenizer, example_batch):
    # This is the most naive guess, `utils_ner` suggest the actual procedure is harder
    # TODO use the utils_ner conversion -- it'll also process the labels correctly..
    features = tokenizer(example_batch['words'])
    features["labels"] = example_batch["pos"]
    return features

convert_func_dict = {
    "sts_b": convert_to_stsb_features, # Ok, either use theirs, or ours. Um.
    "sst_2": convert_to_sst2_features,
    "mnli": convert_to_mnli_features,
    "pos": convert_to_pos_features
}

columns_dict = {
    "sst_2": ['input_ids', 'attention_mask', 'labels'],
    "sts_b": ['input_ids', 'attention_mask', 'labels'],
    "mnli": ['input_ids', 'attention_mask', 'labels'],
    "pos": ['input_ids', 'attention_mask', 'labels'],
}

def load_features_dict(tokenizer, cfg):
    # Returns features dict keyed by task, and then by phase
    features_dict = {}
    # I don't think we need this, but I don't know the API for direct cache loading atm.
    dataset_dict = {
        "sts_b": lambda : nlp.load_dataset('glue', name="stsb", cache_dir="/srv/share/svanga3/bert-representations/nlp_datasets/glue_data/stsb"),
        "sst_2": lambda : nlp.load_dataset('glue', name="sst2", cache_dir="/srv/share/svanga3/bert-representations/nlp_datasets/glue_data/sst2"),
        "mnli": lambda : nlp.load_dataset('glue', name="mnli", cache_dir="/srv/share/svanga3/bert-representations/nlp_datasets/glue_data/MNLI"),
        "pos": lambda : nlp.load_dataset('conll2003', cache_dir="/srv/share/svanga3/bert-representations/nlp_datasets/POS/"),
    }
    for task_name in cfg.TASK.TASKS:
        dataset = dataset_dict[task_name]()
        features_dict[task_name] = {}
        for phase, phase_dataset in dataset.items():
            features_dict[task_name][phase] = phase_dataset.map(
                lambda x: convert_func_dict[task_name](cfg, tokenizer, x),
                batched=True,
                load_from_cache_file=True,
                cache_file_name=f"/srv/share/svanga3/bert-representations/nlp_datasets/cached_batches/{task_name}_{phase}.cache"
            )
            print(task_name, phase, len(phase_dataset), len(features_dict[task_name][phase]))
            features_dict[task_name][phase].set_format(
                type="torch",
                columns=columns_dict[task_name],
            )
            # print(task_name, phase, len(phase_dataset), len(features_dict[task_name][phase]))
    return features_dict