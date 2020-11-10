#!/usr/bin/env python3
# Registry e.g. for config
# * We use this abstraction so that multi-task models can be init-ed simply

from typing import Dict, List, Optional, Tuple
from yacs.config import CfgNode as CN

import numpy as np
import torch.nn as nn

import transformers
from transformers import AutoConfig, glue_tasks_num_labels
import datasets as nlp

from src.utils import (
    logger,
    ModelArguments,
    TASK_KEY_TO_NAME
)
from src.utils.common import POS_LABELS

def get_glue_config(cfg: CN, model_args, name: str):
    num_labels = glue_tasks_num_labels[name]
    logger.info(f"Num {name} Labels: \t {num_labels}")

    return (AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=name
    ),)

def get_pos_config(cfg: CN, model_args, *args):
    labels = POS_LABELS
    label_map: Dict[int, str] = {i: label for i, label in enumerate(labels)}
    num_labels = len(labels)
    print(f"labels:{labels}")

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        id2label=label_map,
        label2id={label: i for i, label in enumerate(labels)},
        cache_dir=model_args.cache_dir,
    )
    return config, labels, label_map

CONFIG_MAP = {
    "pos": get_pos_config,
    "mnli": get_glue_config,
    "sts_b": get_glue_config,
    "sst_2": get_glue_config,
}

def get_config(task_key: str, cfg: CN):
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
        model_cfg = get_config(task_key, cfg)[0]
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
        inputs, max_length=cfg.MODEL.MAX_LENGTH, padding=True, truncation=True,
    )
    features["labels"] = example_batch["label"]
    return features

def convert_to_mnli_features(cfg, tokenizer, example_batch):
    inputs = list(zip(example_batch['hypothesis'], example_batch['premise']))
    features = tokenizer.batch_encode_plus(
        inputs, max_length=cfg.MODEL.MAX_LENGTH, padding=True, truncation=True,
    )
    features["labels"] = example_batch["label"]
    return features

def convert_to_sst2_features(cfg, tokenizer, example_batch):
    features = tokenizer.batch_encode_plus(
        example_batch["sentence"], max_length=cfg.MODEL.MAX_LENGTH, padding=True, truncation=True,
    )
    features["labels"] = example_batch["label"]
    return features


def convert_to_pos_features(cfg, tokenizer, examples):
    # Reference: https://github.com/huggingface/transformers/blob/master/examples/token-classification/run_ner.py
    # padding = "max_length" # if data_args.pad_to_max_length else False
    # Labeling (this part will be easier when https://github.com/huggingface/datasets/issues/797 is solved)
    # def get_label_list(labels):
    #     unique_labels = set()
    #     for label in labels:
    #         unique_labels = unique_labels | set(label)
    #     label_list = list(unique_labels)
    #     label_list.sort()
    #     return label_list
    # label_list = get_label_list(datasets["train"][label_column_name])
    label_list = POS_LABELS
    label_to_id = {l: i for i, l in enumerate(label_list)}
    tokenized_inputs = tokenizer(
        examples["words"],
        # max_length=cfg.MODEL.MAX_LENGTH,
        padding=False, # It'll get padded in the collator
        truncation=True,
        # We use this argument because the texts in our dataset are lists of words (with a label for each word).
        is_pretokenized=True, # 3.1.0
        # is_split_into_words=True, # 3.4.0
        return_offsets_mapping=True,
    )
    offset_mappings = tokenized_inputs.pop("offset_mapping")
    labels = []
    for label, offset_mapping in zip(examples["pos"], offset_mappings):
        label_index = 0
        current_label = -100
        label_ids = []
        for offset in offset_mapping:
            # We set the label for the first token of each word. Special characters will have an offset of (0, 0)
            # so the test ignores them.
            if offset[0] == 0 and offset[1] != 0:
                current_label = label_to_id[label[label_index]]
                label_index += 1
                label_ids.append(current_label)
            # For special tokens, we set the label to -100 so it's automatically ignored in the loss function.
            elif offset[0] == 0 and offset[1] == 0:
                label_ids.append(-100)
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                # label_ids.append(current_label if data_args.label_all_tokens else -100)
                label_ids.append(-100)

        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs
    # These are returned untensorized,

columns_dict = {
    "sst_2": ['input_ids', 'attention_mask', 'labels'],
    "sts_b": ['input_ids', 'attention_mask', 'labels'],
    "mnli": ['input_ids', 'attention_mask', 'labels'],
    "pos": ['input_ids', 'attention_mask', 'labels'],
}

def load_features_dict(tokenizer, cfg):
    convert_func_dict = {
        "sts_b": convert_to_stsb_features, # Ok, either use theirs, or ours. Um.
        "sst_2": convert_to_sst2_features,
        "mnli": convert_to_mnli_features,
        "pos": convert_to_pos_features,
    }
    # Returns features dict keyed by task (task_key), and then by phase
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
            # need to make sure the irrelevant columns are dropped
            print(task_name, phase, len(phase_dataset), len(features_dict[task_name][phase]))
            # TokenCollator doesn't play well with pre-cast tensors
            features_dict[task_name][phase].set_format(
                type="torch" if task_name != "pos" else None,
                columns=columns_dict[task_name],
            )

            # print(task_name, phase, len(phase_dataset), len(features_dict[task_name][phase]))
    return features_dict