#!/usr/bin/env python3

import os
import glob
import os.path as osp

import socket
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
from transformers import (
    EvalPrediction,
    glue_compute_metrics,
)

# Some utils for sequential training
def find_most_recent_path(base):
    r""" Gets most recent subfile."""
    list_of_files = glob.glob(f"{base}/*")
    return max(list_of_files, key=osp.getctime)

def find_data_path(base, task_name):
    r""" Find subdirectory that matches task"""
    list_of_paths = glob.glob(f"{base}/*")
    list_of_files = [osp.split(s)[1] for s in list_of_paths]
    list_of_lower_files = [s.lower() for s in list_of_files]
    index = list_of_lower_files.index(task_name.lower())
    return list_of_paths[index]

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

# * Unused
def default_tbdir() -> str:
    """
    Same default as PyTorch
    """
    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    return os.path.join("runs", current_time + "_" + socket.gethostname())

def compute_glue_eval_metrics(task_name: str, p: EvalPrediction) -> Dict:
    preds = np.argmax(p.predictions, axis=1)
    return glue_compute_metrics(task_name, preds, p.label_ids)

def compute_glue_eval_metrics_regression(task_name: str, p: EvalPrediction) -> Dict:
    preds = np.squeeze(p.predictions)
    return glue_compute_metrics(task_name, preds, p.label_ids)


EVAL_METRICS_FUNC_DICT = {
    'sts-b': lambda p: compute_glue_eval_metrics_regression('sts-b', p)
}

def get_eval_metrics_func(task_name) -> Dict:
    r""" Wrapper for all tasks """
    if task_name in EVAL_METRICS_FUNC_DICT:
        return EVAL_METRICS_FUNC_DICT[task_name]
    else:
        return lambda p: compute_glue_eval_metrics(task_name, p)


# For data args

TASK_KEY_TO_NAME = {
    "mnli": "mnli",
    "sts_b": "sts-b",
    "sst_2": "sst-2",
    "pos": "POS",
    # ! ADD DP @ Ayush
}