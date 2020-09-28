#!/usr/bin/env python3

import os
import socket
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
from transformers import (
    EvalPrediction,
    glue_compute_metrics,
)

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

