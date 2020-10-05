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

def align_predictions(predictions: np.ndarray, label_ids: np.ndarray) -> Tuple[List[int], List[int]]:
    preds = np.argmax(predictions, axis=2)

    batch_size, seq_len = preds.shape

    out_label_list = [[] for _ in range(batch_size)]
    preds_list = [[] for _ in range(batch_size)]

    for i in range(batch_size):
        for j in range(seq_len):
            if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index:
                out_label_list[i].append(label_map[label_ids[i][j]])
                preds_list[i].append(label_map[preds[i][j]])

    return preds_list, out_label_list

def compute_metrics_pos(p: EvalPrediction) -> Dict:
    preds_list, out_label_list = align_predictions(p.predictions, p.label_ids)
    return {
        "accuracy_score": accuracy_score(out_label_list, preds_list),
        "precision": precision_score(out_label_list, preds_list),
        "recall": recall_score(out_label_list, preds_list),
        "f1": f1_score(out_label_list, preds_list),
    }

EVAL_METRICS_FUNC_DICT = {
    'sts-b': lambda p: compute_glue_eval_metrics_regression('sts-b', p)
    'pos': lambda p: compute_metrics_pos(p)
}

def get_eval_metrics_func(task_name) -> Dict:
    r""" Wrapper for all tasks """
    if task_name in EVAL_METRICS_FUNC_DICT:
        return EVAL_METRICS_FUNC_DICT[task_name]
    else:
        return lambda p: compute_glue_eval_metrics(task_name, p)

