#!/usr/bin/env python3

import os
import glob
import os.path as osp

import socket
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, Optional, Union, Tuple, List
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score

import numpy as np
import torch
import torch.nn as nn
from transformers import (
    EvalPrediction,
    glue_compute_metrics,
    PreTrainedTokenizerBase
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

def compute_pos_metrics(label_map, p: EvalPrediction) -> Dict:
    # label_map will be bound in run script
    # TODO maybe refactor ^
    def align_predictions(predictions: np.ndarray, label_ids: np.ndarray) -> Tuple[List[int], List[int]]:
        # Token predictions need to be reduced to word predictions
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
    preds_list, out_label_list = align_predictions(p.predictions, p.label_ids)
    return {
        "accuracy_score": accuracy_score(out_label_list, preds_list),
        "precision": precision_score(out_label_list, preds_list),
        "recall": recall_score(out_label_list, preds_list),
        "f1": f1_score(out_label_list, preds_list),
    }


EVAL_METRICS_FUNC_DICT = {
    'sts-b': lambda p: compute_glue_eval_metrics_regression('sts-b', p),
    'POS': compute_pos_metrics
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
}

# POS_LABELS = [
#     "ADJ",
#     "ADP",
#     "ADV",
#     "AUX",
#     "CCONJ",
#     "DET",
#     "INTJ",
#     "NOUN",
#     "NUM",
#     "PART",
#     "PRON",
#     "PROPN",
#     "PUNCT",
#     "SCONJ",
#     "SYM",
#     "VERB",
#     "X",
# ]
POS_LABELS = ['"', # A more fine-grained POS, from NLP
 '$',
 "''",
 '(',
 ')',
 ',',
 '.',
 ':',
 'CC',
 'CD',
 'DT',
 'EX',
 'FW',
 'IN',
 'JJ',
 'JJR',
 'JJS',
 'LS',
 'MD',
 'NN',
 'NNP',
 'NNPS',
 'NNS',
 'NN|SYM',
 'PDT',
 'POS',
 'PRP',
 'PRP$',
 'RB',
 'RBR',
 'RBS',
 'RP',
 'SYM',
 'TO',
 'UH',
 'VB',
 'VBD',
 'VBG',
 'VBN',
 'VBP',
 'VBZ',
 'WDT',
 'WP',
 'WP$',
 'WRB']


# From 3.4.0 https://github.com/huggingface/transformers/blob/master/src/transformers/data/data_collator.py#L119
@dataclass
class DataCollatorForTokenClassification:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.
    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignore by PyTorch loss functions).
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str] = True # dropped paddingstrategy
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            # Conversion to tensors will fail if we have labels as they are not of the same length yet.
            return_tensors="pt" if labels is None else None,
        )

        if labels is None:
            return batch

        sequence_length = torch.tensor(batch["input_ids"]).shape[1]
        padding_side = self.tokenizer.padding_side
        if padding_side == "right":
            batch["labels"] = [label + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels]
        else:
            batch["labels"] = [[self.label_pad_token_id] * (sequence_length - len(label)) + label for label in labels]
        batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items()}
        return batch