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
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader

from tqdm import tqdm
import transformers
from transformers import (
    EvalPrediction,
    glue_compute_metrics,
    PreTrainedTokenizerBase,
    AdamW
)
from transformers.trainer_utils import (
    PredictionOutput
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
    'sts_b': lambda p: compute_glue_eval_metrics_regression('sts-b', p),
    'pos': compute_pos_metrics
}

def get_eval_metrics_func(task_key) -> Dict:
    r""" Wrapper for all tasks """
    if task_key in EVAL_METRICS_FUNC_DICT:
        func = EVAL_METRICS_FUNC_DICT[task_key]
        if task_key == "pos":
            _, label_map = pos_label_info()
            return lambda p: func(label_map, p)
        return func
    else:
        task_name = TASK_KEY_TO_NAME[task_key]
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
    padding: Union[bool, str] = True # "max_length" # Is normally true, not sure how they line things up on evaluation in that case.
    # padding: Union[bool, str] = "max_length" # Is normally true, not sure how they line things up on evaluation in that case.
    max_length: Optional[int] = None
    # max_length: Optional[int] = 128 # Is normally 512
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

def pos_label_info():
    labels = POS_LABELS
    label_map: Dict[int, str] = {i: label for i, label in enumerate(labels)}
    return labels, label_map


# ===
# https://github.com/huggingface/transformers/blob/b93569457fd758a60f15d94ac7b3ba3a245096c0/src/transformers/trainer_pt_utils.py#L75
def torch_pad_and_concatenate(tensor1, tensor2, padding_index=-100):
    """Concatenates `tensor1` and `tensor2` on first axis, applying padding on the second if necessary."""
    if len(tensor1.shape) == 1 or tensor1.shape[1] == tensor2.shape[1]:
        return torch.cat((tensor1, tensor2), dim=0)

    # Let's figure out the new shape
    new_shape = (tensor1.shape[0] + tensor2.shape[0], max(tensor1.shape[1], tensor2.shape[1])) + tensor1.shape[2:]

    # Now let's fill the result tensor
    result = tensor1.new_full(new_shape, padding_index)
    result[: tensor1.shape[0], : tensor1.shape[1]] = tensor1
    result[tensor1.shape[0] :, : tensor2.shape[1]] = tensor2
    return result

def nested_concat(tensors, new_tensors, padding_index=-100):
    """
    Concat the `new_tensors` to `tensors` on the first dim and pad them on the second if needed. Works for tensors or
    nested list/tuples of tensors.
    """
    assert type(tensors) == type(
        new_tensors
    ), f"Expected `tensors` and `new_tensors` to have the same type but found {type(tensors)} and {type(new_tensors)}."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_concat(t, n, padding_index=padding_index) for t, n in zip(tensors, new_tensors))
    elif isinstance(tensors, torch.Tensor):
        return torch_pad_and_concatenate(tensors, new_tensors, padding_index=padding_index)
    # elif isinstance(tensors, np.ndarray):
    #     return numpy_pad_and_concatenate(tensors, new_tensors, padding_index=padding_index)
    else:
        raise TypeError(f"Unsupported type for concatenation: got {type(tensors)}")

class FixedTrainer(transformers.Trainer):
    # fixed LR # TODO and multitask patch
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through :obj:`optimizers`, or subclass and override this method in a subclass.
        """
        if self.optimizer is None:
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            self.optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
            )
        self.lr_scheduler = LambdaLR(self.optimizer, lambda x: 1) # constant


    def prediction_loop(
        self, dataloader: DataLoader, description: str, prediction_loss_only: Optional[bool] = None
    ) -> PredictionOutput:
        """
        Prediction/evaluation loop, shared by :obj:`Trainer.evaluate()` and :obj:`Trainer.predict()`.

        Works both with or without labels.
        """
        prediction_loss_only = (
            prediction_loss_only if prediction_loss_only is not None else self.args.prediction_loss_only
        )

        model = self.model
        # multi-gpu eval
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)
        else:
            model = self.model
        # Note: in torch.distributed mode, there's no point in wrapping the model
        # inside a DistributedDataParallel as we'll be under `no_grad` anyways.

        batch_size = dataloader.batch_size
        eval_losses: List[float] = []
        preds: torch.Tensor = None
        label_ids: torch.Tensor = None
        model.eval()

        if self.args.past_index >= 0:
            self._past = None

        disable_tqdm = not self.is_local_process_zero() or self.args.disable_tqdm
        samples_count = 0
        for inputs in tqdm(dataloader, desc=description, disable=disable_tqdm):
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only)
            batch_size = inputs[list(inputs.keys())[0]].shape[0]
            samples_count += batch_size
            if loss is not None:
                eval_losses.append(loss * batch_size)
            if logits is not None:
                preds = logits if preds is None else nested_concat(preds, logits, padding_index=-100)
            if labels is not None:
                label_ids = labels if label_ids is None else nested_concat(label_ids, labels, padding_index=-100)

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        if self.args.local_rank != -1:
            # In distributed mode, concatenate all results from all nodes:
            if preds is not None:
                preds = self.distributed_concat(preds, num_total_examples=self.num_examples(dataloader))
            if label_ids is not None:
                label_ids = self.distributed_concat(label_ids, num_total_examples=self.num_examples(dataloader))

        # Finally, turn the aggregated tensors into numpy arrays.
        if preds is not None:
            preds = preds.cpu().numpy()
        if label_ids is not None:
            label_ids = label_ids.cpu().numpy()

        if self.compute_metrics is not None and preds is not None and label_ids is not None:
            metrics = self.compute_metrics(EvalPrediction(predictions=preds, label_ids=label_ids))
        else:
            metrics = {}
        if len(eval_losses) > 0:
            metrics["eval_loss"] = np.sum(eval_losses) / samples_count

        # Prefix all keys with eval_
        for key in list(metrics.keys()):
            if not key.startswith("eval_"):
                metrics[f"eval_{key}"] = metrics.pop(key)

        return PredictionOutput(predictions=preds, label_ids=label_ids, metrics=metrics)



# https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-subobjects-chained-properties/31174427#31174427
import functools

def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)

# using wonder's beautiful simplification: https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects/31174427?noredirect=1#comment86638618_31174427

def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))