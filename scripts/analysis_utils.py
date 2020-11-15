import os
import os.path as osp

import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.utils.common import POS_LABELS, NUM_BERT_LAYERS
from src.utils import cka

sst_2 = "sst_2"
sts_b = "sts_b"
pos = "pos"
mnli = "mnli"
base = "base"

SOURCES = [
    sst_2, sts_b, pos, mnli, base
]
TARGETS = [
    sst_2, sts_b, pos, mnli
]

metric_key = {
    sst_2: "eval_acc",
    sts_b: "eval_pearson",
    pos: "eval_accuracy_score",
    mnli: "eval_mnli/acc"
}

def metric_sans_eval(task):
    return metric_key[task][5:]

NAME_TO_KEY = {
    "POS": pos,
    "mnli": mnli,
    "sst-2": sst_2,
    "sts-b": sts_b
}

AT_CHANCE_RESULTS = {
    mnli: 1.0 / 3,
    sst_2: 0.5,
    pos: 1 / len(POS_LABELS),
    sts_b: 0.0, # uncorrelated
}

SINGLE_TASK_EVALS = "{}_checkpoint-6000_validation.eval"
eval_dir = "../eval"

def get_metric(filepath, task):
    info = torch.load(osp.join(eval_dir, filepath))
    return info[metric_key[task]]

def get_multi_metric(filepath, eval_dir="../eval"):
    info = torch.load(osp.join(eval_dir, filepath))
    return {NAME_TO_KEY[k]: v[metric_sans_eval(NAME_TO_KEY[k])] for k, v in info.items()}

def get_normalization_range(task):
    # Normalize such that worst is at-chance, best is pretrain->fine-tune.
    # We take "single task performance" as training each task for 6K epochs.
    best = get_metric(SINGLE_TASK_EVALS.format(task), task)
    return (AT_CHANCE_RESULTS[task], best)

def normalize_scores(scores, task):
    norm = get_normalization_range(task)
    return (scores - norm[0]) / (norm[1] - norm[0])

PRETTY_PRINT = {
    mnli: "MNLI",
    sst_2: "SST-2",
    sts_b: "STS-B",
    pos: "POS",
    base: "Base"
}

def pretty_print(tasks):
    if isinstance(tasks, str):
        return PRETTY_PRINT[tasks]
    return [PRETTY_PRINT[t] for t in tasks]

SMALL_SIZE = 12
MEDIUM_SIZE = 15
LARGE_SIZE = 18

def prep_plt(spine_alpha=1.0):
    plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
    plt.rc('axes', labelsize=LARGE_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    # plt.rc('title', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels

    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.style.use('seaborn-muted')
    # plt.figure(figsize=(6,4))

    spine_alpha = 0.5
    plt.gca().spines['right'].set_alpha(0.0)
    plt.gca().spines['bottom'].set_alpha(spine_alpha)
    # plt.gca().spines['bottom'].set_alpha(0)
    plt.gca().spines['left'].set_alpha(spine_alpha)
    # plt.gca().spines['left'].set_alpha(0)
    plt.gca().spines['top'].set_alpha(0.0)

    plt.tight_layout()


def get_repr_from_fn(fn, device=None):
    return torch.from_numpy(
        np.load(fn)
    ).float().to(device)

repr_template = "/srv/share/jye72/bert-representations/{}/extracted/checkpoint-{}.npy"

def get_repr(task, ckpt, template=repr_template, **kwargs):
    return get_repr_from_fn(template.format(task, ckpt), **kwargs)

def get_layer_similarity(x, y):
    local_sim = np.zeros((NUM_BERT_LAYERS, NUM_BERT_LAYERS))
    for dim_x in range(NUM_BERT_LAYERS):
        for dim_y in range(NUM_BERT_LAYERS):
            X_1 = x[:, dim_x, :].squeeze()
            Y_1 = y[:, dim_y, :].squeeze()
            local_sim[dim_x, dim_y] = cka(X_1, Y_1).cpu().item()
    return local_sim
