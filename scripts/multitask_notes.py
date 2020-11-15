#%%
# Multitask models are typically motivated as trunk/branch, because one monolithic architecture will make it difficult to learn representations relevant to every downstream task.
# Given sufficient data/compute, we could learn individual branches.
# But we don't generally have this. Given limited model capacity for multitask learning, what is the best way to combine the tasks?
# Specifically, which combination of leads to best overall downstream task performance, given a fixed number of "heads" (transformer branches)

# We haven't really looked into this.
import os
import os.path as osp
import sys
module_path = os.path.abspath(osp.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

# ! SET YOUR DEVICE HERE
ALLOCATED_DEVICE_ID = 3
os.environ["CUDA_VISIBLE_DEVICES"] = str(ALLOCATED_DEVICE_ID)
import torch

if torch.cuda.device_count() >= 0:
    device = torch.device("cuda", 0)
else:
    device = torch.device("cpu")

from src.utils.common import NUM_BERT_LAYERS
from src.utils import cka
from analysis_utils import (
    sst_2, sts_b, mnli, pos, base,
    SOURCES, TARGETS,
    get_metric, normalize_scores, get_multi_metric,
    pretty_print,
    prep_plt,
    get_repr_from_fn, get_repr, get_layer_similarity,
    get_multi_repr, quick_map
)

#%%
# This is plain sequential. Might be worth analyzing, but not really sure what to say.
def get_variant(arr):
    return "_".join(quick_map(a) for a in arr)
variant = get_variant([mnli, pos, sts_b, sst_2])
ckpts = np.arange(3000, 27000, 3000)

def get_all_metrics(variant, suffix="validation", seq=True, ckpts=ckpts):
    for ckpt in ckpts:
        fn = f'{variant}_checkpoint-{ckpt}_{suffix}.eval'
        metrics = get_multi_metric(fn)
        print(metrics)
    #     task_a_results.append(metrics[task_a])
    #     task_b_results.append(metrics[task_b])
    # task_a_results = np.array(task_a_results)
    # task_b_results = np.array(task_b_results)
    # return normalize_scores(task_a_results, task_a), normalize_scores(task_b_results, task_b)
get_all_metrics(variant)

#%%
# variants = ["pos_sst2_branch"]