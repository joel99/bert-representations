#%%
# Understand how shuffled learning mitigates forgetting (relative to sequential forgetting),
# By examining representation similarities over the course of training for both
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

# SET YOUR DEVICE HERE
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
    get_repr_from_fn, get_repr, get_layer_similarity
)

#%%
# Pull up both domains representations for a given checkpoint
# We can use net shift between diagonals.
template = "/srv/share/jye72/bert-representations/{}/extracted/{}_checkpoint-{}.npy"
def get_multi_repr(variant, task, checkpoint):
    return get_repr_from_fn(template.format(variant, task, checkpoint), device=device)

def quick_map(task):
    if task == "sts_b":
        return "stsb"
    elif task == "sst_2":
        return "sst2"
    return task

def get_all_metrics(task_a, task_b, variant=None, suffix="validation", ckpts=np.arange(500, 2500, 500)):
    task_a_results = []
    task_b_results = []
    for ckpt in ckpts:
        if variant is None:
            variant = f'{quick_map(task_a)}_{quick_map(task_b)}_eq-seq'
        fn = f'{variant}_checkpoint-{ckpt}_{suffix}.eval'
        metrics = get_multi_metric(fn)
        task_a_results.append(metrics[task_a])
        task_b_results.append(metrics[task_b])
    task_a_results = np.array(task_a_results)
    task_b_results = np.array(task_b_results)
    return normalize_scores(task_a_results, task_a), normalize_scores(task_b_results, task_b)

def get_reprs(task_a, task_b, seq=True, checkpoint=500):
    seq = True
    variant = f"{quick_map(task_a)}_{quick_map(task_b)}_eq-{'seq' if seq else 'sam'}"
    reprs_a = get_multi_repr(variant, task_a, checkpoint)
    reprs_b = get_multi_repr(variant, task_b, checkpoint)
    return reprs_a, reprs_b

#%%
def plot_repr_sim(src, tar, ckpt, ax=plt.gca(), is_src=True):
    ref_str = src if is_src else tar
    reference = get_repr(ref_str, 6000 if ref_str == mnli else 2000, device=device)
    value = get_reprs(src, tar, checkpoint=ckpt)
    sim = get_layer_similarity(reference, value[0 if is_src else 1])
    sns.heatmap(sim, vmin=0.0, vmax=1.0, ax=ax, cbar=False)
    ax.invert_yaxis()
    ax.axis('off')
    # ax.set_ylabel(f"{pretty_print(src)} Layer")
    # ax.set_xlabesl(f"{pretty_print(tar)} Layer")
    # ax.set_title(f"Checkpoint {ckpt}")

def plot_super(src, tar, updates, is_src=True):
    ref_str = src if is_src else tar
    f, axes = plt.subplots(1, len(updates), figsize=(len(updates) * 1, 1), sharex=True, sharey=True)
    axes = axes.flatten()
    for i, c in enumerate(updates):
        plot_repr_sim(src, tar, c, ax=axes[i], is_src=is_src)
    plt.tight_layout()
    plt.suptitle(f"Representations of {pretty_print(ref_str)} Tokens", y=1.1)
    f.add_subplot(111, frame_on=False)
    plt.tick_params(labelcolor="none", bottom=False, left=False)
    plt.xlabel(f"Active ({pretty_print(src)} -> {pretty_print(tar)}) Layers")
    plt.ylabel(f"Ref ({pretty_print(ref_str)}) Layers")
    # plt.savefig("test.png", dpi=300, bbox_inches="tight")

updates = np.arange(1000, 2500, 500)
src = pos
tar = sst_2

updates = np.arange(2000, 16000, 2000)
src = mnli
tar = sst_2

plot_super(src, tar, updates, is_src=True)

"""
TODO:
Integrate the CF plots
Do runs with more granularity
Check the other variants
Obs:
STS-B -> POS:
- we already know top layers change a lot
- in combination with the knowledge that intermediate layers hold POS syntax, we see that we essentially drop the higher level knowledge, and POS is being pulled upwards.

SST_2 -> POS:
- Similar, but slightly more higher levels are used in SST_2 (supporting STS being a harder task), and layer separation is *clean*
(Note that there's a lot of forgetting here, but not vice versa)

POS -> SST_2:
- Little CKA shift implies no forgetting. Huh.
- Reflection:
- 1. shortsighted to say a model is forgetting something just because it's not at the top layer -- we need a different metric
- 2. we need a different fine-tuning procedure that will learn to not overwrite top layers if it's not necessary.

SST_2 -> MNLI:

"""

#%%
prep_plt()
def plot_cf(src, tar, **kwargs):
    a_res, b_res = get_all_metrics(src, tar, **kwargs)
    x = np.linspace(0, 1, len(a_res))
    plt.plot(x, a_res, label=pretty_print(src))
    plt.scatter(x, a_res)
    plt.plot(x, b_res, label=pretty_print(tar))
    plt.ylabel("Normalized Task Metric")
    plt.scatter(x, b_res)
    plt.legend()
    plt.savefig("test.png", dpi=300)
# plot_cf(pos, sst_2)
# plot_cf(sst_2, pos)
plot_cf(sst_2, mnli, variant="sst2_mnli_long", suffix="validation_matched", ckpts=np.arange(1000, 11000, 1000))


#%%
# In domain out domain replica
ins = []
updates = np.arange(1000, 12000, 1000)
palette = sns.color_palette(palette='muted', n_colors=len(updates), desat=0.9)
def plot_in_out(src, tar, update, is_src=True):
    ref_str = src if is_src else tar
    plt.gcf().set_size_inches(6, 6)
    for i, c in enumerate(updates):
        ref_str = src if is_src else tar
        reference = get_repr(ref_str, 6000 if ref_str == mnli else 2000, device=device)
        value = get_multi_repr(f"{quick_map(src)}_{tar}_long", ref_str, checkpoint=c)
        # value = get_reprs(src, tar, checkpoint=c)
        sim = torch.tensor(get_layer_similarity(reference, value))
        diag_sim = torch.diag(sim)
        progress = float(i) / len(updates)
        plt.plot(np.linspace(0, 1, diag_sim.size(0)), diag_sim, color=palette[i], label=f"{ref_str} {progress:.2f}", linestyle="--" if is_src else "-")
    plt.tight_layout()
    plt.suptitle(f"Tuning: New domain ({tar}) and Old domain ({src})", y=1.1)
    # Representations of {pretty_print(ref_str)} Tokens", y=1.1)
    plt.tick_params(labelcolor="none", bottom=False, left=False)
    plt.xlabel(f"Attention Layer")
    plt.ylabel(f"CKA")
    plt.legend(loc=(1.05, 0.0))
    # plt.savefig("test.png", dpi=300, bbox_inches="tight")

# As we keep trainng, the dotted lines (sst2 old domain) mainly drop in the higher layers. I don't really interpret any overfitting though...
# Forgetting is very robust -- still highly dependent on old
src = sst_2
tar = mnli
plot_in_out(src, tar, updates)
plot_in_out(src, tar, updates, is_src=False)
