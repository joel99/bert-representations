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

def get_all_metrics(task_a, task_b, variant=None, suffix="validation", seq=True, ckpts=np.arange(500, 2500, 500)):
    task_a_results = []
    task_b_results = []
    for ckpt in ckpts:
        if variant is None:
            variant = f"{quick_map(task_a)}_{quick_map(task_b)}_eq-{'seq' if seq else 'sam'}"
        fn = f'{variant}_checkpoint-{ckpt}_{suffix}.eval'
        metrics = get_multi_metric(fn)
        task_a_results.append(metrics[task_a])
        task_b_results.append(metrics[task_b])
    task_a_results = np.array(task_a_results)
    task_b_results = np.array(task_b_results)
    return normalize_scores(task_a_results, task_a), normalize_scores(task_b_results, task_b)

def get_reprs(task_a, task_b, seq=True, checkpoint=500):
    variant = f"{quick_map(task_a)}_{quick_map(task_b)}_eq-{'seq' if seq else 'sam'}"
    reprs_a = get_multi_repr(variant, task_a, checkpoint)
    reprs_b = get_multi_repr(variant, task_b, checkpoint)
    return reprs_a, reprs_b

#%%
def plot_repr_sim(src, tar, ckpt, ax=plt.gca(), is_src=True, **kwargs):
    ref_str = src if is_src else tar
    reference = get_repr(ref_str, 6000 if ref_str == mnli else 2000, device=device)
    value = get_reprs(src, tar, checkpoint=ckpt, **kwargs)
    sim = get_layer_similarity(reference, value[0 if is_src else 1])
    sns.heatmap(sim, vmin=0.0, vmax=1.0, ax=ax, cbar=False)
    ax.invert_yaxis()
    ax.axis('off')
    # ax.set_ylabel(f"{pretty_print(src)} Layer")
    # ax.set_xlabesl(f"{pretty_print(tar)} Layer")
    # ax.set_title(f"Checkpoint {ckpt}")

def plot_super(src, tar, updates, is_src=True, **kwargs):
    ref_str = src if is_src else tar
    f, axes = plt.subplots(1, len(updates), figsize=(len(updates) * 1.5, 1.5), sharex=True, sharey=True)
    axes = axes.flatten()
    for i, c in enumerate(updates):

        plot_repr_sim(src, tar, c, ax=axes[i], is_src=is_src, **kwargs)
    plt.tight_layout()
    plt.suptitle(f"{pretty_print(ref_str)} Representations", y=1.1)
    f.add_subplot(111, frame_on=False)
    plt.tick_params(labelcolor="none", bottom=False, left=False)
    plt.xlabel(f"Active ({pretty_print(src)} -> {pretty_print(tar)})")
    plt.ylabel(f"Ref ({pretty_print(ref_str)})")
    # plt.savefig("test.png", dpi=300, bbox_inches="tight")

updates = np.arange(1000, 2200, 200)
src = sst_2
tar = pos

src = pos
tar = sst_2

# src = sts_b
# tar = pos


# updates = np.arange(2000, 16000, 2000)
# src = mnli
# tar = sst_2

# plot_super(src, tar, updates, is_src=True, seq=True)

src = sst_2
tar = pos
updates = np.arange(200, 2200, 400)
plot_super(src, tar, updates, is_src=True, seq=False)
"""
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
def plot_cf(src, tar, ax=plt.gca(), **kwargs):
    a_res, b_res = get_all_metrics(src, tar, **kwargs)
    x = np.linspace(0, 1, len(a_res))
    ax.plot(x, a_res, label=pretty_print(src))
    ax.scatter(x, a_res)
    ax.plot(x, b_res, label=pretty_print(tar))
    ax.set_ylabel("Normalized Task Metric")
    ax.scatter(x, b_res)
    ax.legend()
    # plt.savefig("test.png", dpi=300)

src = sst_2
# src = pos
tar = pos
# tar = sst_2
ckpts = np.arange(1000, 2200, 200)
seq = True

# Check shuffling rules
ckpts = np.arange(200, 2200, 200)
seq = False

# plot_cf(src, tar, ckpts=ckpts, seq=seq)
# plt.title(f"{pretty_print(src)} -> {pretty_print(tar)}")
# plot_cf(sst_2, pos, ckpts=np.arange(1000, 2000, 200))
# plot_cf(sts_b, pos, ckpts=np.arange(1000, 2200, 200))

plot_cf(sst_2, mnli, variant="sst2_mnli_long", suffix="validation_matched", ckpts=np.arange(2000, 12000, 2000))


#%%

def plot_ev(src, tar, ckpts, **kwargs):
    num = len(ckpts)
    widths = [1] * num
    heights = [1, 2, 1]
    gs_kw = dict(width_ratios=widths, height_ratios=heights)
    f, all_axs = plt.subplots(nrows=3, ncols=num, figsize=(6, 4), gridspec_kw=gs_kw)
    gs = all_axs[1, 2].get_gridspec()
    # top
    def inner(is_src, axes):
        ref_str = src if is_src else tar
        for i, c in enumerate(ckpts):
            plot_repr_sim(src, tar, c, ax=axes[i], is_src=is_src, **kwargs)
            axes[i].axis('on')
            axes[i].set_xticks([])
            axes[i].set_yticks([])
            for side in ['bottom','right','top']: #,'left']:
                axes[i].spines[side].set_visible(False)
            # axes[i].get_xaxis().set_visible(False)
            # axes[i].get_yaxis().set_visible(False)
        axes[0].set_ylabel(f"{pretty_print(src if is_src else tar)} Ref", rotation=90, labelpad=0, size=14)
    inner(True, all_axs[-1])
    inner(False, all_axs[0])
    # midsection
    for ax in all_axs[1]:
        ax.remove()
    axbig = f.add_subplot(gs[1, :])
    plot_cf(src, tar, ax=axbig, ckpts=ckpts, **kwargs)
    axbig.set_ylabel("Perf")
    axbig.set_xticks([])
    for side in ['bottom','right','top']: #,'left']:
        axbig.spines[side].set_visible(False)
    ax.arrow(0, 0, 1, 0., fc='k', ec='k', lw = 1,
             head_width=0.1, head_length=0.1, overhang = 0.3,
             length_includes_head= True, clip_on = False)
    # axbig.annotate('Big Axes \nGridSpec[1:, -1]', (0.1, 0.5),
    #             xycoords='axes fraction', va='center')
    f.add_subplot(111, frame_on=False)
    plt.tick_params(labelcolor="none", bottom=False, left=False)
    # plt.suptitle(f"({pretty_print(src)} -> {pretty_print(tar)})")
    plt.xlabel(f"{pretty_print(src)} Representations")
    plt.suptitle(f"{pretty_print(tar)} Representations")

# plot_ev(sts_b, pos, ckpts=np.arange(1000, 2000, 200),)
# plot_ev(sst_2, pos, ckpts=np.arange(1000, 2200, 200))

# plot_ev(sst_2, pos, ckpts=np.arange(200, 2200, 400), seq=False)
plot_ev(sts_b, pos, ckpts=np.arange(200, 2200, 400), seq=False)

#%%
# In domain out domain replica
src = sst_2
tar = mnli
updates = np.arange(2000, 14000, 4000)
prep_plt()
palette = sns.color_palette(palette='rocket', n_colors=len(updates), desat=0.9)
def plot_in_out(src, tar, update, is_src=True):
    ref_str = src if is_src else tar
    plt.gcf().set_size_inches(6, 4.5)
    for i, c in enumerate(updates):
        ref_str = src if is_src else tar
        reference = get_multi_repr(f"{quick_map(src)}_{tar}_long", ref_str, checkpoint=2000)
        # reference = get_repr(ref_str, 6000 if ref_str == mnli else 2000, device=device)
        value = get_multi_repr(f"{quick_map(src)}_{tar}_long", ref_str, checkpoint=c)
        # value = get_reprs(src, tar, checkpoint=c)
        sim = torch.tensor(get_layer_similarity(reference, value))
        diag_sim = torch.diag(sim)
        progress = float(i) / len(updates)
        plt.plot(np.linspace(0, 1, diag_sim.size(0)), diag_sim, color=palette[i], label=f"{pretty_print(ref_str)} {progress:.1f}", linestyle="--" if is_src else "-")
    plt.tight_layout()
    # plt.suptitle(f"Changes in CKA representations for {pretty_print(src if is_src else tar)}", y=1)
    plt.suptitle(f"Repr changes in {pretty_print(src)} and {pretty_print(tar)}", y=1)
    # Representations of {pretty_print(ref_str)} Tokens", y=1.1)
    plt.legend(loc="bottom left", ncol=1)
    plt.tick_params(labelcolor="none", bottom=False, left=False)
    plt.xlabel(f"Attention Layer")
    plt.ylabel(f"CKA")
    # plt.savefig("test.png", dpi=300, bbox_inches="tight")

# As we keep trainng, the dotted lines (sst2 old domain) mainly drop in the higher layers. I don't really interpret any overfitting though...
# Forgetting is very robust -- still highly dependent on old

# CKA between training model and base (ckpt 0) on out-domain (SST)
plot_in_out(src, tar, updates)

# CKA between training model and base (ckpt 0) on in-domain (MNLI)
plot_in_out(src, tar, updates, is_src=False)

# Observations: generalization gap does close in upper layers, but not in middle layers
