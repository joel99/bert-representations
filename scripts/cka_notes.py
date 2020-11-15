#%%
#%%
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
ALLOCATED_DEVICE_ID = 0
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
    get_metric, normalize_scores,
    pretty_print,
    prep_plt,
    get_repr_from_fn, get_repr, get_layer_similarity,
    get_avg_transfer
)
#%%
vivek_template = "/srv/share/svanga3/bert-representations/{}/extracted/checkpoint-{}.npy"

#%%
extract_template = "/srv/share/jye72/bert-representations/{}/extract_{}/extracted/checkpoint-{}.npy"
def get_ckpt(task):
    if task == "mnli":
        return 6000
    if task == "base":
        return 1
    return 2000

def get_target_domain_similarity(source, target):
    source_fn = extract_template.format(source, target, get_ckpt(source))
    target_fn = extract_template.format(target, target, get_ckpt(target))
    source_rep = get_repr_from_fn(source_fn, device=device)
    target_rep = get_repr_from_fn(target_fn, device=device)
    return get_layer_similarity(source_rep, target_rep)

def get_task_self_similarity(task):
    # Similarity wrt noise -- using Vivek's runs
    if task in [mnli, pos]:
        checkpoint = 6000
    elif task == sts_b:
        checkpoint = 2000
    elif task == sst_2:
        checkpoint = 4000
    elif task == base:
        checkpoint = 1
    task_rep_1 = get_repr(task, checkpoint, template=vivek_template, device=device)
    task_rep_2 = get_repr(task, checkpoint, template=vivek_template, device=device)
    return get_layer_similarity(task_rep_1, task_rep_2)

def get_task_CKA_similarity(source, target):
    # Load source and target repr
    # Load source seed similarity
    # Calculate transfer similarity normalized by self-similarity
    transfer_sim = get_target_domain_similarity(source, target)
    self_sim = get_task_self_similarity(target)
    delta = transfer_sim - self_sim
    # Simply add - don't take the norm because relevant content may be captured in different layers, and we should be giving more similarity points if there's an excess match over self-sim.
    diff = delta.mean() # Since 0 is full alignment, (-1 is none), we add 1 to be on more interpretable 0-1 scale
    return diff + 1

#%%
# Visually motivate the method
prep_plt()
# sim = get_task_self_similarity(sst_2)
# sim = get_task_self_similarity(sts_b)
sim = get_task_self_similarity(pos)
# sim = get_task_self_similarity(mnli)
ax = sns.heatmap(sim, cbar=False)
fig = plt.gcf()
fig.set_size_inches(6, 6)
ax.axis('off')
ax.invert_yaxis()
plt.title("POS Self-similarity (2 Seeds)")
plt.savefig('test.png', dpi=300)

#%%
prep_plt()
def plot_two_sim(src, tar):
    sim = get_target_domain_similarity(src, tar)
    # sim = get_task_self_similarity(tar)
    ax = sns.heatmap(sim, cbar=False)
    ax.invert_yaxis()
    fig = plt.gcf()
    fig.set_size_inches(6, 6)
    ax.axis('off')
    plt.title(f"{pretty_print(src)} ~ {pretty_print(tar)} Reprs of {pretty_print(tar)} Tokens")
    # ! Labels might be flipped, nbd
    ax.set_xlabel(f"{pretty_print(src)} Layer")
    ax.set_ylabel(f"{pretty_print(tar)} Layer")
    plt.savefig('test.png', dpi=300)

plot_two_sim(mnli, pos)
print(get_task_CKA_similarity(mnli, pos))

#%%
# Describe the method visually
transfer_sim = get_target_domain_similarity(mnli, pos)
self_sim = get_task_self_similarity(pos)
delta = transfer_sim - self_sim
ax = sns.heatmap(delta, cbar=False)
ax.invert_yaxis()
fig = plt.gcf()
fig.set_size_inches(6, 6)
ax.axis('off')
plt.title("Diff")
plt.savefig('test.png', dpi=300)

#%%
# Heatmap of CKA-based similarity between tasks
prep_plt()
sim = np.zeros((len(SOURCES), len(TARGETS)))
for s_i, source in enumerate(SOURCES):
    for t_j, target in enumerate(TARGETS):
        if source == target:
            sim[s_i, t_j] = 1.0
        else:
            sim[s_i, t_j] = get_task_CKA_similarity(source, target)
#%%
ax = sns.heatmap(sim) #, vmin=0.7, vmax=1.0)
ax.set_xticklabels(pretty_print(TARGETS))
ax.set_yticklabels(pretty_print(SOURCES), rotation=20)
plt.title("CKA Similarities")
ax.invert_yaxis()
plt.savefig("test.png", dpi=300, bbox_inches="tight")

#%%

avg_transfer = get_avg_transfer(sim)
print([f"{pretty_print(s)}: {t:.3f}" for s, t in zip(SOURCES, avg_transfer)])

#%%
torch.save(sim, "cka_sim.pth")