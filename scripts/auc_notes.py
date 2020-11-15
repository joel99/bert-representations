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

from src.utils.common import NUM_BERT_LAYERS
from analysis_utils import (
    sst_2, sts_b, mnli, pos, base,
    SOURCES, TARGETS,
    get_metric, normalize_scores,
    pretty_print,
    prep_plt,
    get_avg_transfer
)

freeze_template = "target_{}-freeze_{}-{}_checkpoint-{}_validation.eval"

def get_transfer_metric(source, target, layer):
    # target -- target domain
    # source -- training domain
    # layer -- max layers frozen (0-indexed)
    if source == base:
        checkpoint = 1
    else:
        checkpoint = 6000
    fn = freeze_template.format(target, layer, source, checkpoint)
    return get_metric(fn, target)
# print(get_transfer_metric(base, pos, 1))
#%%
palette = sns.color_palette(palette='muted', n_colors=len(SOURCES), desat=0.9)
key_colors = {}
for i, key in enumerate(SOURCES):
    key_colors[key] = palette[i]

x = np.arange(NUM_BERT_LAYERS)
x_labels = x + 1

def get_frozen_scores(source, target):
    frozen_metrics = [get_transfer_metric(source, target, l) for l in x]
    return normalize_scores(np.array(frozen_metrics), target)

def get_frozen_auc(source, target):
    # A metric for task transferability such that:
    # 0.0 -- source features can't be transfered at every layer
    # 1.0 -- source features as as well aligned to target as reference (e.g. target -> target AuC = 1.0) (we use single-task training as reference)
    # Ideally, this would be somemwhat normalized... why is MNLI so bad? Shouldn't the base reflect perfectly?
    # * In our experiments, this transfer score is CONDITIONED on MLM pretraining.
    scores = get_frozen_scores(source, target)
    auc = metrics.auc(np.arange(0,1, 1.0/len(scores)), scores)
    return auc

def get_unfrozen_score(source, target):
    # Note, we didn't carefully run with no frozen layers at all, so we use 0th layer frozen as proxy
    return np.array(normalize_scores(get_transfer_metric(source, target, 0), target))

def plot_frozen(source, target, ax):
    scores = get_frozen_scores(source, target)
    ax.scatter(x_labels, scores, color=key_colors[source])
    ax.plot(x_labels, scores, label=f"{pretty_print(source)}", color=key_colors[source])

f, axes = plt.subplots(4, sharex=True, figsize=(6, 6))

def plot_series(target, ax):
    for key in SOURCES:
        if key == target:
            continue
        plot_frozen(key, target, ax)
    ax.set_title(f"Target domain: {pretty_print(target)}")

plot_series(pos, axes[0])
plot_series(mnli, axes[1])
plot_series(sts_b, axes[2])
plot_series(sst_2, axes[3])

axes[0].legend()
plt.tight_layout()
plt.show()

#%%
# We only want to give MNLI as a more clear example
prep_plt()
plot_series(mnli, plt.gca())
plt.legend()
plt.xlabel("Layers frozen")
plt.ylabel("Normalized task performance")
plt.savefig("test.png", dpi=300)

#%%
# Motivation: Regular fine-tuning doesn't give any signal.
prep_plt()
# Heatmap of normalized scores, without any freezing
sim = np.zeros((len(SOURCES), len(TARGETS)))
for s_i, source in enumerate(SOURCES):
    for t_j, target in enumerate(TARGETS):
        if source == target:
            sim[s_i, t_j] = 1.0
        else:
            sim[s_i, t_j] = get_unfrozen_score(source, target)
# ax = sns.heatmap(sim)
ax = sns.heatmap(sim, vmin=0.7, vmax=1.0)
ax.set_xticklabels(pretty_print(TARGETS))
ax.set_yticklabels(pretty_print(SOURCES), rotation=20)
ax.set_title("Fine-Tuning Transfer, no freezing")
ax.invert_yaxis()
plt.savefig("test.png", dpi=300, bbox_inches="tight")


#%%
# Approach
# Heatmap of task transferability
prep_plt()
sim = np.zeros((len(SOURCES), len(TARGETS)))
for s_i, source in enumerate(SOURCES):
    for t_j, target in enumerate(TARGETS):
        if source == target:
            sim[s_i, t_j] = 1.0
        else:
            sim[s_i, t_j] = get_frozen_auc(source, target)
ax = sns.heatmap(sim, vmin=0.7, vmax=1.0)
ax.set_xticklabels(pretty_print(TARGETS))
ax.set_yticklabels(pretty_print(SOURCES), rotation=20)
plt.title("Fine-Tuning Transfer, freezing AuC")
ax.invert_yaxis()
plt.savefig("test.png", dpi=300, bbox_inches="tight")


# Observations -- what does AuC tell us?
# 1. MNLI <-> STS_B well, makes sense given task similarity
# 2. Most transfer well to POS, but not vice versa => supports intuitive understanding that POS is a more specialized task

avg_transfer = get_avg_transfer(sim)

print([f"{pretty_print(s)}: {t:.3f}" for s, t in zip(SOURCES, avg_transfer)])
# 3. Suggests MNLI is most useful, supporting it's use as an intermediate STILTS task
torch.save(sim, "auc_sim.pth")

#%%
# Scatter cka vs auc for each pair
from sklearn.metrics import explained_variance_score, r2_score

auc_sim = torch.tensor(torch.load("auc_sim.pth"))
cka_sim = torch.tensor(torch.load("cka_sim.pth"))
blacklist_diag = [0, 5, 10, 15] # diag indices
blacklist_mask = torch.ones_like(auc_sim).flatten().bool()
blacklist_mask[blacklist_diag] = 0
auc_points = torch.masked_select(auc_sim.flatten(), blacklist_mask)
cka_points = torch.masked_select(cka_sim.flatten(), blacklist_mask)

plt.scatter(auc_points, cka_points)
print(r2_score(auc_points, cka_points, multioutput='uniform_average'))