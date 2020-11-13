#%%
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import os.path as osp
# SET YOUR DEVICE HERE
ALLOCATED_DEVICE_ID = 3
os.environ["CUDA_VISIBLE_DEVICES"] = str(ALLOCATED_DEVICE_ID)
import torch

if torch.cuda.device_count() >= 0:
    device = torch.device("cuda", 0)
else:
    device = torch.device("cpu")
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from src.utils import cka
#%%
a_checkpoint_3 = np.load("/srv/share/svanga3/bert-representations/sst_2/extracted/checkpoint-3500.npy")
b_checkpoint_3 = np.load("/srv/share/svanga3/bert-representations/sst_different_seed/extracted/checkpoint-3500.npy")

#%%

print(device)
X = torch.from_numpy(a_checkpoint_3).float().to(device)
Y = torch.from_numpy(b_checkpoint_3).float().to(device)
sim = np.zeros((12, 12))
for dim_x in range(12):
    for dim_y in range(12):
        X_1 = X[:, dim_x, :].squeeze()
        Y_1 = Y[:, dim_y, :].squeeze()
        sim[dim_x, dim_y] = cka(X_1, Y_1)
ax = sns.heatmap(sim)
plt.title("heatmap title")
ax.invert_yaxis()
#%%
print("Just a test")