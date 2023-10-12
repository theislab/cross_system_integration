# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.0
#   kernelspec:
#     display_name: cs_integration
#     language: python
#     name: cs_integration
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import os
import sys
import argparse

import scanpy as sc
import pickle as pkl
import pandas as pd
from scipy.sparse import csr_matrix, find
import numpy as np
from sklearn.preprocessing import minmax_scale
from scipy.stats import entropy
from sklearn.cluster import MiniBatchKMeans

import gc

from matplotlib.pyplot import rcParams
import matplotlib.pyplot as plt
import seaborn as sb

# %%
import torch
from cross_system_integration.model._xxjointmodel import XXJointModel
from pytorch_lightning.callbacks.base import Callback

# %% [markdown]
# ## Config

# %%
SYSTEM_KEY = 'system'
BATCH_KEYS = ['batch']
CT_KEY = 'cell_type_eval'

# %%
path_data = os.path.expanduser("~/data/cs_integration/combined_orthologuesHVG.h5ad")
path_fig = os.path.expanduser(f"~/io/cs_integration/figures/")

RUNS_TO_LOAD = {
    '2_prior_system_0': os.path.expanduser('~/io/cs_integration/vamp_testing_pancreas_combined_orthologuesHVG_n_prior_2_trainable_prior_True_init_system_0/'),
    '2_prior_balanced': os.path.expanduser('~/io/cs_integration/vamp_testing_pancreas_combined_orthologuesHVG_n_prior_2_trainable_prior_True_init_most_balanced/'),
    '2_prior_system_1': os.path.expanduser('~/io/cs_integration/vamp_testing_pancreas_combined_orthologuesHVG_n_prior_2_trainable_prior_True_init_system_1/'),
    '4_prior_balanced': os.path.expanduser('~/io/cs_integration/vamp_testing_pancreas_combined_orthologuesHVG_n_prior_4_trainable_prior_True_init_most_balanced/'),
    '10_prior_balanced': os.path.expanduser('~/io/cs_integration/vamp_testing_pancreas_combined_orthologuesHVG_n_prior_10_trainable_prior_True_init_most_balanced/'),
}
sc.settings.figdir = os.path.expanduser(f"~/io/cs_integration/figures/")

# %%
SYSTEM_MAP = {'0': 'Mouse', '1': 'Human', '0.0': 'Mouse', '1.0': 'Human'}

# %% [markdown]
# ## Some Utils

# %% [markdown]
# ## Pancreas

# %%
adata=sc.read(path_data)
adata

# %%
with open(os.path.expanduser('~/io/cs_integration/colors/obs_col_cmap.pkl'), 'rb') as f:
    obs_cmap = pkl.load(f)

# %% [markdown]
# ### init population

# %%
adata_training = adata.copy()
# model = XXJointModel.load(output_filename, adata=adata_training)

# %%
size = 3
fig,axs=plt.subplots(2, 3,figsize=(3*size, 2*size),
                     sharey='row')
for i, (run, title) in enumerate(zip(
    ['2_prior_system_0', '2_prior_balanced', '2_prior_system_1'],
    ['2 priors from Mouse', 'Mixed (1 + 1)', '2 priors from Human'],
)):
    embed = sc.read(os.path.join(RUNS_TO_LOAD[run], 'embed.h5ad'))
    embed_all = sc.read(os.path.join(RUNS_TO_LOAD[run], 'embed_all.h5ad'))
    for j, col in enumerate(['system', CT_KEY]):
        ax = axs[j, i]
        cmap = obs_cmap['pancreas'][col]
        if col == 'system':
            embed.obs['system'] = embed.obs['system'].astype(str).map(SYSTEM_MAP).astype(str)
            cmap = {SYSTEM_MAP[k]: v for k, v in cmap.items()}
        sc.pl.umap(embed, color=col, ax=ax, show=False, palette=cmap, frameon=False, title='' if j != 0 else title, legend_loc='none' if i != 2 else 'right margin')

plt.subplots_adjust(left=0.05,
                    bottom=0.05,
                    right=0.95,
                    top=0.95,
                    wspace=0.01,
                    hspace=0.01)
plt.savefig(path_fig+f'init_system_choice.pdf', dpi=300, bbox_inches='tight')

# %%

# %%

# %% [markdown]
# ## prior update over time

# %%
run = '2_prior_balanced'
adata_training = adata.copy()
model = XXJointModel.load(RUNS_TO_LOAD[run], adata=adata_training)
embed_all = sc.read(os.path.join(RUNS_TO_LOAD[run], 'embed_all.h5ad'))
embed_all.obs['system'] = embed_all.obs['system'].astype(str).map(SYSTEM_MAP).astype(str).replace('nan', float('nan'))
sc.pp.subsample(embed_all, fraction=1)

# %%
data_dict = next(iter(model._make_data_loader(adata=adata_training, indices=np.arange(adata_training.n_obs), 
                                              batch_size=adata_training.n_obs, shuffle=False)))
original_cov = sc.AnnData(
    torch.cat([data_dict['covariates'], data_dict['system']], 1).numpy(),
    obs=adata_training.obs
)
del data_dict
original_cov

# %%
prior_history = model.train_prior_history_
n_steps = len(prior_history)
n_points = prior_history[0][0].shape[0]

prior_x = sc.AnnData(np.concatenate([x[0] for x in prior_history]), var=adata_training.var)
prior_x.obs['pseudoinput_id'] = [i % n_points for i in range(n_steps * n_points)]
prior_x.obs['pseudoinput_id'] = [i % n_points for i in range(n_steps * n_points)]
prior_x.obs['pseudoinput_id'] = prior_x.obs['pseudoinput_id'].astype('category')
prior_x.obs['pseudoinput_time'] = [i // n_points for i in range(n_steps * n_points)]

prior_cov = sc.AnnData(np.concatenate([x[1] for x in prior_history]), obs=prior_x.obs)

prior_x, prior_cov

# %%
origin_all = sc.concat([adata_training, prior_x], merge='unique', join='outer')
sc.tl.pca(origin_all)
origin_all

# %%
cov_all = sc.concat([original_cov, prior_cov], merge='unique', join='outer')
sc.tl.pca(cov_all)
cov_all

# %%

# %%
sc.pl.pca(origin_all, color=['system', CT_KEY, 'pseudoinput_time'], components=['1,2', '3,4'], ncols=4)

# %%
sc.pl.pca(cov_all, color=['system', CT_KEY, 'pseudoinput_time'], components=['1,2', '3,4'], ncols=4, size=50)

# %%

# %%
# sc.pp.subsample(embed_all, fraction=0.01)

# %%
size = 5
fig,axs=plt.subplots(3, 3,figsize=(3*size, 2*size),
                     sharey='row')
for i, (components, title) in enumerate(zip(
    ['1,2', '3,4', '5,6'],
    ['PC1 & PC2', 'PC3 & PC4', 'PC5 & P6'],
)):
    for j, col in enumerate(['pseudoinput_time', 'system', CT_KEY]):
        ax = axs[j, i]
        if col == 'pseudoinput_time':
            if components == '1,2':
                embed_all.obsm['X_pca_sub'] = embed_all.obsm['X_pca'][:, 0:2]
            if components == '3,4':
                embed_all.obsm['X_pca_sub'] = embed_all.obsm['X_pca'][:, 2:4]
            if components == '5,6':
                embed_all.obsm['X_pca_sub'] = embed_all.obsm['X_pca'][:, 4:6]
            df = pd.DataFrame(embed_all.obsm['X_pca_sub'], columns=['x', 'y'])
            embed_all.obs['dummy'] = float('nan')
            sc.pl.pca(embed_all[embed_all.obs['pseudoinput_time'].isna()], alpha=0.01,
                      size=10, components=components, ax=ax, show=False, frameon=False, title='', legend_loc='none', zorder=0)
            plot = sb.kdeplot(data=df, x="x", y="y", ax=ax, zorder=5)
        if col == 'pseudoinput_time':
            cmap = None
        else:
            cmap = obs_cmap['pancreas'][col]
            if col == 'system':
                cmap = {SYSTEM_MAP[k]: v for k, v in cmap.items()}
        sc.pl.pca(embed_all if col != 'pseudoinput_time' else embed_all[~embed_all.obs['pseudoinput_time'].isna()], 
                  color=col, size=50 if j == 0 else None, components=components, ax=ax, show=False, palette=cmap, frameon=False, title='' if j != 0 else title, legend_loc='none' if i != 2 else 'right margin', zorder=10)

plt.subplots_adjust(left=0.05,
                    bottom=0.05,
                    right=0.95,
                    top=0.95,
                    wspace=0.01,
                    hspace=0.01)
plt.savefig(path_fig+f'pi_movement.pdf', dpi=300, bbox_inches='tight')

# %%

# %%

# %%
