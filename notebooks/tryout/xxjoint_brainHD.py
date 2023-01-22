# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: csp
#     language: python
#     name: csp
# ---

# %%
# Data pp imports
import scanpy as sc
import pickle as pkl
import pandas as pd
from scipy.sparse import csr_matrix, find
import numpy as np

import gc

from matplotlib.pyplot import rcParams
import matplotlib.pyplot as plt
import seaborn as sb

# %%
# Modelling imports
import torch

import sys
sys.path.insert(0, '/lustre/groups/ml01/code/karin.hrovatin/cross_species_prediction/')
import constraint_pancreas_example.model._xxjointmodel as xxjm
import importlib
importlib.reload(xxjm)
from constraint_pancreas_example.model._xxjointmodel import XXJointModel

# %%
path_data='/lustre/groups/ml01/workspace/karin.hrovatin/data/cross_species_prediction/brainHD_example/'

# %%
adata=sc.read(path_data+'combined_orthologues.h5ad')

# %%
adata_training = XXJointModel.setup_anndata(
    adata=adata,
    system_key='system',
    group_key='CellType',
    categorical_covariate_keys=['Batch'],
)

# %% [markdown]
# ## cVAE + z_distance_cycle

# %%
model = XXJointModel(adata=adata_training)
model.train(max_epochs=40,
           plan_kwargs={'loss_weights':dict(
            kl_weight= 1.0,
            kl_cycle_weight = 0,
            reconstruction_weight= 1.0,
            reconstruction_mixup_weight = 0,
            reconstruction_cycle_weight= 0,
            z_distance_cycle_weight = 2,
            translation_corr_weight = 0,
            z_contrastive_weight = 0,
           )})

# %%
# Plot all loses
losses=[k for k in model.trainer.logger.history.keys() 
        if '_step' not in k and '_epoch' not in k]
fig,axs=plt.subplots(2,len(losses),figsize=(len(losses)*3,4))
for ax_i,l in enumerate(losses):
    axs[0,ax_i].plot(
        model.trainer.logger.history[l].index,
        model.trainer.logger.history[l][l])
    axs[0,ax_i].set_title(l)
    axs[1,ax_i].plot(
        model.trainer.logger.history[l].index[20:],
        model.trainer.logger.history[l][l][20:])
fig.tight_layout()

# %%
embed = model.embed(
        adata=adata_training,
        indices=None,
        batch_size=None,
        as_numpy=True)

# %%
embed=sc.AnnData(embed,obs=adata_training.obs)
embed.obs['species']=embed.obs.system.map({0:'mm',1:'hs'})

# %%
sc.pp.neighbors(embed, use_rep='X')
sc.tl.umap(embed)

# %%
rcParams['figure.figsize']=(8,8)
sc.pl.umap(embed,color=['species','CellType', 'Condition','Grade','dataset','Batch'],s=10,wspace=0.5)

# %%
for ds in embed.obs['dataset'].unique():
    sc.pl.umap(embed[embed.obs.dataset==ds,:], color='Condition',s=10,wspace=0.5,title=ds)

# %% [markdown]
# C: Effect of disease after integration is not seen or not accordingly in mouse and human. Even for oligodendrocytes that show strongest effect (although they differ more between species), also not for the SPN which should be more similar between the two species.

# %%
