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
from sklearn.preprocessing import minmax_scale

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
path_data='/lustre/groups/ml01/workspace/karin.hrovatin/data/cross_species_prediction/pancreas_example/'
path_genes='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/gene_lists/'

# %%
adata=sc.read(path_data+'combined_orthologues.h5ad')

# %%
adata_training = XXJointModel.setup_anndata(
    adata=adata,
    system_key='system',
    #class_key='cell_type_final',
    categorical_covariate_keys=['study_sample'],
)

# %% [markdown]
# ## cVAE

# %%
model = XXJointModel(adata=adata_training)
model.train(max_epochs=20,
           plan_kwargs={'loss_weights':{
               'reconstruction_mixup_weight':0,
               'reconstruction_cycle_weight':0,
               'kl_cycle_weight':0,
               'z_distance_cycle_weight':0,
               'translation_corr_weight':0,
               'z_contrastive_weight':0,
               
           }})

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
embed_first = model.embed(
        adata=adata_training,
        indices=None,
        batch_size=None,
        as_numpy=True)
embed_cyc = model.embed(
        adata=adata_training,
        cycle=True,
        indices=None,
        batch_size=None,
        as_numpy=True)

# %%
embed=sc.AnnData(np.concatenate([embed_first,embed_cyc]),
                 obs=pd.concat([adata_training.obs,adata_training.obs]))
embed.obs['species']=embed.obs.system.map({0:'mm',1:'hs'})
embed.obs['z_type']=['first']*adata_training.shape[0]+['cycle']*adata_training.shape[0]
# Remove duplicated obs names
embed.obs['cell']=embed.obs_names
embed.obs_names=[str(i) for i in range(embed.shape[0])]

# %%
sc.pp.neighbors(embed, use_rep='X')
sc.tl.umap(embed)

# %%
rcParams['figure.figsize']=(8,8)
np.random.seed(0)
random_indices=np.random.permutation(list(range(embed.shape[0])))
sc.pl.umap(embed[random_indices,:],
           color=['cell_type_final', 'study_sample',  'species', 'z_type'],s=10, wspace = 0.5)

# %%
for group,cells in embed.obs.groupby(['species','z_type']):
    fig,ax=plt.subplots(figsize=(5,5))
    sc.pl.umap(embed,ax=ax,show=False)
    sc.pl.umap(embed[cells.index,:],color=['cell_type_final'],s=10, wspace = 0.5, ax=ax,
              title='_'.join(group))

# %%
fig,ax=plt.subplots(1,2,figsize=(8,3))
for idx,z_type in enumerate(embed.obs.z_type.unique()):
    ax[idx].violinplot(embed[embed.obs.z_type==z_type,:].X)
    ax[idx].set_title(z_type)

# %%
for group,cells in embed.obs.groupby(['species','z_type']):
    embed_sub=embed[cells.index,:]
    # Minmax scale the embed comonents so that perceived var is not affected by range
    x=pd.DataFrame(minmax_scale(embed_sub.X),index=embed_sub.obs_names)
    #var_idx=x.var().sort_values(ascending=False).index
    x['cell_type']=embed_sub.obs['cell_type_final']
    x=x.groupby('cell_type').mean()
    #x=x[var_idx]
    g=sb.clustermap(x,cmap='cividis',figsize=(7,5),
                    col_cluster=False,row_cluster=False
                   )
    g.fig.suptitle('_'.join(group))

# %% [markdown]
# C: The latent and cycle space seem similar, with some minor shifts.

# %%
