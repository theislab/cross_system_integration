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
path_data='/lustre/groups/ml01/workspace/karin.hrovatin/data/cross_species_prediction/pancreas_example/'
path_genes='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/gene_lists/'
path_tabula='/lustre/groups/ml01/workspace/karin.hrovatin/data/cross_species_prediction/tabula/'

# %%
adata=sc.read(path_data+'combined_tabula_orthologues.h5ad')

# %%
adata_training = XXJointModel.setup_anndata(
    adata=adata,
    system_key='system',
    group_key='cell_type',
    categorical_covariate_keys=['batch'],
)

# %% [markdown]
# ## Tabula + pancreas integration (cVAE)

# %%
model = XXJointModel(adata=adata_training)
model.train(max_epochs=40,
           plan_kwargs={'loss_weights':dict(
            kl_weight= 1.0,
            kl_cycle_weight = 0,
            reconstruction_weight= 1.0,
            reconstruction_mixup_weight = 0,
            reconstruction_cycle_weight= 0,
            z_distance_cycle_weight = 0,
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
sc.pl.umap(embed,color=['species','cell_type','dataset','batch'],s=10,wspace=0.5)

# %%
embed_sub=embed[embed.obs.dataset=='pancreas',:].copy()
embed_sub.obs['cell_type']=embed_sub.obs['cell_type'].astype(str)
if 'cell_type_colors' in embed_sub.uns:
    del embed_sub.uns['cell_type_colors']
sc.pl.umap(embed_sub,
           color=['species','cell_type','batch'],s=10,wspace=0.5)
del embed_sub

# %% [markdown]
# C: The integration between the two species is worse when adding tabula datasets.

# %%
ct_tabula_matched=pd.read_table(path_tabula+'cell_type_mapping.tsv',index_col=0)

# %%
embed_sub=embed[embed.obs.dataset=='tabula',:].copy()
sc.pl.umap(embed_sub,
           color=['system'],s=10,wspace=0.5)
del embed_sub

embed_sub=embed[(embed.obs.dataset=='tabula').values & 
                (embed.obs.system==1).values,:].copy()
embed_sub.obs['cell_type_matched']=ct_tabula_matched.loc[embed_sub.obs_names,'cell_type']
fig,ax=plt.subplots(figsize=(8,8))
sc.pl.umap(embed,ax=ax,show=False)
sc.pl.umap(embed_sub,
           color=['cell_type_matched'],s=10,wspace=0.5,ax=ax)

del embed_sub
embed_sub=embed[(embed.obs.dataset=='tabula').values & 
                (embed.obs.system==0).values,:].copy()
embed_sub.obs['cell_type_matched']=ct_tabula_matched.loc[embed_sub.obs_names,'cell_type']
fig,ax=plt.subplots(figsize=(8,8))
sc.pl.umap(embed,ax=ax,show=False)
sc.pl.umap(embed_sub,
           color=['cell_type_matched'],s=10,wspace=0.5,ax=ax)
del embed_sub

# %% [markdown]
# ## Add z cycle distance

# %%
model = XXJointModel(adata=adata_training)
model.train(max_epochs=40,
           plan_kwargs={'loss_weights':dict(
            kl_weight= 1.0,
            kl_cycle_weight = 0,
            reconstruction_weight= 1.0,
            reconstruction_mixup_weight = 0,
            reconstruction_cycle_weight= 0,
            z_distance_cycle_weight = 5,
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
sc.pl.umap(embed,color=['species','cell_type','dataset','batch'],s=10,wspace=0.5)

# %%
embed_sub=embed[embed.obs.dataset=='pancreas',:].copy()
embed_sub.obs['cell_type']=embed_sub.obs['cell_type'].astype(str)
if 'cell_type_colors' in embed_sub.uns:
    del embed_sub.uns['cell_type_colors']
sc.pl.umap(embed_sub,
           color=['species','cell_type','batch'],s=10,wspace=0.5)
del embed_sub

# %% [markdown]
# C: The integration between the two species is worse when adding tabula datasets.

# %%
ct_tabula_matched=pd.read_table(path_tabula+'cell_type_mapping.tsv',index_col=0)

# %%
embed_sub=embed[embed.obs.dataset=='tabula',:].copy()
sc.pl.umap(embed_sub,
           color=['system'],s=10,wspace=0.5)
del embed_sub

embed_sub=embed[(embed.obs.dataset=='tabula').values & 
                (embed.obs.system==1).values,:].copy()
embed_sub.obs['cell_type_matched']=ct_tabula_matched.loc[embed_sub.obs_names,'cell_type']
fig,ax=plt.subplots(figsize=(8,8))
sc.pl.umap(embed,ax=ax,show=False)
sc.pl.umap(embed_sub,
           color=['cell_type_matched'],s=10,wspace=0.5,ax=ax)

del embed_sub
embed_sub=embed[(embed.obs.dataset=='tabula').values & 
                (embed.obs.system==0).values,:].copy()
embed_sub.obs['cell_type_matched']=ct_tabula_matched.loc[embed_sub.obs_names,'cell_type']
fig,ax=plt.subplots(figsize=(8,8))
sc.pl.umap(embed,ax=ax,show=False)
sc.pl.umap(embed_sub,
           color=['cell_type_matched'],s=10,wspace=0.5,ax=ax)
del embed_sub

# %%
