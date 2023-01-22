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

from matplotlib.pyplot import rcParams
import matplotlib.pyplot as plt
import seaborn as sb

# %%
# Modelling imports
import torch

import sys
sys.path.insert(0, '/lustre/groups/ml01/code/karin.hrovatin/cross_species_prediction/')
import constraint_pancreas_example.model._xybimodel as xybm
import importlib
importlib.reload(xybm)
from constraint_pancreas_example.model._xybimodel import XYBiModel

# %%
path_data='/lustre/groups/ml01/workspace/karin.hrovatin/data/cross_species_prediction/samap_tabula/'

# %% [markdown]
# ## Data pp

# %%
adata=sc.read(path_data+'H_M_1101_new.h5ad')

# %% [markdown]
# TODO figure out batches within species

# %%
# Define which gene is from mouse/human
adata.var['species']=[v.split('_')[0] for v in adata.var_names]
adata.var['species']=adata.var['species'].map({'mo':'x','hu':'y'})
adata.var.species.value_counts()

# %%
# Many genes are unexpressed so don't use them
adata.var['n_cells']=np.array((adata.X>0).sum(axis=0)).ravel()

# %%
# Make combined adata with mouse and human cells in the same rows
obs_names_mm=adata.obs_names[adata.obs.species=='mo']
obs_names_hs=adata.obs_names[adata.obs.species=='hu']
mnn_mm,mnn_hs,_=find(adata.uns['mdata']['mnn']>0)
paired_mm=obs_names_mm[mnn_mm]
paired_hs=obs_names_hs[mnn_hs]
unpaired_mm=list(set(obs_names_mm)-set(paired_mm))
unpaired_hs=list(set(obs_names_hs)-set(paired_hs))
paired_obs_names=[m+'__'+h for m,h in zip(paired_mm,paired_hs)]
adata_in=sc.concat([
    sc.AnnData(adata[unpaired_hs+unpaired_mm,:].X,
        obs=pd.DataFrame(
            {'train_x':[0]*len(unpaired_hs)+[1]*len(unpaired_mm),
             'train_y':[1]*len(unpaired_hs)+[0]*len(unpaired_mm),
             'cell_x':[np.nan]*len(unpaired_hs)+list(adata[unpaired_mm,:].obs_names),
             'cell_y':list(adata[unpaired_hs,:].obs_names)+[np.nan]*len(unpaired_mm)},
            index=adata[unpaired_hs+unpaired_mm,:].obs_names),
        var=adata.var),
    sc.concat([
        sc.AnnData(adata[paired_hs,adata.var.species=='y'].X,
                   pd.DataFrame(
                       {'cell_y':adata[paired_hs,:].obs_names,
                       'train_x':[1]*len(paired_hs),
                       'train_y':[1]*len(paired_hs)},
                       index=paired_obs_names),
                   var=adata[:,adata.var.species=='y'].var), 
        sc.AnnData(adata[paired_mm,adata.var.species=='x'].X,
                   pd.DataFrame(
                       {'cell_x':adata[paired_mm,:].obs_names},
                       index=paired_obs_names),
                   var=adata[:,adata.var.species=='x'].var),  
    ], axis=1, 
        # To merge obs - columns unique to one adata
        merge='only')
],
    # To keep var info from the first object (same in both anyways)
    merge='first')

display(adata_in)

# %%
# Remove unexpressed genes
adata_in=adata_in[:,adata_in.var.n_cells>=10]
adata_in.shape

# %%
# Subset to HVGs, for now just use top 1000 genes for in/out per species
# Do this per species data subset as else may have problems with 0-s
hvg_mm=adata_in.var_names[adata_in.var.species=='x'][sc.pp.highly_variable_genes(
    adata_in[adata_in.obs.train_x.astype(bool),adata_in.var.species=='x'],
    flavor="cell_ranger",n_top_genes=1000, inplace=False)['highly_variable']]
hvg_hs=adata_in.var_names[adata_in.var.species=='y'][sc.pp.highly_variable_genes(
    adata_in[adata_in.obs.train_y.astype(bool),adata_in.var.species=='y'],
    flavor="cell_ranger",n_top_genes=1000, inplace=False)['highly_variable']]
adata_in=adata_in[:,list(hvg_hs)+list(hvg_mm)]
adata_in.shape

# %% [markdown]
# ## Model

# %% [markdown]
# ### Train

# %%
adata_training = XYBiModel.setup_anndata(
    adata=adata_in,
    xy_key='species',
    train_x_key='train_x',
    train_y_key='train_y',)
model = XYBiModel(adata=adata_training)

# %%
model.train(max_epochs=20)

# %%
losses=['reconstruction_loss_train', 'reconstruction_loss_cycle_train', 
        'kl_local_train', 'kl_local_cycle_train', 
        'z_distance_paired_train', 'z_distance_cycle_train']
fig,axs=plt.subplots(1,len(losses),figsize=(len(losses)*3,2))
for ax,l in zip(axs,losses):
    ax.plot(
        model.trainer.logger.history[l].index,
        model.trainer.logger.history[l][l])
    ax.set_title(l)

# %%
embed = model.embed(
        adata=adata_training,
        indices=None,
        batch_size=None,
        as_numpy=True)

# %%
# Make adata out of embedding
embed=sc.AnnData(
    # Sepearte embeding from x and y modalities
    np.concatenate([
        embed[0][adata_training.obs['train_x'].astype(bool),:],
        embed[1][adata_training.obs['train_y'].astype(bool),:],
    ]),
    # Add obs and add modality origin
    obs=pd.concat([
        pd.concat([
            adata_training.obs[adata_training.obs['train_x'].astype(bool)],
            pd.DataFrame(
                {'species':['x']*adata_training.obs['train_x'].sum()},
                 index=adata_training.obs_names[adata_training.obs['train_x'].astype(bool)])
        ], axis=1),
        pd.concat([
            adata_training.obs[adata_training.obs['train_y'].astype(bool)],
            pd.DataFrame(
                {'species':['y']*adata_training.obs['train_y'].sum()},
                 index=adata_training.obs_names[adata_training.obs['train_y'].astype(bool)])
        ], axis=1)
    ]))
# Some cells wont be unique so make them
embed.obs_names_make_unique()
# Info on paired cells
embed.obs['paired']=((embed.obs['train_x']==1).values & (embed.obs['train_y']==1).values
                    ).astype(str)

# %%
# randomly subset the embed to ensure that umap does not take forever
np.random.seed(0)
random_indices=np.random.permutation(list(range(embed.shape[0])))[:10000]
embed_sub=embed[random_indices,:].copy()
embed_sub.obs.groupby('paired')['species'].value_counts()

# %%
sc.pp.neighbors(embed_sub, use_rep='X')
sc.tl.umap(embed_sub)

# %%
sc.pl.umap(embed_sub,color=['species','paired'],s=5)

# %%
embed_sub.obs.loc[embed_sub.obs.query('species=="x"').index,'leiden_clusters']=\
    adata.obs.reindex(embed_sub.obs.query('species=="x"')['cell_x'])['leiden_clusters'].values
embed_sub.obs.loc[embed_sub.obs.query('species=="y"').index,'leiden_clusters']=\
    adata.obs.reindex(embed_sub.obs.query('species=="y"')['cell_y'])['leiden_clusters'].values
embed_sub.obs['leiden_clusters']=embed_sub.obs['leiden_clusters'].astype(str)

# %%
groups=['mo_176','mo_174','hu_181','hu_91']
with plt.rc_context({'figure.figsize':(4,4)}):
    colors=dict(zip(embed_sub.obs.leiden_clusters.unique(),
                    ['gray']*embed_sub.obs.leiden_clusters.nunique()))
    colors.update(dict(zip(groups,['r']*4)))
    for group in groups:
        if group in embed_sub.obs.leiden_clusters.unique():
            sc.pl.umap(embed_sub,
               color=['leiden_clusters'],groups=[group],s=40, palette=colors)

# %% [markdown]
# C: already on the SAMap integration these clusters were split up and only some parts overlapped (here top right). The same happens here so it seems that bio may be preserved and species integrated. Need dataset with better integration to confirm.

# %%
