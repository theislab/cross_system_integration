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
from scipy.stats import entropy
from sklearn.cluster import MiniBatchKMeans

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
path_tabula='/lustre/groups/ml01/workspace/karin.hrovatin/data/cross_species_prediction/tabula/'

# %%
adata=sc.read(path_data+'combined_tabula_orthologues.h5ad')

# %%
adata_training = XXJointModel.setup_anndata(
    adata=adata,
    system_key='system',
    #group_key='cell_type',
    categorical_covariate_keys=['batch'],
)

# %%
ct_tabula_matched=pd.read_table(path_tabula+'cell_type_mapping.tsv',index_col=0)

# %% [markdown]
# ## cVAE with pre-training

# %%
model = XXJointModel(adata=adata_training)
model.train(max_epochs=30,
            check_val_every_n_epoch=1,
            indices=np.argwhere(
                adata_training.obs['dataset'].values.ravel()=='tabula'
                ).ravel(),
            plan_kwargs={'loss_weights':{
               'reconstruction_mixup_weight':0,
               'reconstruction_cycle_weight':0,
               'kl_cycle_weight':0,
               'z_distance_cycle_weight':0,
               'translation_corr_weight':0,
               'z_contrastive_weight':0,
               
           }})
model.train(max_epochs=30,
            check_val_every_n_epoch=1,
            indices=np.argwhere(
                adata_training.obs['dataset'].values.ravel()!='tabula'
                ).ravel(),
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
losses=[k for k in model.history.keys() 
        if '_step' not in k and '_epoch' not in k and 'validation' not in k]
fig,axs=plt.subplots(2,len(losses),figsize=(len(losses)*3,4))
for ax_i,l_train in enumerate(losses):
    l_val=l_train.replace('_train','_validation')
    l_name=l_train.replace('_train','')
    for l,c in [(l_train,'tab:blue'),(l_val,'tab:orange')]:
        axs[0,ax_i].plot(
            model.history[l].index,
            model.history[l][l],c=c)
        axs[0,ax_i].set_title(l_name)
        axs[1,ax_i].plot(
            model.history[l].index[10:],
            model.history[l][l][10:],c=c)
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
rcParams['figure.figsize']=(5,3)
_=plt.violinplot(embed.to_df())

# %%
embed_sub=embed[np.random.RandomState(seed=0).permutation(embed.obs_names)[:10000],:].copy()

# %%
sc.pp.neighbors(embed_sub, use_rep='X')
sc.tl.umap(embed_sub)

# %%
rcParams['figure.figsize']=(8,8)
sc.pl.umap(embed_sub,color=['species','dataset','batch'],s=20,wspace=0.5)

# %%
embed_sub1=embed_sub[embed_sub.obs.dataset=='pancreas',:].copy()
embed_sub1.obs['cell_type']=embed_sub1.obs['cell_type'].astype(str)
if 'cell_type_colors' in embed_sub1.uns:
    del embed_sub1.uns['cell_type_colors']
sc.pl.umap(embed_sub1,
           color=['species','cell_type','batch'],s=20,wspace=0.5)
del embed_sub1

# %% [markdown]
# C: The post training dataset wasnt well integrated accross species.

# %%
embed_sub1=embed_sub[embed_sub.obs.dataset=='tabula',:].copy()
sc.pl.umap(embed_sub1,
           color=['system'],s=10,wspace=0.5)
del embed_sub1

embed_sub1=embed_sub[(embed_sub.obs.dataset=='tabula').values & 
                (embed_sub.obs.system==1).values,:].copy()
embed_sub1.obs['cell_type_matched']=ct_tabula_matched.loc[embed_sub1.obs_names,'cell_type']
fig,ax=plt.subplots(figsize=(8,8))
sc.pl.umap(embed_sub,ax=ax,show=False)
sc.pl.umap(embed_sub1,
           color=['cell_type_matched'],s=10,wspace=0.5,ax=ax)

del embed_sub1
embed_sub1=embed_sub[(embed_sub.obs.dataset=='tabula').values & 
                (embed_sub.obs.system==0).values,:].copy()
embed_sub1.obs['cell_type_matched']=ct_tabula_matched.loc[embed_sub1.obs_names,'cell_type']
fig,ax=plt.subplots(figsize=(8,8))
sc.pl.umap(embed_sub,ax=ax,show=False)
sc.pl.umap(embed_sub1,
           color=['cell_type_matched'],s=10,wspace=0.5,ax=ax)
del embed_sub1

# %% [markdown]
# C: It seems that cells are at least integrated between pre and post training datasets

# %% [markdown]
# ## cVAE + VampPrior with pre training

# %%
model = XXJointModel(adata=adata_training, 
                     prior='vamp', n_prior_components=500,
                     pseudoinputs_data_init=True)
model.train(max_epochs=30,
            check_val_every_n_epoch=1,
            indices=np.argwhere(
                adata_training.obs['dataset'].values.ravel()=='tabula'
                ).ravel(),
            plan_kwargs={'loss_weights':{
               'reconstruction_mixup_weight':0,
               'reconstruction_cycle_weight':0,
               'kl_cycle_weight':0,
               'z_distance_cycle_weight':0,
               'translation_corr_weight':0,
               'z_contrastive_weight':0,
               
           }})
model.train(max_epochs=30,
            check_val_every_n_epoch=1,
            indices=np.argwhere(
                adata_training.obs['dataset'].values.ravel()!='tabula'
                ).ravel(),
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
losses=[k for k in model.history.keys() 
        if '_step' not in k and '_epoch' not in k and 'validation' not in k]
fig,axs=plt.subplots(2,len(losses),figsize=(len(losses)*3,4))
for ax_i,l_train in enumerate(losses):
    l_val=l_train.replace('_train','_validation')
    l_name=l_train.replace('_train','')
    for l,c in [(l_train,'tab:blue'),(l_val,'tab:orange')]:
        axs[0,ax_i].plot(
            model.history[l].index,
            model.history[l][l],c=c)
        axs[0,ax_i].set_title(l_name)
        axs[1,ax_i].plot(
            model.history[l].index[10:],
            model.history[l][l][10:],c=c)
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
rcParams['figure.figsize']=(5,3)
_=plt.violinplot(embed.to_df())

# %%
embed_sub=embed[np.random.RandomState(seed=0).permutation(embed.obs_names)[:10000],:].copy()

# %%
sc.pp.neighbors(embed_sub, use_rep='X')
sc.tl.umap(embed_sub)

# %%
rcParams['figure.figsize']=(8,8)
sc.pl.umap(embed_sub,color=['species','dataset','batch'],s=20,wspace=0.5)

# %%
embed_sub1=embed_sub[embed_sub.obs.dataset=='pancreas',:].copy()
embed_sub1.obs['cell_type']=embed_sub1.obs['cell_type'].astype(str)
if 'cell_type_colors' in embed_sub1.uns:
    del embed_sub1.uns['cell_type_colors']
sc.pl.umap(embed_sub1,
           color=['species','cell_type','batch'],s=20,wspace=0.5)
del embed_sub1

# %% [markdown]
# C: It seems like integration between species completely fails even with vamp prior and even between the two datasets.

# %%
embed_sub1=embed_sub[embed_sub.obs.dataset=='tabula',:].copy()
sc.pl.umap(embed_sub1,
           color=['system'],s=10,wspace=0.5)
del embed_sub1

embed_sub1=embed_sub[(embed_sub.obs.dataset=='tabula').values & 
                (embed_sub.obs.system==1).values,:].copy()
embed_sub1.obs['cell_type_matched']=ct_tabula_matched.loc[embed_sub1.obs_names,'cell_type']
fig,ax=plt.subplots(figsize=(8,8))
sc.pl.umap(embed_sub,ax=ax,show=False)
sc.pl.umap(embed_sub1,
           color=['cell_type_matched'],s=10,wspace=0.5,ax=ax)

del embed_sub1
embed_sub1=embed_sub[(embed_sub.obs.dataset=='tabula').values & 
                (embed_sub.obs.system==0).values,:].copy()
embed_sub1.obs['cell_type_matched']=ct_tabula_matched.loc[embed_sub1.obs_names,'cell_type']
fig,ax=plt.subplots(figsize=(8,8))
sc.pl.umap(embed_sub,ax=ax,show=False)
sc.pl.umap(embed_sub1,
           color=['cell_type_matched'],s=10,wspace=0.5,ax=ax)
del embed_sub1

# %% [markdown]
# C: the species are not integrated well (some cts are but others arent), but cells are integrated accross datasets

# %%
