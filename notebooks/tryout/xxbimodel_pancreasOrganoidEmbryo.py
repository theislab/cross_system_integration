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
import torch

import sys
sys.path.insert(0, '/lustre/groups/ml01/code/karin.hrovatin/cross_species_prediction/')
import constraint_pancreas_example.model._xxjointmodel as xxjm
import importlib
importlib.reload(xxjm)
from constraint_pancreas_example.model._xxjointmodel import XXJointModel

# %%
path_data='/lustre/groups/ml01/workspace/karin.hrovatin/data/cross_species_prediction/pancreas_organoid_embryo/'

# %%
adata=sc.read(path_data+'organoid_embryo.h5ad')

# %%
adata_training = XXJointModel.setup_anndata(
    adata=adata,
    system_key='system',
    #group_key='CellType',
    categorical_covariate_keys=['sample'],
)

# %% [markdown]
# ## cVAE

# %%
model = XXJointModel(adata=adata_training)
model.train(max_epochs=50,
            check_val_every_n_epoch=1,
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
        if '_step' not in k and '_epoch' not in k and 'validation' not in k]
fig,axs=plt.subplots(2,len(losses),figsize=(len(losses)*3,4))
for ax_i,l_train in enumerate(losses):
    l_val=l_train.replace('_train','_validation')
    l_name=l_train.replace('_train','')
    for l,c in [(l_train,'tab:blue'),(l_val,'tab:orange')]:
        axs[0,ax_i].plot(
            model.trainer.logger.history[l].index,
            model.trainer.logger.history[l][l],c=c)
        axs[0,ax_i].set_title(l_name)
        axs[1,ax_i].plot(
            model.trainer.logger.history[l].index[10:],
            model.trainer.logger.history[l][l][10:],c=c)
fig.tight_layout()

# %%
embed_full = model.embed(
        adata=adata_training,
        indices=None,
        batch_size=None,
        as_numpy=True)

# %%
random_indices=np.random.permutation(list(range(embed_full.shape[0])))[:10000]
embed=sc.AnnData(embed_full[random_indices,:],obs=adata_training[random_indices,:].obs)
embed.obs['system_name']=embed.obs.system.map({0:'organoid',1:'embryo'})

# %%
sc.pp.neighbors(embed, use_rep='X')
sc.tl.umap(embed)

# %%
# Make cell type column per dataset since annotationb is not unified accross datasets
# so it will be easier to look at
for ds in embed.obs.dataset.unique():
    embed.obs['cell_type_'+ds]=embed.obs.apply(
        lambda x: x['cell_type'] if x['dataset']==ds else np.nan, axis=1)

# %%
rcParams['figure.figsize']=(6,6)
sc.pl.umap(embed,color=['system_name', 'dataset', 'sample']+
           ['cell_type_'+ds for ds in embed.obs.dataset.unique()],s=30,wspace=0.8,ncols=3)

# %% [markdown]
# ## cVAE + z_distance_cycle

# %%
model = XXJointModel(adata=adata_training)
model.train(max_epochs=50,
            check_val_every_n_epoch=1,
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
        if '_step' not in k and '_epoch' not in k and 'validation' not in k]
fig,axs=plt.subplots(2,len(losses),figsize=(len(losses)*3,4))
for ax_i,l_train in enumerate(losses):
    l_val=l_train.replace('_train','_validation')
    l_name=l_train.replace('_train','')
    for l,c in [(l_train,'tab:blue'),(l_val,'tab:orange')]:
        axs[0,ax_i].plot(
            model.trainer.logger.history[l].index,
            model.trainer.logger.history[l][l],c=c)
        axs[0,ax_i].set_title(l_name)
        axs[1,ax_i].plot(
            model.trainer.logger.history[l].index[10:],
            model.trainer.logger.history[l][l][10:],c=c)
fig.tight_layout()

# %%
embed_full = model.embed(
        adata=adata_training,
        indices=None,
        batch_size=None,
        as_numpy=True)

# %%
random_indices=np.random.permutation(list(range(embed_full.shape[0])))[:10000]
embed=sc.AnnData(embed_full[random_indices,:],obs=adata_training[random_indices,:].obs)
embed.obs['system_name']=embed.obs.system.map({0:'organoid',1:'embryo'})

# %%
sc.pp.neighbors(embed, use_rep='X')
sc.tl.umap(embed)

# %%
# Make cell type column per dataset since annotationb is not unified accross datasets
# so it will be easier to look at
for ds in embed.obs.dataset.unique():
    embed.obs['cell_type_'+ds]=embed.obs.apply(
        lambda x: x['cell_type'] if x['dataset']==ds else np.nan, axis=1)

# %%
rcParams['figure.figsize']=(6,6)
sc.pl.umap(embed,color=['system_name', 'dataset', 'sample']+
           ['cell_type_'+ds for ds in embed.obs.dataset.unique()],s=30,wspace=0.8,ncols=3)

# %% [markdown]
# TODO: Change z distance, add to model init `z_dist_metric = 'MSE_standard'`

# %% [markdown]
# ## cVAE + VampPrior

# %%
model = XXJointModel(adata=adata_training, 
                     prior='vamp', 
                     n_prior_components=100,
                     pseudoinputs_data_init=True)
model.train(max_epochs=50,
            check_val_every_n_epoch=1,
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
        if '_step' not in k and '_epoch' not in k and 'validation' not in k]
fig,axs=plt.subplots(2,len(losses),figsize=(len(losses)*3,4))
for ax_i,l_train in enumerate(losses):
    l_val=l_train.replace('_train','_validation')
    l_name=l_train.replace('_train','')
    for l,c in [(l_train,'tab:blue'),(l_val,'tab:orange')]:
        axs[0,ax_i].plot(
            model.trainer.logger.history[l].index,
            model.trainer.logger.history[l][l],c=c)
        axs[0,ax_i].set_title(l_name)
        axs[1,ax_i].plot(
            model.trainer.logger.history[l].index[10:],
            model.trainer.logger.history[l][l][10:],c=c)
fig.tight_layout()

# %%
embed_full = model.embed(
        adata=adata_training,
        indices=None,
        batch_size=None,
        as_numpy=True)

# %%
random_indices=np.random.permutation(list(range(embed_full.shape[0])))[:10000]
embed=sc.AnnData(embed_full[random_indices,:],obs=adata_training[random_indices,:].obs)
embed.obs['system_name']=embed.obs.system.map({0:'organoid',1:'embryo'})

# %%
sc.pp.neighbors(embed, use_rep='X')
sc.tl.umap(embed)

# %%
# Make cell type column per dataset since annotationb is not unified accross datasets
# so it will be easier to look at
for ds in embed.obs.dataset.unique():
    embed.obs['cell_type_'+ds]=embed.obs.apply(
        lambda x: x['cell_type'] if x['dataset']==ds else np.nan, axis=1)

# %%
rcParams['figure.figsize']=(6,6)
sc.pl.umap(embed,color=['system_name', 'dataset', 'sample']+
           ['cell_type_'+ds for ds in embed.obs.dataset.unique()],s=30,wspace=0.8,ncols=3)

# %% [markdown]
# C: Here the VampPrior alone does not work that well - not integrating one organoid sample?! But it integrates the other sample and embryo well. Maybe would need also datast as batch besides sample?

# %% [markdown]
# ## cVAE + VampPrior + z_dist_cycle

# %% [markdown]
# TODO run with same N epochs as above - in previous run it seemed to still be improving, but now changed z cyc dist metric and seems 50 would be enough

# %%
model = XXJointModel(adata=adata_training, 
                     prior='vamp', 
                     n_prior_components=100,
                     pseudoinputs_data_init=True,
                     z_dist_metric = 'MSE_standard')
model.train(max_epochs=100,
            check_val_every_n_epoch=1,
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
        if '_step' not in k and '_epoch' not in k and 'validation' not in k]
fig,axs=plt.subplots(2,len(losses),figsize=(len(losses)*3,4))
for ax_i,l_train in enumerate(losses):
    l_val=l_train.replace('_train','_validation')
    l_name=l_train.replace('_train','')
    for l,c in [(l_train,'tab:blue'),(l_val,'tab:orange')]:
        axs[0,ax_i].plot(
            model.trainer.logger.history[l].index,
            model.trainer.logger.history[l][l],c=c)
        axs[0,ax_i].set_title(l_name)
        axs[1,ax_i].plot(
            model.trainer.logger.history[l].index[10:],
            model.trainer.logger.history[l][l][10:],c=c)
fig.tight_layout()

# %%
embed_full = model.embed(
        adata=adata_training,
        indices=None,
        batch_size=None,
        as_numpy=True)

# %%
random_indices=np.random.permutation(list(range(embed_full.shape[0])))[:10000]
embed=sc.AnnData(embed_full[random_indices,:],obs=adata_training[random_indices,:].obs)
embed.obs['system_name']=embed.obs.system.map({0:'organoid',1:'embryo'})

# %%
sc.pp.neighbors(embed, use_rep='X')
sc.tl.umap(embed)

# %%
# Make cell type column per dataset since annotationb is not unified accross datasets
# so it will be easier to look at
for ds in embed.obs.dataset.unique():
    embed.obs['cell_type_'+ds]=embed.obs.apply(
        lambda x: x['cell_type'] if x['dataset']==ds else np.nan, axis=1)

# %%
rcParams['figure.figsize']=(6,6)
sc.pl.umap(embed,color=['system_name', 'dataset', 'sample']+
           ['cell_type_'+ds for ds in embed.obs.dataset.unique()],s=30,wspace=0.8,ncols=3)

# %% [markdown]
# C: This setting of cycle z dist and vamp just does not work. Maybe need dataset info for vamp?

# %% [markdown]
# ## Dataset as additional batch
# Since VampPrior integrates well one organoid and embryo but not the other maybe it also requires dataset as batch besides organoid

# %%
adata_training = XXJointModel.setup_anndata(
    adata=adata,
    system_key='system',
    #group_key='CellType',
    categorical_covariate_keys=['sample','dataset'],
)

# %% [markdown]
# ## cVAE + VampPrior with dataset as additional batch

# %%
model = XXJointModel(adata=adata_training, 
                     prior='vamp', 
                     n_prior_components=100,
                     pseudoinputs_data_init=True)
model.train(max_epochs=50,
            check_val_every_n_epoch=1,
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
        if '_step' not in k and '_epoch' not in k and 'validation' not in k]
fig,axs=plt.subplots(2,len(losses),figsize=(len(losses)*3,4))
for ax_i,l_train in enumerate(losses):
    l_val=l_train.replace('_train','_validation')
    l_name=l_train.replace('_train','')
    for l,c in [(l_train,'tab:blue'),(l_val,'tab:orange')]:
        axs[0,ax_i].plot(
            model.trainer.logger.history[l].index,
            model.trainer.logger.history[l][l],c=c)
        axs[0,ax_i].set_title(l_name)
        axs[1,ax_i].plot(
            model.trainer.logger.history[l].index[10:],
            model.trainer.logger.history[l][l][10:],c=c)
fig.tight_layout()

# %%
embed_full = model.embed(
        adata=adata_training,
        indices=None,
        batch_size=None,
        as_numpy=True)

# %%
random_indices=np.random.permutation(list(range(embed_full.shape[0])))[:10000]
embed=sc.AnnData(embed_full[random_indices,:],obs=adata_training[random_indices,:].obs)
embed.obs['system_name']=embed.obs.system.map({0:'organoid',1:'embryo'})

# %%
sc.pp.neighbors(embed, use_rep='X')
sc.tl.umap(embed)

# %%
# Make cell type column per dataset since annotationb is not unified accross datasets
# so it will be easier to look at
for ds in embed.obs.dataset.unique():
    embed.obs['cell_type_'+ds]=embed.obs.apply(
        lambda x: x['cell_type'] if x['dataset']==ds else np.nan, axis=1)

# %%
rcParams['figure.figsize']=(6,6)
sc.pl.umap(embed,color=['system_name', 'dataset', 'sample']+
           ['cell_type_'+ds for ds in embed.obs.dataset.unique()],s=30,wspace=0.8,ncols=3)

# %% [markdown]
# C: E5 simply does not integrate well with VAMP prior

# %% [markdown]
# ## cVAE with dataset as additional batch

# %%
model = XXJointModel(adata=adata_training)
model.train(max_epochs=50,
            check_val_every_n_epoch=1,
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
        if '_step' not in k and '_epoch' not in k and 'validation' not in k]
fig,axs=plt.subplots(2,len(losses),figsize=(len(losses)*3,4))
for ax_i,l_train in enumerate(losses):
    l_val=l_train.replace('_train','_validation')
    l_name=l_train.replace('_train','')
    for l,c in [(l_train,'tab:blue'),(l_val,'tab:orange')]:
        axs[0,ax_i].plot(
            model.trainer.logger.history[l].index,
            model.trainer.logger.history[l][l],c=c)
        axs[0,ax_i].set_title(l_name)
        axs[1,ax_i].plot(
            model.trainer.logger.history[l].index[10:],
            model.trainer.logger.history[l][l][10:],c=c)
fig.tight_layout()

# %%
embed_full = model.embed(
        adata=adata_training,
        indices=None,
        batch_size=None,
        as_numpy=True)

# %%
random_indices=np.random.permutation(list(range(embed_full.shape[0])))[:10000]
embed=sc.AnnData(embed_full[random_indices,:],obs=adata_training[random_indices,:].obs)
embed.obs['system_name']=embed.obs.system.map({0:'organoid',1:'embryo'})

# %%
sc.pp.neighbors(embed, use_rep='X')
sc.tl.umap(embed)

# %%
# Make cell type column per dataset since annotationb is not unified accross datasets
# so it will be easier to look at
for ds in embed.obs.dataset.unique():
    embed.obs['cell_type_'+ds]=embed.obs.apply(
        lambda x: x['cell_type'] if x['dataset']==ds else np.nan, axis=1)

# %%
rcParams['figure.figsize']=(6,6)
sc.pl.umap(embed,color=['system_name', 'dataset', 'sample']+
           ['cell_type_'+ds for ds in embed.obs.dataset.unique()],s=30,wspace=0.8,ncols=3)

# %%
