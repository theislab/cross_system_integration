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

import gc

from matplotlib.pyplot import rcParams
import matplotlib.pyplot as plt
import seaborn as sb

# %%
# Modelling imports
import torch

from constraint_pancreas_example.model._xxjointmodel import XXJointModel

# %%
path_data='/lustre/groups/ml01/workspace/karin.hrovatin/data/cross_species_prediction/pancreas_example/'
path_genes='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/gene_lists/'

# %%
adata_hs=sc.read(path_data+'human_orthologues.h5ad')
adata_mm=sc.read(path_data+'mouse_orthologues.h5ad')

# %% [markdown]
# ## PP

# %%
# Prepare adatas for concat and concat
# Human
adata_hs_sub=adata_hs.copy()
adata_hs_sub.obs=adata_hs_sub.obs[['cell_type_final','study_sample']]
adata_hs_sub.obs['system']=1
adata_hs_sub.var['EID']=adata_hs_sub.var_names
adata_hs_sub.var_names=adata_mm.var_names
del adata_hs_sub.obsm
# Mouse
adata_mm_sub=adata_mm.copy()
adata_mm_sub.obs=adata_mm_sub.obs[['cell_type_final','study_sample']]
adata_mm_sub.obs['system']=0
del adata_mm_sub.obsm
# Concat
adata=sc.concat([adata_mm_sub,adata_hs_sub])

del adata_mm_sub
del adata_hs_sub
gc.collect()

display(adata)

# %% [markdown]
# ## Train with different z cycle consitency weights

# %%
adata_training = XXJointModel.setup_anndata(
    adata=adata,
    system_key='system',
    group_key='cell_type_final',
    categorical_covariate_keys=['study_sample'],
)

# %% [markdown]
# ### MSE loss

# %%
models={}

# %%
# Models with different cycle consistency weights
for weight in [0.0,1.0,5.0,100.0]:
    # Compute model only if not yet computed
    if weight not in models:
        model = XXJointModel(adata=adata_training)
        model.train(max_epochs=40,
                   plan_kwargs={'loss_weights':dict(
                    kl_weight= 1.0,
                    kl_cycle_weight = 0,
                    reconstruction_weight= 1.0,
                    reconstruction_mixup_weight = 0,
                    reconstruction_cycle_weight= 0,
                    z_distance_cycle_weight = weight,
                    translation_corr_weight = 0,
                    z_contrastive_weight = 0,
                   )})
        models[weight]=model
models={k:models[k] for k in sorted(models)}

# %%
# Losses of all models with different weight settings
losses=['reconstruction_loss_train','kl_local_train', 'z_distance_cycle_train']
epoch_detail=20
for weight,model in models.items():
    fig,axs=plt.subplots(2,len(losses),figsize=(len(losses)*3,4))
    fig.suptitle(str(weight))
    for ax_i,l in enumerate(losses):
        axs[0,ax_i].plot(
            model.trainer.logger.history[l].index,
            model.trainer.logger.history[l][l])
        axs[0,ax_i].set_title(l)
        axs[1,ax_i].plot(
            model.trainer.logger.history[l].index[epoch_detail:],
            model.trainer.logger.history[l][l][epoch_detail:])
    fig.tight_layout()

# %% [markdown]
# C: reconstruction loss would be improving further, although to a lesser extent

# %%
# Embeddings of different models
embeds={}
for weight,model in models.items():
    embeds[weight] = model.embed(
        adata=adata_training,
        indices=None,
        batch_size=None,
        as_numpy=True)

# %%
for weight,embed in embeds.items():
    fig,ax=plt.subplots()
    plt.violinplot(embeds[weight])
    ax.set_title(weight)

# %% [markdown]
# C: Cycle consistency loss as used currently (MSE) actually rduces the span of latent space and gives more importance to fewer latent components. Need different metric that is invariant to size of latent space. Also if we used cosine it could still set some components to 0 to achieve lower distance.
#
# C: TODO could check how much each z component explains cell type variation (e.g. per system).
#
# C: Need to check if variability is reduced during the training. Var would look in both batch and ct, thus use a metric that includes ct.

# %% [markdown]
# Since the span of features is not informative about their variability look at how they vary accross cts. To make this less afected by system do only for human data. Normalise features to [0,1] before averaging per ct to ensure that feature size does not affect the perceived variability. Then compute var of feature accross ct means.

# %%
ct_var={}
ct_entropy={}
for weight,embed in embeds.items():
    # Human cells
    cells= adata_training.obs.system==1
    embed_sub=embed[cells,:]
    # Minmax scale the embed comonents so that perceived var is not affected by range
    x=pd.DataFrame(minmax_scale(embed_sub),index=adata_training.obs_names[cells])
    #var_idx=x.var().sort_values(ascending=False).index
    x['cell_type']=adata_training[x.index,:].obs['cell_type_final']
    x=x.groupby('cell_type').mean()
    #x=x[var_idx]
    g=sb.clustermap(x,cmap='cividis',figsize=(7,5),#col_cluster=False
                   )
    g.fig.suptitle(weight)
    ct_var[weight]=x.var()
    ct_entropy[weight]=entropy(x)
ct_var=pd.DataFrame(ct_var)
ct_entropy=pd.DataFrame(ct_entropy)
fig,ax=plt.subplots(1,2,figsize=(8,3))
ct_var.boxplot(ax=ax[0])
ax[0].set_title('var over cts')
ct_entropy.boxplot(ax=ax[1])
ax[1].set_title('entropy over cts')

# %% [markdown]
# TOD: Use correlation for latent space- MSE not meaningful

# %%
for weight,embed in embeds.items():
    # Minmax scale the embed comonents so that perceived var is not affected by range
    x=pd.DataFrame(minmax_scale(embed),index=adata_training.obs_names)
    #var_idx=x.var().sort_values(ascending=False).index
    x['system']=adata_training[x.index,:].obs['system']
    x=x.groupby('system').mean()
    #x=x[var_idx]
    g=sb.clustermap(x,cmap='cividis',figsize=(7,1),#col_cluster=False
                   )
    g.fig.suptitle(weight)


# %% [markdown]
# ### MSE loss on standardised data

# %%
models={}

# %%
# Models with different cycle consistency weights
for weight in [0.0,1.0,5.0,100.0]:
    # Compute model only if not yet computed
    if weight not in models:
        model = XXJointModel(adata=adata_training, z_dist_metric = 'MSE_standard')
        model.train(max_epochs=40,
                   plan_kwargs={'loss_weights':dict(
                    kl_weight= 1.0,
                    kl_cycle_weight = 0,
                    reconstruction_weight= 1.0,
                    reconstruction_mixup_weight = 0,
                    reconstruction_cycle_weight= 0,
                    z_distance_cycle_weight = weight,
                    translation_corr_weight = 0,
                    z_contrastive_weight = 0,
                   )})
        models[weight]=model
models={k:models[k] for k in sorted(models)}

# %%
# Losses of all models with different weight settings
losses=['reconstruction_loss_train','kl_local_train', 'z_distance_cycle_train']
epoch_detail=20
for weight,model in models.items():
    fig,axs=plt.subplots(2,len(losses),figsize=(len(losses)*3,4))
    fig.suptitle(str(weight))
    for ax_i,l in enumerate(losses):
        axs[0,ax_i].plot(
            model.trainer.logger.history[l].index,
            model.trainer.logger.history[l][l])
        axs[0,ax_i].set_title(l)
        axs[1,ax_i].plot(
            model.trainer.logger.history[l].index[epoch_detail:],
            model.trainer.logger.history[l][l][epoch_detail:])
    fig.tight_layout()

# %%
# Embeddings of different models
embeds={}
for weight,model in models.items():
    embeds[weight] = model.embed(
        adata=adata_training,
        indices=None,
        batch_size=None,
        as_numpy=True)

# %%
for weight,embed in embeds.items():
    fig,ax=plt.subplots()
    plt.violinplot(embeds[weight])
    ax.set_title(weight)

# %% [markdown]
# Since the span of features is not informative about their variability look at how they vary accross cts. To make this less afected by system do only for human data. Normalise features to [0,1] before averaging per ct to ensure that feature size does not affect the perceived variability. Then compute var of feature accross ct means.

# %%
ct_var={}
ct_entropy={}
for weight,embed in embeds.items():
    # Human cells
    cells= adata_training.obs.system==1
    embed_sub=embed[cells,:]
    # Minmax scale the embed comonents so that perceived var is not affected by range
    x=pd.DataFrame(minmax_scale(embed_sub),index=adata_training.obs_names[cells])
    #var_idx=x.var().sort_values(ascending=False).index
    x['cell_type']=adata_training[x.index,:].obs['cell_type_final']
    x=x.groupby('cell_type').mean()
    #x=x[var_idx]
    g=sb.clustermap(x,cmap='cividis',figsize=(7,5),#col_cluster=False
                   )
    g.fig.suptitle(weight)
    ct_var[weight]=x.var()
    ct_entropy[weight]=entropy(x)
ct_var=pd.DataFrame(ct_var)
ct_entropy=pd.DataFrame(ct_entropy)
fig,ax=plt.subplots(1,2,figsize=(8,3))
ct_var.boxplot(ax=ax[0])
ax[0].set_title('var over cts')
ct_entropy.boxplot(ax=ax[1])
ax[1].set_title('entropy over cts')

# %%
for weight,embed in embeds.items():
    # Minmax scale the embed comonents so that perceived var is not affected by range
    x=pd.DataFrame(minmax_scale(embed),index=adata_training.obs_names)
    #var_idx=x.var().sort_values(ascending=False).index
    x['system']=adata_training[x.index,:].obs['system']
    x=x.groupby('system').mean()
    #x=x[var_idx]
    g=sb.clustermap(x,cmap='cividis',figsize=(7,1),#col_cluster=False
                   )
    g.fig.suptitle(weight)


# %% [markdown]
# ## Train with different z KL weights

# %%
adata_training = XXJointModel.setup_anndata(
    adata=adata,
    system_key='system',
    group_key='cell_type_final',
    categorical_covariate_keys=['study_sample'],
)

# %%
models={}

# %%
# Models with different cycle consistency weights
for weight in [1.0,2.0,5.0,20.0]:
    # Compute model only if not yet computed
    if weight not in models:
        model = XXJointModel(adata=adata_training)
        model.train(max_epochs=40,
                   plan_kwargs={'loss_weights':dict(
                    kl_weight= weight,
                    kl_cycle_weight = 0,
                    reconstruction_weight= 1.0,
                    reconstruction_mixup_weight = 0,
                    reconstruction_cycle_weight= 0,
                    z_distance_cycle_weight = 0,
                    translation_corr_weight = 0,
                    z_contrastive_weight = 0,
                   )})
        models[weight]=model
models={k:models[k] for k in sorted(models)}

# %%
# Losses of all models with different weight settings
losses=['reconstruction_loss_train','kl_local_train']
epoch_detail=20
for weight,model in models.items():
    fig,axs=plt.subplots(2,len(losses),figsize=(len(losses)*3,4))
    fig.suptitle(str(weight))
    for ax_i,l in enumerate(losses):
        axs[0,ax_i].plot(
            model.trainer.logger.history[l].index,
            model.trainer.logger.history[l][l])
        axs[0,ax_i].set_title(l)
        axs[1,ax_i].plot(
            model.trainer.logger.history[l].index[epoch_detail:],
            model.trainer.logger.history[l][l][epoch_detail:])
    fig.tight_layout()

# %%
# Embeddings of different models
embeds={}
for weight,model in models.items():
    embeds[weight] = model.embed(
        adata=adata_training,
        indices=None,
        batch_size=None,
        as_numpy=True)

# %%
for weight,embed in embeds.items():
    fig,ax=plt.subplots()
    plt.violinplot(embeds[weight])
    ax.set_title(weight)

# %% [markdown]
# C: Also the KL leads to less var in features.

# %% [markdown]
# Since the span of features is not informative about their variability look at how they vary accross cts. To make this less afected by system do only for human data. Normalise features to [0,1] before averaging per ct to ensure that feature size does not affect the perceived variability. Then compute var of feature accross ct means.

# %%
ct_var={}
for weight,embed in embeds.items():
    # Human cells
    cells= adata_training.obs.system==1
    embed_sub=embed[cells,:]
    # Minmax scale the embed comonents so that perceived var is not affected by range
    x=pd.DataFrame(minmax_scale(embed_sub),index=adata_training.obs_names[cells])
    #var_idx=x.var().sort_values(ascending=False).index
    x['cell_type']=adata_training[x.index,:].obs['cell_type_final']
    x=x.groupby('cell_type').mean()
    #x=x[var_idx]
    g=sb.clustermap(x,cmap='cividis',figsize=(7,5),#col_cluster=False
                   )
    g.fig.suptitle(weight)
    ct_var[weight]=x.var()
ct_var=pd.DataFrame(ct_var)
fig,ax=plt.subplots()
ct_var.boxplot()

# %% [markdown]
# C: interesting that while KL seems to decrease feature range (Gaussian) it also increases variability of features accross cts.

# %%
