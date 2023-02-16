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

# %% [markdown]
# Try different var modes

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

from scipy.stats import norm

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

# %% [markdown]
# ## Prepare data

# %%
adata=sc.read(path_data+'combined_orthologues.h5ad')

# %%
train_filter=(adata.obs['cell_type_final']!='beta').values |\
             (adata.obs['system']!=1).values
adata_training = XXJointModel.setup_anndata(
    adata=adata[train_filter,:],
    system_key='system',
    #class_key='cell_type_final',
    categorical_covariate_keys=['study_sample'],
)

# %%
# Cell used for translation in ref system and ground truth in query system
eval_cells_query=''
cells_ref=list(adata.obs.query(
    f'cell_type_final=="beta" &'+ 
    f'system!=1'+eval_cells_query).index)
cells_query=list(adata.obs.query(
    f'cell_type_final=="beta" &'+ 
    f'system==1'+eval_cells_query).index)
print(f'N cells ref: {len(cells_ref)} and query: {len(cells_query)}')

# %%
# True expression summary
mean_ref=adata[cells_ref,:].to_df().mean()
mean_ref.name='ref'
var_ref=adata[cells_ref,:].to_df().std()**2
var_ref.name='ref'
mean_query=adata[cells_query,:].to_df().mean()
mean_query.name='query'
var_query=adata[cells_query,:].to_df().std()**2
var_query.name='query'

# %% [markdown]
# ## out var mode = feature

# %%
model = XXJointModel(adata=adata_training, out_var_mode='feature',)
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
# Predict mean and var
mean_ref_pred_cells,var_ref_pred_cells=model.translate(
            adata= adata_training[cells_ref,:],
            switch_system = False,
            covariates = adata[cells_ref,:].obs,
            give_mean=True,
            give_var=True,
        )
mean_query_pred_cells,var_query_pred_cells=model.translate(
            adata= adata_training[cells_ref,:],
            switch_system = True,
            covariates = adata[cells_ref,:].obs,
            give_mean=True,
            give_var=True,
        )

# %% [markdown]
# Comparison of real and predicted means and vars in ref and query

# %%
fig,axs=plt.subplots(1,2,figsize=(6,3),sharey=True,sharex=True,
                     #subplot_kw={'xscale':'log','yscale':'log'}
                    )
axs[0].scatter(mean_ref,mean_ref_pred_cells.mean(axis=0),s=1)
axs[1].scatter(mean_query,mean_query_pred_cells.mean(axis=0),s=1)
axs[0].set_xlabel('mean true')
axs[0].set_ylabel('mean pred')
axs[0].set_title('ref')
axs[1].set_title('query')

# %% [markdown]
# C: mean is predicted much better for ref than query

# %%
fig,axs=plt.subplots(1,2,figsize=(6,3),sharey=True,sharex=True,
                     #subplot_kw={'xscale':'log','yscale':'log'}
                    )
axs[0].scatter(var_ref,var_ref_pred_cells.mean(axis=0),s=1)
axs[1].scatter(var_query,var_query_pred_cells.mean(axis=0),s=1)
axs[0].set_xlabel('var true')
axs[0].set_ylabel('var pred')
axs[0].set_title('ref')
axs[1].set_title('query')
axs[0].plot([0,2],[0,2],c='orange',lw=1)
axs[1].plot([0,2],[0,2],c='orange',lw=1)

# %%
fig,axs=plt.subplots(1,2,figsize=(6,3),sharey=True,sharex=True,
                     subplot_kw={'xscale':'log','yscale':'log'}
                    )
axs[0].scatter(var_ref,var_ref_pred_cells.mean(axis=0),s=1)
axs[1].scatter(var_query,var_query_pred_cells.mean(axis=0),s=1)
axs[0].set_xlabel('var true')
axs[0].set_ylabel('var pred')
axs[0].set_title('ref')
axs[1].set_title('query')
axs[0].plot([0,2],[0,2],c='orange',lw=1)
axs[1].plot([0,2],[0,2],c='orange',lw=1)

# %% [markdown]
# C: variance is not realistic if fitting one per feature, often too big (low expr genes) or too small

# %% [markdown]
# Comparison of mean vs var distn in real and pred query and ref cells

# %%
fig,axs=plt.subplots(2,2,figsize=(6,6),sharey=True,sharex=True,
                     #subplot_kw={'xscale':'log','yscale':'log'}
                    )
axs[0,0].scatter(mean_ref,var_ref,s=1)
axs[0,1].scatter(mean_ref_pred_cells.mean(axis=0),var_ref_pred_cells.mean(axis=0),s=1)
axs[1,0].scatter(mean_query,var_query,s=1)
axs[1,1].scatter(mean_query_pred_cells.mean(axis=0),var_query_pred_cells.mean(axis=0),s=1)
axs[1,0].set_xlabel('mean')
axs[0,0].set_ylabel('ref\nvar')
axs[1,0].set_ylabel('query\nvar')
axs[0,0].set_title('true')
_=axs[0,1].set_title('pred')

# %%
fig,axs=plt.subplots(2,2,figsize=(6,6),sharey=True,sharex=True,
                     subplot_kw={'xscale':'log','yscale':'log'}
                    )
axs[0,0].scatter(mean_ref,var_ref,s=1)
axs[0,1].scatter(mean_ref_pred_cells.mean(axis=0),var_ref_pred_cells.mean(axis=0),s=1)
axs[1,0].scatter(mean_query,var_query,s=1)
axs[1,1].scatter(mean_query_pred_cells.mean(axis=0),var_query_pred_cells.mean(axis=0),s=1)
axs[1,0].set_xlabel('mean')
axs[0,0].set_ylabel('ref\nvar')
axs[1,0].set_ylabel('query\nvar')
axs[0,0].set_title('true')
_=axs[0,1].set_title('pred')

# %% [markdown]
# C: in ref pred the mean-var relationship is not realistic

# %% [markdown]
# Relationship of means and vars in prediction per cell

# %%
fig,axs=plt.subplots(1,2,figsize=(6,3),sharey=True,sharex=True,
                     #subplot_kw={'xscale':'log','yscale':'log'}
                    )
axs[0].scatter(mean_ref_pred_cells.ravel(),var_ref_pred_cells.ravel(),s=0.1)
axs[1].scatter(mean_query_pred_cells.ravel(),var_query_pred_cells.ravel(),s=0.1)
axs[0].set_xlabel('mean')
axs[0].set_ylabel('var')
axs[0].set_title('ref')
_=axs[1].set_title('query')

# %% [markdown]
# C: var is same for all cells but not mean (per gene) thus there are vertical lines

# %% [markdown]
# Variance of vars and means accross cells

# %%
fig,axs=plt.subplots(1,2,figsize=(6,3),sharey=True,sharex=True,
                     #subplot_kw={'xscale':'log','yscale':'log'}
                    )
axs[0].scatter(mean_ref_pred_cells.mean(axis=0),mean_ref_pred_cells.var(axis=0),s=1)
axs[1].scatter(mean_query_pred_cells.mean(axis=0),mean_query_pred_cells.var(axis=0),s=1)
axs[0].set_xlabel('mean or mean')
axs[0].set_ylabel('var or mean')
axs[0].set_title('ref')
_=axs[1].set_title('query')

# %%
fig,axs=plt.subplots(1,2,figsize=(6,3),sharey=True,sharex=True,
                     #subplot_kw={'xscale':'log','yscale':'log'}
                    )
axs[0].scatter(var_ref_pred_cells.mean(axis=0),var_ref_pred_cells.var(axis=0),s=1)
axs[1].scatter(var_query_pred_cells.mean(axis=0),var_query_pred_cells.var(axis=0),s=1)
axs[0].set_xlabel('mean or var')
axs[0].set_ylabel('var or var')
axs[0].set_title('ref')
_=axs[1].set_title('query')

# %%
fig,axs=plt.subplots(1,2,figsize=(6,2),sharey=True,sharex=True,
                     #subplot_kw={'xscale':'log','yscale':'log'}
                    )
axs[0].hist(mean_ref_pred_cells.var(axis=0),bins=100)
axs[1].hist(mean_query_pred_cells.var(axis=0),bins=100)
axs[0].set_xlabel('var of mean')
axs[0].set_ylabel('N genes')
axs[0].set_title('ref')
_=axs[0].set_title('query')

# %%
fig,axs=plt.subplots(1,2,figsize=(6,2),sharey=True,sharex=True,
                     #subplot_kw={'xscale':'log','yscale':'log'}
                    )
axs[0].hist(var_ref_pred_cells.var(axis=0),bins=100)
axs[1].hist(var_query_pred_cells.var(axis=0),bins=100)
axs[0].set_xlabel('var of var')
axs[0].set_ylabel('N genes')
axs[0].set_title('ref')
_=axs[0].set_title('query')

# %% [markdown]
# C: Since mode is feture the var should be the same for a gene accross all cells. There are some minute differences - probably due to numerical reasons.

# %% [markdown]
# ## out var mode = sample_feature

# %%
model = XXJointModel(adata=adata_training, out_var_mode='sample_feature',)
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
# Predict mean and var
mean_ref_pred_cells,var_ref_pred_cells=model.translate(
            adata= adata_training[cells_ref,:],
            switch_system = False,
            covariates = adata[cells_ref,:].obs,
            give_mean=True,
            give_var=True,
        )
mean_query_pred_cells,var_query_pred_cells=model.translate(
            adata= adata_training[cells_ref,:],
            switch_system = True,
            covariates = adata[cells_ref,:].obs,
            give_mean=True,
            give_var=True,
        )

# %% [markdown]
# Comparison of real and predicted means and vars in ref and query

# %%
fig,axs=plt.subplots(1,2,figsize=(6,3),sharey=True,sharex=True,
                     #subplot_kw={'xscale':'log','yscale':'log'}
                    )
axs[0].scatter(mean_ref,mean_ref_pred_cells.mean(axis=0),s=1)
axs[1].scatter(mean_query,mean_query_pred_cells.mean(axis=0),s=1)
axs[0].set_xlabel('mean true')
axs[0].set_ylabel('mean pred')
axs[0].set_title('ref')
axs[1].set_title('query')

# %%
fig,axs=plt.subplots(1,2,figsize=(6,3),sharey=True,sharex=True,
                     #subplot_kw={'xscale':'log','yscale':'log'}
                    )
axs[0].scatter(var_ref,var_ref_pred_cells.mean(axis=0),s=1)
axs[1].scatter(var_query,var_query_pred_cells.mean(axis=0),s=1)
axs[0].set_xlabel('var true')
axs[0].set_ylabel('var pred')
axs[0].set_title('ref')
axs[1].set_title('query')
axs[0].plot([0,2],[0,2],c='orange',lw=1)
axs[1].plot([0,2],[0,2],c='orange',lw=1)

# %%
fig,axs=plt.subplots(1,2,figsize=(6,3),sharey=True,sharex=True,
                     subplot_kw={'xscale':'log','yscale':'log'}
                    )
axs[0].scatter(var_ref,var_ref_pred_cells.mean(axis=0),s=1)
axs[1].scatter(var_query,var_query_pred_cells.mean(axis=0),s=1)
axs[0].set_xlabel('var true')
axs[0].set_ylabel('var pred')
axs[0].set_title('ref')
axs[1].set_title('query')
axs[0].plot([0,2],[0,2],c='orange',lw=1)
axs[1].plot([0,2],[0,2],c='orange',lw=1)

# %% [markdown]
# C: var looks betetr than when var is predicted per feature, but still too low.

# %% [markdown]
# Comparison of mean vs var distn in real and pred query and ref cells

# %%
fig,axs=plt.subplots(2,2,figsize=(6,6),sharey=True,sharex=True,
                     #subplot_kw={'xscale':'log','yscale':'log'}
                    )
axs[0,0].scatter(mean_ref,var_ref,s=1)
axs[0,1].scatter(mean_ref_pred_cells.mean(axis=0),var_ref_pred_cells.mean(axis=0),s=1)
axs[1,0].scatter(mean_query,var_query,s=1)
axs[1,1].scatter(mean_query_pred_cells.mean(axis=0),var_query_pred_cells.mean(axis=0),s=1)
axs[1,0].set_xlabel('mean')
axs[0,0].set_ylabel('ref\nvar')
axs[1,0].set_ylabel('query\nvar')
axs[0,0].set_title('true')
_=axs[0,1].set_title('pred')

# %%
fig,axs=plt.subplots(2,2,figsize=(6,6),sharey=True,sharex=True,
                     subplot_kw={'xscale':'log','yscale':'log'}
                    )
axs[0,0].scatter(mean_ref,var_ref,s=1)
axs[0,1].scatter(mean_ref_pred_cells.mean(axis=0),var_ref_pred_cells.mean(axis=0),s=1)
axs[1,0].scatter(mean_query,var_query,s=1)
axs[1,1].scatter(mean_query_pred_cells.mean(axis=0),var_query_pred_cells.mean(axis=0),s=1)
axs[1,0].set_xlabel('mean')
axs[0,0].set_ylabel('ref\nvar')
axs[1,0].set_ylabel('query\nvar')
axs[0,0].set_title('true')
_=axs[0,1].set_title('pred')

# %% [markdown]
# C: mean-var relationships look more realistic than when var is predicted per fetaure

# %% [markdown]
# Relationship of means and vars in prediction per cell

# %%
fig,axs=plt.subplots(1,2,figsize=(6,3),sharey=True,sharex=True,
                     #subplot_kw={'xscale':'log','yscale':'log'}
                    )
axs[0].scatter(mean_ref_pred_cells.ravel(),var_ref_pred_cells.ravel(),s=0.1)
axs[1].scatter(mean_query_pred_cells.ravel(),var_query_pred_cells.ravel(),s=0.1)
axs[0].set_xlabel('mean')
axs[0].set_ylabel('var')
axs[0].set_title('ref')
_=axs[1].set_title('query')

# %% [markdown]
# Variance of vars and means accross cells

# %%
fig,axs=plt.subplots(1,2,figsize=(6,3),sharey=True,sharex=True,
                     #subplot_kw={'xscale':'log','yscale':'log'}
                    )
axs[0].scatter(mean_ref_pred_cells.mean(axis=0),mean_ref_pred_cells.var(axis=0),s=1)
axs[1].scatter(mean_query_pred_cells.mean(axis=0),mean_query_pred_cells.var(axis=0),s=1)
axs[0].set_xlabel('mean or mean')
axs[0].set_ylabel('var or mean')
axs[0].set_title('ref')
_=axs[1].set_title('query')

# %%
fig,axs=plt.subplots(1,2,figsize=(6,3),sharey=True,sharex=True,
                     #subplot_kw={'xscale':'log','yscale':'log'}
                    )
axs[0].scatter(var_ref_pred_cells.mean(axis=0),var_ref_pred_cells.var(axis=0),s=1)
axs[1].scatter(var_query_pred_cells.mean(axis=0),var_query_pred_cells.var(axis=0),s=1)
axs[0].set_xlabel('mean or var')
axs[0].set_ylabel('var or var')
axs[0].set_title('ref')
_=axs[1].set_title('query')

# %%
fig,axs=plt.subplots(1,2,figsize=(6,2),sharey=True,sharex=True,
                     #subplot_kw={'xscale':'log','yscale':'log'}
                    )
axs[0].hist(mean_ref_pred_cells.var(axis=0),bins=100)
axs[1].hist(mean_query_pred_cells.var(axis=0),bins=100)
axs[0].set_xlabel('var of mean')
axs[0].set_ylabel('N genes')
axs[0].set_title('ref')
_=axs[0].set_title('query')

# %%
fig,axs=plt.subplots(1,2,figsize=(6,2),sharey=True,sharex=True,
                     #subplot_kw={'xscale':'log','yscale':'log'}
                    )
axs[0].hist(var_ref_pred_cells.var(axis=0),bins=100)
axs[1].hist(var_query_pred_cells.var(axis=0),bins=100)
axs[0].set_xlabel('var of var')
axs[0].set_ylabel('N genes')
axs[0].set_title('ref')
_=axs[0].set_title('query')

# %% [markdown]
# C: fitting var based on embeddin actually works quite well.

# %% [markdown]
# ## out var mode = linear

# %%
model = XXJointModel(adata=adata_training, out_var_mode='linear',)
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
# Predict mean and var
mean_ref_pred_cells,var_ref_pred_cells=model.translate(
            adata= adata_training[cells_ref,:],
            switch_system = False,
            covariates = adata[cells_ref,:].obs,
            give_mean=True,
            give_var=True,
        )
mean_query_pred_cells,var_query_pred_cells=model.translate(
            adata= adata_training[cells_ref,:],
            switch_system = True,
            covariates = adata[cells_ref,:].obs,
            give_mean=True,
            give_var=True,
        )

# %% [markdown]
# Comparison of real and predicted means and vars in ref and query

# %%
fig,axs=plt.subplots(1,2,figsize=(6,3),sharey=True,sharex=True,
                     #subplot_kw={'xscale':'log','yscale':'log'}
                    )
axs[0].scatter(mean_ref,mean_ref_pred_cells.mean(axis=0),s=1)
axs[1].scatter(mean_query,mean_query_pred_cells.mean(axis=0),s=1)
axs[0].set_xlabel('mean true')
axs[0].set_ylabel('mean pred')
axs[0].set_title('ref')
axs[1].set_title('query')

# %% [markdown]
# C: mean is not predicted well in this mode - too low for highly expr genes, also less realistic. Probably needs to reduce mean to reduce var and thus increase LL.

# %%
fig,axs=plt.subplots(1,2,figsize=(6,3),sharey=True,sharex=True,
                     #subplot_kw={'xscale':'log','yscale':'log'}
                    )
axs[0].scatter(var_ref,var_ref_pred_cells.mean(axis=0),s=1)
axs[1].scatter(var_query,var_query_pred_cells.mean(axis=0),s=1)
axs[0].set_xlabel('var true')
axs[0].set_ylabel('var pred')
axs[0].set_title('ref')
axs[1].set_title('query')
axs[0].plot([0,2],[0,2],c='orange',lw=1)
axs[1].plot([0,2],[0,2],c='orange',lw=1)

# %%
fig,axs=plt.subplots(1,2,figsize=(6,3),sharey=True,sharex=True,
                     subplot_kw={'xscale':'log','yscale':'log'}
                    )
axs[0].scatter(var_ref,var_ref_pred_cells.mean(axis=0),s=1)
axs[1].scatter(var_query,var_query_pred_cells.mean(axis=0),s=1)
axs[0].set_xlabel('var true')
axs[0].set_ylabel('var pred')
axs[0].set_title('ref')
axs[1].set_title('query')
axs[0].plot([0,2],[0,2],c='orange',lw=1)
axs[1].plot([0,2],[0,2],c='orange',lw=1)

# %% [markdown]
# Comparison of mean vs var distn in real and pred query and ref cells

# %%
fig,axs=plt.subplots(2,2,figsize=(6,6),sharey=True,sharex=True,
                     #subplot_kw={'xscale':'log','yscale':'log'}
                    )
axs[0,0].scatter(mean_ref,var_ref,s=1)
axs[0,1].scatter(mean_ref_pred_cells.mean(axis=0),var_ref_pred_cells.mean(axis=0),s=1)
axs[1,0].scatter(mean_query,var_query,s=1)
axs[1,1].scatter(mean_query_pred_cells.mean(axis=0),var_query_pred_cells.mean(axis=0),s=1)
axs[1,0].set_xlabel('mean')
axs[0,0].set_ylabel('ref\nvar')
axs[1,0].set_ylabel('query\nvar')
axs[0,0].set_title('true')
_=axs[0,1].set_title('pred')

# %%
fig,axs=plt.subplots(2,2,figsize=(6,6),sharey=True,sharex=True,
                     subplot_kw={'xscale':'log','yscale':'log'}
                    )
axs[0,0].scatter(mean_ref,var_ref,s=1)
axs[0,1].scatter(mean_ref_pred_cells.mean(axis=0),var_ref_pred_cells.mean(axis=0),s=1)
axs[1,0].scatter(mean_query,var_query,s=1)
axs[1,1].scatter(mean_query_pred_cells.mean(axis=0),var_query_pred_cells.mean(axis=0),s=1)
axs[1,0].set_xlabel('mean')
axs[0,0].set_ylabel('ref\nvar')
axs[1,0].set_ylabel('query\nvar')
axs[0,0].set_title('true')
_=axs[0,1].set_title('pred')

# %% [markdown]
# Relationship of means and vars in prediction per cell

# %%
fig,axs=plt.subplots(1,2,figsize=(6,3),sharey=True,sharex=True,
                     #subplot_kw={'xscale':'log','yscale':'log'}
                    )
axs[0].scatter(mean_ref_pred_cells.ravel(),var_ref_pred_cells.ravel(),s=0.1)
axs[1].scatter(mean_query_pred_cells.ravel(),var_query_pred_cells.ravel(),s=0.1)
axs[0].set_xlabel('mean')
axs[0].set_ylabel('var')
axs[0].set_title('ref')
_=axs[1].set_title('query')

# %% [markdown]
# Variance of vars and means accross cells

# %%
fig,axs=plt.subplots(1,2,figsize=(6,3),sharey=True,sharex=True,
                     #subplot_kw={'xscale':'log','yscale':'log'}
                    )
axs[0].scatter(mean_ref_pred_cells.mean(axis=0),mean_ref_pred_cells.var(axis=0),s=1)
axs[1].scatter(mean_query_pred_cells.mean(axis=0),mean_query_pred_cells.var(axis=0),s=1)
axs[0].set_xlabel('mean or mean')
axs[0].set_ylabel('var or mean')
axs[0].set_title('ref')
_=axs[1].set_title('query')

# %%
fig,axs=plt.subplots(1,2,figsize=(6,3),sharey=True,sharex=True,
                     #subplot_kw={'xscale':'log','yscale':'log'}
                    )
axs[0].scatter(var_ref_pred_cells.mean(axis=0),var_ref_pred_cells.var(axis=0),s=1)
axs[1].scatter(var_query_pred_cells.mean(axis=0),var_query_pred_cells.var(axis=0),s=1)
axs[0].set_xlabel('mean or var')
axs[0].set_ylabel('var or var')
axs[0].set_title('ref')
_=axs[1].set_title('query')

# %%
fig,axs=plt.subplots(1,2,figsize=(6,2),sharey=True,sharex=True,
                     #subplot_kw={'xscale':'log','yscale':'log'}
                    )
axs[0].hist(mean_ref_pred_cells.var(axis=0),bins=100)
axs[1].hist(mean_query_pred_cells.var(axis=0),bins=100)
axs[0].set_xlabel('var of mean')
axs[0].set_ylabel('N genes')
axs[0].set_title('ref')
_=axs[0].set_title('query')

# %%
fig,axs=plt.subplots(1,2,figsize=(6,2),sharey=True,sharex=True,
                     #subplot_kw={'xscale':'log','yscale':'log'}
                    )
axs[0].hist(var_ref_pred_cells.var(axis=0),bins=100)
axs[1].hist(var_query_pred_cells.var(axis=0),bins=100)
axs[0].set_xlabel('var of var')
axs[0].set_ylabel('N genes')
axs[0].set_title('ref')
_=axs[0].set_title('query')

# %%
