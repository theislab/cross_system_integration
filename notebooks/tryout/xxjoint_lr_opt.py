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
import pytorch_lightning as pl

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

# %% [markdown]
# ## Pancreas

# %%
adata=sc.read(path_data+'combined_orthologues.h5ad')

# %%
adata_training = XXJointModel.setup_anndata(
    adata=adata,
    system_key='system',
    #class_key='cell_type_final',
    categorical_covariate_keys=['study_sample'],
)

# %%
model = XXJointModel(adata=adata_training)
model.train(max_epochs=200,
            check_val_every_n_epoch=1,
            plan_kwargs={
               'lr':0.01,
               'lr_scheduler_metric':'train_loss',
               'optimizer':'AdamW',
               'reduce_lr_on_plateau':True,
               'lr_patience':5,
               "lr_factor":0.1,
               'lr_min':1e-7,
               'lr_threshold':10,
               "lr_threshold_mode":'abs',
               'loss_weights':{
                   'reconstruction_mixup_weight':0,
                   'reconstruction_cycle_weight':0,
                   'kl_cycle_weight':0,
                   'z_distance_cycle_weight':0,
                   'translation_corr_weight':0,
                   'z_contrastive_weight':0,

               }},
          callbacks=[pl.callbacks.LearningRateMonitor(logging_interval='epoch')])

# %% [markdown]
# The threshold needs to be set to -x with rel mode as we have negative losses so if we want -880 not to be seen as improvement over -800 we would need to set threshold to -0.1 as margin= -800*(1-(-0.1))=-800\*1.1=-880

# %%
# Plot all loses
losses=[k for k in model.trainer.logger.history.keys() 
        if '_step' not in k and '_epoch' not in k and 'validation' not in k ]
fig,axs=plt.subplots(2,len(losses),figsize=(len(losses)*3,4))
for ax_i,l_train in enumerate(losses):
    l_val=l_train.replace('_train','_validation')
    l_name=l_train.replace('_train','')
    for l,c in [(l_train,'tab:blue'),(l_val,'tab:orange')]:
        axs[0,ax_i].plot(
            model.trainer.logger.history[l].index if 'lr' not in l 
                else list(range(model.trainer.logger.history[l].index.shape[0])),
            model.trainer.logger.history[l][l],c=c)
        axs[0,ax_i].set_title(l_name)
        epochs_sub=20
        axs[1,ax_i].plot(
            model.trainer.logger.history[l].index[epochs_sub:] if 'lr' not in l 
                else list(range(epochs_sub,
                                model.trainer.logger.history[l].index.shape[0])),
            model.trainer.logger.history[l][l][epochs_sub:],c=c)
fig.tight_layout()

# %% [markdown]
# C: Based on trial and error determined that training often fails if starting with lr=0.1 - thus start with 0.01.
#
# C: Observed that need quite a large threshold as the loss variaes a lot.
#
# C: Need to go to low lr to stabilise training at the end - otherwise jumps arround too much.
#
# C: The patience can be low it seems.
#
# C: lr factor needs to be strong as need to get to low training losses.

# %% [markdown]
# Relative thresholding mode

# %% [markdown]
# ## Pancreas + tabula

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
model = XXJointModel(adata=adata_training)
model.train(max_epochs=200,
            check_val_every_n_epoch=1,
            plan_kwargs={
               'lr':0.01,
               'lr_scheduler_metric':'train_loss',
               'optimizer':'AdamW',
               'reduce_lr_on_plateau':True,
               'lr_patience':5,
               "lr_factor":0.1,
               'lr_min':1e-7,
               'lr_threshold':10,
               "lr_threshold_mode":'abs',
               'loss_weights':{
                   'reconstruction_mixup_weight':0,
                   'reconstruction_cycle_weight':0,
                   'kl_cycle_weight':0,
                   'z_distance_cycle_weight':0,
                   'translation_corr_weight':0,
                   'z_contrastive_weight':0,

               }},
          callbacks=[pl.callbacks.LearningRateMonitor(logging_interval='epoch')])

# %%
sorted(model.trainer.logger.history.keys() )

# %%
# Plot all loses
losses=[k for k in model.trainer.logger.history.keys() 
        if '_step' not in k and '_epoch' not in k and 'validation' not in k ]
fig,axs=plt.subplots(2,len(losses),figsize=(len(losses)*3,4))
for ax_i,l_train in enumerate(losses):
    l_val=l_train.replace('_train','_validation')
    l_name=l_train.replace('_train','')
    for l,c in [(l_train,'tab:blue'),(l_val,'tab:orange')]:
        axs[0,ax_i].plot(
            model.trainer.logger.history[l].index if 'lr' not in l 
                else list(range(model.trainer.logger.history[l].index.shape[0])),
            model.trainer.logger.history[l][l],c=c)
        axs[0,ax_i].set_title(l_name)
        epochs_sub=20
        axs[1,ax_i].plot(
            model.trainer.logger.history[l].index[epochs_sub:] if 'lr' not in l 
                else list(range(epochs_sub,
                                model.trainer.logger.history[l].index.shape[0])),
            model.trainer.logger.history[l][l][epochs_sub:],c=c)
fig.tight_layout()

# %%
