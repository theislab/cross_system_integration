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
import numpy as np
# Make random number for seed before scvi import sets seed to 0
seed=np.random.randint(0,1000000)
import argparse
import os
import string

from matplotlib.pyplot import rcParams
import matplotlib.pyplot as plt
import seaborn as sb

from constraint_pancreas_example.model._xxjointmodel import XXJointModel
import pytorch_lightning as pl


# %%
#import wandb
#wandb.init(project="vmp-debug", entity="k-hrovatin")

# %%
args=pkl.load(open('/lustre/groups/ml01/workspace/karin.hrovatin/data/cross_species_prediction/eval/presFeb23/translation/ST1GTtype B pancreatic cellMANoneSDFalseKLW1KLCW0RW1RMW0RCW0ZDCW0TCW0ZCW0PvNPC50NL2NH256_JtREolDe/args.pkl','rb'))

# %%
args

# %%
from scvi._settings import ScviConfig
config=ScviConfig()
config.seed=args.seed

# %%
adata=sc.read(args.path_adata)

# %%
# Cell used for translation in ref system and ground truth in query system
eval_cells_query='' if args.cells_eval is None else f'& {args.cells_eval}==True'
cells_ref=list(adata.obs.query(
    f'{args.group_key}=="{args.group_translate}" &'+ 
    f'{args.system_key}!={args.system_translate}'+eval_cells_query).index)
cells_query=list(adata.obs.query(
    f'{args.group_key}=="{args.group_translate}" &'+ 
    f'{args.system_key}=={args.system_translate}'+eval_cells_query).index)
print(f'N cells ref: {len(cells_ref)} and query: {len(cells_query)}')

# %%
adata.uns['eval_info']={
    'ref-pred_ref':{
        'cells_in':cells_ref, 
        'switch_system':False, 
        'cells_target':cells_ref, 
        'genes':set(adata.var_names if args.genes_eval is None 
                    else adata.var_names[adata.var[args.genes_eval]])
    },
    'query-pred_query':{
        'cells_in':cells_ref, 
        'switch_system':True, 
        'cells_target':cells_query, 
        'genes':set(adata.var_names if args.genes_eval is None 
                    else adata.var_names[adata.var[args.genes_eval]])
    },
}

# %% [markdown]
# ### Training

# %%
train_filter=(adata.obs[args.group_key]!=args.group_translate).values |\
             (adata.obs[args.system_key]!=args.system_translate).values
print('N cells for training:',train_filter.sum())

# %%
# Setup adata
adata_training = XXJointModel.setup_anndata(
    adata=adata[train_filter,:],
    system_key=args.system_key,
    group_key=args.group_key,
    categorical_covariate_keys=[args.batch_key],
)
adata_eval=XXJointModel.setup_anndata(
    adata=adata,
    system_key=args.system_key,
    group_key=args.group_key,
    categorical_covariate_keys=[args.batch_key],
)

# %%
# Model
model = XXJointModel(
    adata=adata_training,
    mixup_alpha=args.mixup_alpha,
    system_decoders=args.system_decoders,
    prior=args.prior, 
    n_prior_components=args.n_prior_components,
    z_dist_metric = args.z_dist_metric,
    n_layers=args.n_layers,
    n_hidden=args.n_hidden,
    adata_eval=adata_eval,
)
#wandb.watch(model.module, log="all",log_freq= 1)
# Train
max_epochs=args.max_epochs
model.train(max_epochs=max_epochs,
            log_every_n_steps=1,
            check_val_every_n_epoch=1,
            val_check_interval=1.0 if args.log_on_epoch else 1, # Used irregardles of logging
            train_size=args.train_size,
            callbacks=[pl.callbacks.LearningRateMonitor(logging_interval='epoch')] ,
            plan_kwargs={
                'optimizer':args.optimizer,
                'lr':args.lr,
                'reduce_lr_on_plateau':args.reduce_lr_on_plateau,
                'lr_scheduler_metric':args.lr_scheduler_metric,
                'lr_patience':args.lr_patience,
                'lr_factor':args.lr_factor,
                'lr_min':args.lr_min,
                'lr_threshold_mode':args.lr_threshold_mode,
                'lr_threshold':args.lr_threshold,
                'log_on_epoch':args.log_on_epoch,
                'log_on_step':not args.log_on_epoch,
                'loss_weights':{
                   'kl_weight':args.kl_weight,
                   'kl_cycle_weight':args.kl_cycle_weight,
                   'reconstruction_weight':args.reconstruction_weight,
                   'reconstruction_mixup_weight':args.reconstruction_mixup_weight,
                   'reconstruction_cycle_weight':args.reconstruction_cycle_weight,
                   'z_distance_cycle_weight':args.z_distance_cycle_weight,
                   'translation_corr_weight':args.translation_corr_weight,
                   'z_contrastive_weight':args.z_contrastive_weight,
               
           }})

# %%
# Call this as else params and gradients are not logged 
#wandb.log({'end':0})

# %% [markdown]
# C: Couldnt see anything odd with weights in wandb when model was failing

# %% [markdown]
# C: Added quick fix torch.nan_to_num - this prevents failure

# %%
