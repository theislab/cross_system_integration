# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.0
#   kernelspec:
#     display_name: cs_integration
#     language: python
#     name: cs_integration
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import os

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
import torch
from cross_system_integration.model._xxjointmodel import XXJointModel
from pytorch_lightning.callbacks.base import Callback

# %%
path_data = os.path.expanduser("~/data/cs_integration/combined_orthologuesHVG.h5ad")


# %% [markdown]
# ## Some Utils

# %%
class PriorInspectionCallback(Callback):
    def __init__(self):
        super().__init__()
        self.prior_history = []

    def _log_priors(self, trainer, pl_module):
        self.prior_history.append(tuple([
            pl_module.module.prior.u.detach().cpu().numpy(),
            pl_module.module.prior.u_cov.detach().cpu().numpy()
        ]))
    
    def on_train_start(self, trainer, pl_module):
        self._log_priors(trainer, pl_module)

    def on_train_epoch_end(self, trainer, pl_module):
        self._log_priors(trainer, pl_module)


# %% [markdown]
# ## Pancreas

# %%
adata=sc.read(path_data)
adata.obs['uid'] = np.arange(adata.n_obs)
adata

# %%
MAX_EPOCHS = 50
SYSTEM_KEY = 'system'
BATCH_KEYS = ['batch']
CT_KEY = 'cell_type_eval'

N_SAMPLES_TO_PLOT = 10000

# %% [markdown]
# ### cVAE - VampPrior (n_components=100) with mouse init

# %%
N_PRIOR_COMPONENTS = 100

# %%
adata_training = XXJointModel.setup_anndata(
    adata=adata,
    system_key=SYSTEM_KEY,
    #class_key=CT_KEY,
    categorical_covariate_keys=BATCH_KEYS,
)

# %%
pseudoinputs_data_indices = (
    adata_training.obs[adata_training.obs[SYSTEM_KEY] == 0]
    ['uid'].sample(N_PRIOR_COMPONENTS)
)
pseudoinputs_data_indices

# %%
model = XXJointModel(adata=adata_training, prior='vamp', 
                     n_prior_components=N_PRIOR_COMPONENTS,
                     pseudoinputs_data_init=True,
                     pseudoinputs_data_indices=pseudoinputs_data_indices,
                     trainable_priors=True)

prior_inspection_callback = PriorInspectionCallback()
model.train(max_epochs=MAX_EPOCHS,
            check_val_every_n_epoch=1,
            plan_kwargs={'loss_weights':{
                'reconstruction_mixup_weight':0,
                'reconstruction_cycle_weight':0,
                'kl_cycle_weight':0,
                'z_distance_cycle_weight':0,
                'translation_corr_weight':0,
                'z_contrastive_weight':0,
            }},
            callbacks=[prior_inspection_callback])

# %%
# Plot all loses
losses = [k for k in model.trainer.logger.history.keys() 
        if '_step' not in k and '_epoch' not in k]
fig,axs = plt.subplots(2,len(losses),figsize=(len(losses)*3,4))
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

embed = sc.AnnData(embed, obs=adata_training.obs)
embed.obs['species'] = embed.obs[SYSTEM_KEY].map({0:'mm', 1:'hs'})

np.random.seed(0)
random_indices = np.random.permutation(list(range(embed.shape[0])))
embed = embed[random_indices, :]
if N_SAMPLES_TO_PLOT is not None:
     embed = embed[:N_SAMPLES_TO_PLOT, :]
embed = embed.copy()

# %%
sc.pp.neighbors(embed, use_rep='X')
sc.tl.umap(embed)

# %%
rcParams['figure.figsize']=(8,8)
sc.pl.umap(embed, color=[CT_KEY, *BATCH_KEYS, 'species'], s=10, wspace=0.5)

# %%
rcParams['figure.figsize']=(5,3)
_=plt.violinplot(embed.to_df())

# %%
# Encode pseudoinputs
n_steps = len(prior_inspection_callback.prior_history)
n_points = prior_inspection_callback.prior_history[0][0].shape[0]

prior_x = np.concatenate([x[0] for x in prior_inspection_callback.prior_history])
prior_cov = np.concatenate([x[1] for x in prior_inspection_callback.prior_history])

embed_pseudoinputs = model.module.encoder(x=torch.tensor(prior_x, device=model.module.device),
                                          cov=torch.tensor(prior_cov, device=model.module.device))['y'].detach().cpu().numpy()
embed_pseudoinputs = sc.AnnData(embed_pseudoinputs)
embed_pseudoinputs.obs['pseudoinput_id'] = [i % n_points for i in range(n_steps * n_points)]
embed_pseudoinputs.obs['pseudoinput_time'] = [i // n_points for i in range(n_steps * n_points)]

# %%
embed.obs['input_type'] = 'expr'
embed_pseudoinputs.obs['input_type'] = 'pseudo'
embed_all = sc.concat([embed, embed_pseudoinputs], merge='unique', join='outer')

# %%
sc.pp.neighbors(embed_all, use_rep='X')
sc.tl.pca(embed_all)
sc.tl.umap(embed_all)

# %%
embed_final = embed_all[embed_all.obs['pseudoinput_time'].isna() | (embed_all.obs['pseudoinput_time'] == n_steps - 1)]

# %%
rcParams['figure.figsize']=(8,8)
sc.pl.pca(embed_final, color=['input_type', CT_KEY, *BATCH_KEYS, 'species'], s=10, wspace=0.5)

# %%
rcParams['figure.figsize']=(8,8)
sc.pl.umap(embed_final, color=['input_type', CT_KEY, *BATCH_KEYS, 'species'], s=10, wspace=0.5)

# %%
rcParams['figure.figsize']=(8,8)
sc.pl.pca(embed_all, color=['input_type', 'pseudoinput_time'], s=10, wspace=0.5)

# %%
rcParams['figure.figsize']=(8,8)
sc.pl.umap(embed_all, color=['input_type', 'pseudoinput_time'], s=10, wspace=0.5)

# %%
rcParams['figure.figsize']=(8,8)
sc.pl.umap(embed_all, color=['input_type', 'pseudoinput_id'], s=10, wspace=0.5)

# %% [markdown]
# ### cVAE - VampPrior (n_components=10) with mouse init

# %%
N_PRIOR_COMPONENTS = 10

# %%
adata_training = XXJointModel.setup_anndata(
    adata=adata,
    system_key=SYSTEM_KEY,
    #class_key=CT_KEY,
    categorical_covariate_keys=BATCH_KEYS,
)

# %%
pseudoinputs_data_indices = (
    adata_training.obs[adata_training.obs[SYSTEM_KEY] == 0]
    ['uid'].sample(N_PRIOR_COMPONENTS)
)
pseudoinputs_data_indices

# %%
model = XXJointModel(adata=adata_training, prior='vamp', 
                     n_prior_components=N_PRIOR_COMPONENTS,
                     pseudoinputs_data_init=True,
                     pseudoinputs_data_indices=pseudoinputs_data_indices,
                     trainable_priors=True)

prior_inspection_callback = PriorInspectionCallback()
model.train(max_epochs=MAX_EPOCHS,
            check_val_every_n_epoch=1,
            plan_kwargs={'loss_weights':{
                'reconstruction_mixup_weight':0,
                'reconstruction_cycle_weight':0,
                'kl_cycle_weight':0,
                'z_distance_cycle_weight':0,
                'translation_corr_weight':0,
                'z_contrastive_weight':0,
            }},
            callbacks=[prior_inspection_callback])

# %%
# Plot all loses
losses = [k for k in model.trainer.logger.history.keys() 
        if '_step' not in k and '_epoch' not in k]
fig,axs = plt.subplots(2,len(losses),figsize=(len(losses)*3,4))
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

embed = sc.AnnData(embed, obs=adata_training.obs)
embed.obs['species'] = embed.obs[SYSTEM_KEY].map({0:'mm', 1:'hs'})

np.random.seed(0)
random_indices = np.random.permutation(list(range(embed.shape[0])))
embed = embed[random_indices, :]
if N_SAMPLES_TO_PLOT is not None:
     embed = embed[:N_SAMPLES_TO_PLOT, :]
embed = embed.copy()

# %%
sc.pp.neighbors(embed, use_rep='X')
sc.tl.umap(embed)

# %%
rcParams['figure.figsize']=(8,8)
sc.pl.umap(embed, color=[CT_KEY, *BATCH_KEYS, 'species'], s=10, wspace=0.5)

# %%
rcParams['figure.figsize']=(5,3)
_=plt.violinplot(embed.to_df())

# %%
# Encode pseudoinputs
n_steps = len(prior_inspection_callback.prior_history)
n_points = prior_inspection_callback.prior_history[0][0].shape[0]

prior_x = np.concatenate([x[0] for x in prior_inspection_callback.prior_history])
prior_cov = np.concatenate([x[1] for x in prior_inspection_callback.prior_history])

embed_pseudoinputs = model.module.encoder(x=torch.tensor(prior_x, device=model.module.device),
                                          cov=torch.tensor(prior_cov, device=model.module.device))['y'].detach().cpu().numpy()
embed_pseudoinputs = sc.AnnData(embed_pseudoinputs)
embed_pseudoinputs.obs['pseudoinput_id'] = [i % n_points for i in range(n_steps * n_points)]
embed_pseudoinputs.obs['pseudoinput_time'] = [i // n_points for i in range(n_steps * n_points)]

# %%
embed.obs['input_type'] = 'expr'
embed_pseudoinputs.obs['input_type'] = 'pseudo'
embed_all = sc.concat([embed, embed_pseudoinputs], merge='unique', join='outer')

# %%
sc.pp.neighbors(embed_all, use_rep='X')
sc.tl.pca(embed_all)
sc.tl.umap(embed_all)

# %%
embed_final = embed_all[embed_all.obs['pseudoinput_time'].isna() | (embed_all.obs['pseudoinput_time'] == n_steps - 1)]

# %%
rcParams['figure.figsize']=(8,8)
sc.pl.pca(embed_final, color=['input_type', CT_KEY, *BATCH_KEYS, 'species'], s=10, wspace=0.5)

# %%
rcParams['figure.figsize']=(8,8)
sc.pl.umap(embed_final, color=['input_type', CT_KEY, *BATCH_KEYS, 'species'], s=10, wspace=0.5)

# %%
rcParams['figure.figsize']=(8,8)
sc.pl.pca(embed_all, color=['input_type', 'pseudoinput_time'], s=10, wspace=0.5)

# %%
rcParams['figure.figsize']=(8,8)
sc.pl.umap(embed_all, color=['input_type', 'pseudoinput_time'], s=10, wspace=0.5)

# %%
rcParams['figure.figsize']=(8,8)
sc.pl.umap(embed_all, color=['input_type', 'pseudoinput_id'], s=10, wspace=0.5)

# %%

# %%

# %% [markdown]
# ### cVAE - VampPrior (n_components=100) with human init

# %%
N_PRIOR_COMPONENTS = 100

# %%
adata_training = XXJointModel.setup_anndata(
    adata=adata,
    system_key=SYSTEM_KEY,
    #class_key=CT_KEY,
    categorical_covariate_keys=BATCH_KEYS,
)

# %%
pseudoinputs_data_indices = (
    adata_training.obs[adata_training.obs[SYSTEM_KEY] == 1]
    ['uid'].sample(N_PRIOR_COMPONENTS)
)
pseudoinputs_data_indices

# %%
model = XXJointModel(adata=adata_training, prior='vamp', 
                     n_prior_components=N_PRIOR_COMPONENTS,
                     pseudoinputs_data_init=True,
                     pseudoinputs_data_indices=pseudoinputs_data_indices,
                     trainable_priors=True)

prior_inspection_callback = PriorInspectionCallback()
model.train(max_epochs=MAX_EPOCHS,
            check_val_every_n_epoch=1,
            plan_kwargs={'loss_weights':{
                'reconstruction_mixup_weight':0,
                'reconstruction_cycle_weight':0,
                'kl_cycle_weight':0,
                'z_distance_cycle_weight':0,
                'translation_corr_weight':0,
                'z_contrastive_weight':0,
            }},
            callbacks=[prior_inspection_callback])

# %%
# Plot all loses
losses = [k for k in model.trainer.logger.history.keys() 
        if '_step' not in k and '_epoch' not in k]
fig,axs = plt.subplots(2,len(losses),figsize=(len(losses)*3,4))
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

embed = sc.AnnData(embed, obs=adata_training.obs)
embed.obs['species'] = embed.obs[SYSTEM_KEY].map({0:'mm', 1:'hs'})

np.random.seed(0)
random_indices = np.random.permutation(list(range(embed.shape[0])))
embed = embed[random_indices, :]
if N_SAMPLES_TO_PLOT is not None:
     embed = embed[:N_SAMPLES_TO_PLOT, :]
embed = embed.copy()

# %%
sc.pp.neighbors(embed, use_rep='X')
sc.tl.umap(embed)

# %%
rcParams['figure.figsize']=(8,8)
sc.pl.umap(embed, color=[CT_KEY, *BATCH_KEYS, 'species'], s=10, wspace=0.5)

# %%
rcParams['figure.figsize']=(5,3)
_=plt.violinplot(embed.to_df())

# %%
# Encode pseudoinputs
n_steps = len(prior_inspection_callback.prior_history)
n_points = prior_inspection_callback.prior_history[0][0].shape[0]

prior_x = np.concatenate([x[0] for x in prior_inspection_callback.prior_history])
prior_cov = np.concatenate([x[1] for x in prior_inspection_callback.prior_history])

embed_pseudoinputs = model.module.encoder(x=torch.tensor(prior_x, device=model.module.device),
                                          cov=torch.tensor(prior_cov, device=model.module.device))['y'].detach().cpu().numpy()
embed_pseudoinputs = sc.AnnData(embed_pseudoinputs)
embed_pseudoinputs.obs['pseudoinput_id'] = [i % n_points for i in range(n_steps * n_points)]
embed_pseudoinputs.obs['pseudoinput_time'] = [i // n_points for i in range(n_steps * n_points)]

# %%
embed.obs['input_type'] = 'expr'
embed_pseudoinputs.obs['input_type'] = 'pseudo'
embed_all = sc.concat([embed, embed_pseudoinputs], merge='unique', join='outer')

# %%
sc.pp.neighbors(embed_all, use_rep='X')
sc.tl.pca(embed_all)
sc.tl.umap(embed_all)

# %%
embed_final = embed_all[embed_all.obs['pseudoinput_time'].isna() | (embed_all.obs['pseudoinput_time'] == n_steps - 1)]

# %%
rcParams['figure.figsize']=(8,8)
sc.pl.pca(embed_final, color=['input_type', CT_KEY, *BATCH_KEYS, 'species'], s=10, wspace=0.5)

# %%
rcParams['figure.figsize']=(8,8)
sc.pl.umap(embed_final, color=['input_type', CT_KEY, *BATCH_KEYS, 'species'], s=10, wspace=0.5)

# %%
rcParams['figure.figsize']=(8,8)
sc.pl.pca(embed_all, color=['input_type', 'pseudoinput_time'], s=10, wspace=0.5)

# %%
rcParams['figure.figsize']=(8,8)
sc.pl.umap(embed_all, color=['input_type', 'pseudoinput_time'], s=10, wspace=0.5)

# %%
rcParams['figure.figsize']=(8,8)
sc.pl.umap(embed_all, color=['input_type', 'pseudoinput_id'], s=10, wspace=0.5)

# %% [markdown]
# ### cVAE - VampPrior (n_components=10) with human init

# %%
N_PRIOR_COMPONENTS = 10

# %%
adata_training = XXJointModel.setup_anndata(
    adata=adata,
    system_key=SYSTEM_KEY,
    #class_key=CT_KEY,
    categorical_covariate_keys=BATCH_KEYS,
)

# %%
pseudoinputs_data_indices = (
    adata_training.obs[adata_training.obs[SYSTEM_KEY] == 1]
    ['uid'].sample(N_PRIOR_COMPONENTS)
)
pseudoinputs_data_indices

# %%
model = XXJointModel(adata=adata_training, prior='vamp', 
                     n_prior_components=N_PRIOR_COMPONENTS,
                     pseudoinputs_data_init=True,
                     pseudoinputs_data_indices=pseudoinputs_data_indices,
                     trainable_priors=True)

prior_inspection_callback = PriorInspectionCallback()
model.train(max_epochs=MAX_EPOCHS,
            check_val_every_n_epoch=1,
            plan_kwargs={'loss_weights':{
                'reconstruction_mixup_weight':0,
                'reconstruction_cycle_weight':0,
                'kl_cycle_weight':0,
                'z_distance_cycle_weight':0,
                'translation_corr_weight':0,
                'z_contrastive_weight':0,
            }},
            callbacks=[prior_inspection_callback])

# %%
# Plot all loses
losses = [k for k in model.trainer.logger.history.keys() 
        if '_step' not in k and '_epoch' not in k]
fig,axs = plt.subplots(2,len(losses),figsize=(len(losses)*3,4))
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

embed = sc.AnnData(embed, obs=adata_training.obs)
embed.obs['species'] = embed.obs[SYSTEM_KEY].map({0:'mm', 1:'hs'})

np.random.seed(0)
random_indices = np.random.permutation(list(range(embed.shape[0])))
embed = embed[random_indices, :]
if N_SAMPLES_TO_PLOT is not None:
     embed = embed[:N_SAMPLES_TO_PLOT, :]
embed = embed.copy()

# %%
sc.pp.neighbors(embed, use_rep='X')
sc.tl.umap(embed)

# %%
rcParams['figure.figsize']=(8,8)
sc.pl.umap(embed, color=[CT_KEY, *BATCH_KEYS, 'species'], s=10, wspace=0.5)

# %%
rcParams['figure.figsize']=(5,3)
_=plt.violinplot(embed.to_df())

# %%
# Encode pseudoinputs
n_steps = len(prior_inspection_callback.prior_history)
n_points = prior_inspection_callback.prior_history[0][0].shape[0]

prior_x = np.concatenate([x[0] for x in prior_inspection_callback.prior_history])
prior_cov = np.concatenate([x[1] for x in prior_inspection_callback.prior_history])

embed_pseudoinputs = model.module.encoder(x=torch.tensor(prior_x, device=model.module.device),
                                          cov=torch.tensor(prior_cov, device=model.module.device))['y'].detach().cpu().numpy()
embed_pseudoinputs = sc.AnnData(embed_pseudoinputs)
embed_pseudoinputs.obs['pseudoinput_id'] = [i % n_points for i in range(n_steps * n_points)]
embed_pseudoinputs.obs['pseudoinput_time'] = [i // n_points for i in range(n_steps * n_points)]

# %%
embed.obs['input_type'] = 'expr'
embed_pseudoinputs.obs['input_type'] = 'pseudo'
embed_all = sc.concat([embed, embed_pseudoinputs], merge='unique', join='outer')

# %%
sc.pp.neighbors(embed_all, use_rep='X')
sc.tl.pca(embed_all)
sc.tl.umap(embed_all)

# %%
embed_final = embed_all[embed_all.obs['pseudoinput_time'].isna() | (embed_all.obs['pseudoinput_time'] == n_steps - 1)]

# %%
rcParams['figure.figsize']=(8,8)
sc.pl.pca(embed_final, color=['input_type', CT_KEY, *BATCH_KEYS, 'species'], s=10, wspace=0.5)

# %%
rcParams['figure.figsize']=(8,8)
sc.pl.umap(embed_final, color=['input_type', CT_KEY, *BATCH_KEYS, 'species'], s=10, wspace=0.5)

# %%
rcParams['figure.figsize']=(8,8)
sc.pl.pca(embed_all, color=['input_type', 'pseudoinput_time'], s=10, wspace=0.5)

# %%
rcParams['figure.figsize']=(8,8)
sc.pl.umap(embed_all, color=['input_type', 'pseudoinput_time'], s=10, wspace=0.5)

# %%
rcParams['figure.figsize']=(8,8)
sc.pl.umap(embed_all, color=['input_type', 'pseudoinput_id'], s=10, wspace=0.5)

# %%
