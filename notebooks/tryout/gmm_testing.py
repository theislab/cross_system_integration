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
import sys
import argparse

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

# %% [markdown]
# ## Config

# %%
SYSTEM_KEY = 'system'
BATCH_KEYS = ['batch']
CT_KEY = 'cell_type_eval'

# %%
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--max-epochs', type=int, default=50)
parser.add_argument('-n', '--n-priors', type=int, default=100)
parser.add_argument('-m', '--n-cell_plot', type=int, default=None)
parser.add_argument('-f', '--fixed-priors', action='store_true')
parser.add_argument('-d', '--init-priors-from-data', action='store_true')
parser.add_argument('--dry-run', action='store_true')

if hasattr(sys, 'ps1'):
    args = parser.parse_args([
        '-e', '50',
        '-n', '10',
        '-m', '10000',
        '-d',
        # '-f',
        '--dry-run',
    ])
else:
    args = parser.parse_args()
print(args)

# %%
MAX_EPOCHS = args.max_epochs
TRAINABLE_PRIORS = not args.fixed_priors
N_PRIOR_COMPONENTS = args.n_priors
N_SAMPLES_TO_PLOT = args.n_cell_plot
INIT_PRIORS_FROM_DATA = args.init_priors_from_data
DRY_RUN = args.dry_run

# %%
path_data = os.path.expanduser("~/data/cs_integration/combined_orthologuesHVG.h5ad")
output_filename = os.path.expanduser(f"~/io/cs_integration/gmm_testing_pancreas_combined_orthologuesHVG_n_prior_{N_PRIOR_COMPONENTS}_trainable_prior_{TRAINABLE_PRIORS}_data_init_{INIT_PRIORS_FROM_DATA}")
sc.settings.figdir = output_filename


# %% [markdown]
# ## Some Utils

# %%
class PriorInspectionCallback(Callback):
    def __init__(self):
        super().__init__()
        self.prior_history = []

    def _log_priors(self, trainer, pl_module):
        self.prior_history.append(tuple([
            pl_module.module.prior.p_m.detach().cpu().numpy(),
            pl_module.module.prior.p_v.detach().cpu().numpy()
        ]))
    
    def on_train_start(self, trainer, pl_module):
        self._log_priors(trainer, pl_module)

    def on_train_epoch_end(self, trainer, pl_module):
        self._log_priors(trainer, pl_module)


# %% [markdown]
# ## Pancreas

# %%
adata=sc.read(path_data)
adata

# %% [markdown]
# ### cVAE - VampPrior

# %%
adata_training = adata.copy()

# %%
print(f"Target: {output_filename}")

if os.path.exists(output_filename):
    model = XXJointModel.load(output_filename, adata=adata_training)
else:
    XXJointModel.setup_anndata(
        adata=adata_training,
        system_key=SYSTEM_KEY,
        #class_key=CT_KEY,
        categorical_covariate_keys=BATCH_KEYS,
    )
    
    model = XXJointModel(adata=adata_training, prior='gmm', 
                         n_prior_components=N_PRIOR_COMPONENTS,
                         pseudoinputs_data_init=INIT_PRIORS_FROM_DATA,
                         trainable_priors=TRAINABLE_PRIORS,)
    
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
    model.train_logger_history_ = model.trainer.logger.history
    model.train_prior_history_ = prior_inspection_callback.prior_history
    DRY_RUN or model.save(output_filename, overwrite=True)

# %%
# Plot all loses
logger_history = model.train_logger_history_
losses = [k for k in logger_history.keys() 
        if '_step' not in k and '_epoch' not in k]
fig,axs = plt.subplots(2,len(losses),figsize=(len(losses)*3,4))
for ax_i,l in enumerate(losses):
    axs[0,ax_i].plot(
        logger_history[l].index,
        logger_history[l][l])
    axs[0,ax_i].set_title(l)
    axs[1,ax_i].plot(
        logger_history[l].index[20:],
        logger_history[l][l][20:])
DRY_RUN or plt.savefig(os.path.join(output_filename, 'losses.png'))
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
DRY_RUN or embed.write(os.path.join(output_filename, 'embed.h5ad'))

# %%
if not DRY_RUN:
    embed = sc.read(os.path.join(output_filename, 'embed.h5ad'))

# %%
rcParams['figure.figsize']=(8,8)
sc.pl.umap(embed, color=[CT_KEY, *BATCH_KEYS, 'species'], s=10, wspace=0.5, 
           save='_umap_cells.png' if not DRY_RUN else None)

# %%
rcParams['figure.figsize']=(5,3)
_=plt.violinplot(embed.to_df())
DRY_RUN or plt.savefig(os.path.join(output_filename, 'latent_violin.png'))
plt.show()

# %%
# Encode pseudoinputs
prior_history = model.train_prior_history_
n_steps = len(prior_history)
n_points = prior_history[0][0].shape[0]

# %%
embed_pseudoinputs = np.concatenate([x[0] for x in prior_history])
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
DRY_RUN or embed_all.write(os.path.join(output_filename, 'embed_all.h5ad'))

# %%
if not DRY_RUN:
    embed_all = sc.read(os.path.join(output_filename, 'embed_all.h5ad'))

# %%
embed_final = embed_all[embed_all.obs['pseudoinput_time'].isna() | (embed_all.obs['pseudoinput_time'] == n_steps - 1)]

# %%
rcParams['figure.figsize']=(8,8)
sc.pl.pca(embed_final, color=['input_type', CT_KEY, *BATCH_KEYS, 'species'], s=10, wspace=0.5, components=['1,2', '3,4', '5,6', '7,8'], ncols=4,
          save='_pca_all.png' if not DRY_RUN else None)

# %%
rcParams['figure.figsize']=(8,8)
sc.pl.umap(embed_final, color=['input_type', CT_KEY, *BATCH_KEYS, 'species'], s=10, wspace=0.5,
           save='_umap_all.png' if not DRY_RUN else None)

# %%
rcParams['figure.figsize']=(8,8)
sc.pl.pca(embed_all, color=['input_type', 'pseudoinput_time', 'pseudoinput_id'], s=10, wspace=0.5, components=['1,2', '3,4', '5,6', '7,8'], ncols=4,
          save='_pca_pseudoinput_time.png' if not DRY_RUN else None)

# %%
rcParams['figure.figsize']=(8,8)
sc.pl.umap(embed_all, color=['input_type', 'pseudoinput_time', 'pseudoinput_id'], s=10, wspace=0.5,
           save='_umap_pseudoinput_time.png' if not DRY_RUN else None)

# %%

# %%

# %%
