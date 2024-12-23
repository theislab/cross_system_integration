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

import anndata as ad
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
import seaborn as sns

# %%
import torch
from cross_system_integration.model._xxjointmodel import XXJointModel
from pytorch_lightning.callbacks.base import Callback

# %% [markdown]
# ## Config & utils

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
parser.add_argument('-i', '--init-method', default='default', choices=['default', 'random', 'system_0', 'system_1', 'most_balanced'])

if hasattr(sys, 'ps1'):
    args = parser.parse_args([
        '-e', '50',
        '-n', '10',
        # '-m', '10000',
        # '-f',
        '-i', 'system_0',
    ])
else:
    args = parser.parse_args()
print(args)

# %%
MAX_EPOCHS = args.max_epochs
TRAINABLE_PRIORS = not args.fixed_priors
N_PRIOR_COMPONENTS = args.n_priors
N_SAMPLES_TO_PLOT = args.n_cell_plot
INIT_METHOD = args.init_method

# %%
path_data = os.path.expanduser("~/data/cs_integration/combined_orthologuesHVG.h5ad")
output_filename = os.path.expanduser(f"~/io/cs_integration/vamp_testing_pancreas_combined_orthologuesHVG_n_prior_{N_PRIOR_COMPONENTS}_trainable_prior_{TRAINABLE_PRIORS}_init_{INIT_METHOD}")
sc.settings.figdir = output_filename

# %%
print(f"Output {'already exists' if os.path.exists(output_filename) else 'does not exist'}")


# %% [markdown]
# ## Some Utils

# %%
class PriorInspectionCallback(Callback):
    """
    A PyTorch Lightning Callback for logging the pseudoinput information during training.

    This callback records the pseudoinput information (data and covariates) from a PyTorch Lightning module

    Attributes:
        prior_history (list): A list to store pseudoinput information at different training stages.

    Methods:
        on_train_start(trainer, pl_module):
            Called at the beginning of training to log the initial pseudoinput information.
        
        on_train_epoch_end(trainer, pl_module):
            Called at the end of each training epoch to log the current pseudoinput information.
    """
    def __init__(self):
        super().__init__()
        self.prior_history = []

    def _log_priors(self, trainer, pl_module):
        """
        Logs the pseudoinputs and covariates of them from the given PyTorch Lightning module.

        Args:
            trainer: The PyTorch Lightning trainer.
            pl_module: The PyTorch Lightning module containing the pseudoinput information.
        """
        self.prior_history.append(tuple([
            pl_module.module.prior.u.detach().cpu().numpy(),
            pl_module.module.prior.u_cov.detach().cpu().numpy()
        ]))
    
    def on_train_start(self, trainer, pl_module):
        self._log_priors(trainer, pl_module)

    def on_train_epoch_end(self, trainer, pl_module):
        self._log_priors(trainer, pl_module)


# %%
def random_init_algorithm(adata, n_priors):
    """
    Randomly select `n_priors` observations from the given AnnData object as prior init values.

    Args:
        adata: An AnnData object containing observations.
        n_priors (int): The number of priors to select.

    Returns:
        np.ndarray: An array of selected observation indices.
    """
    return np.random.choice(np.arange(adata.n_obs), size=n_priors, replace=False)


def random_from_system_i(adata, n_priors, i=0):
    """
    Randomly select `n_priors` observations from a specific system (i) within the given AnnData object as prior init vlaues.

    Args:
        adata: An AnnData object containing observations.
        n_priors (int): The number of priors to select.
        i (int): The system index to select observations from (default is 0).

    Returns:
        np.ndarray: An array of selected observation indices from the specified system.
    """
    obs = adata.obs.copy()
    obs['uid'] = np.arange(obs.shape[0])
    return obs[obs[SYSTEM_KEY] == i]['uid'].sample(N_PRIOR_COMPONENTS, replace=False).to_numpy()


def most_balanced_algorithm(adata, n_priors):
    """
    Select `n_priors` observations with a balanced distribution between two systems in the AnnData object.
    Please note that this function only accepts two systems.

    Args:
        adata: An AnnData object containing observations with a 'SYSTEM_KEY' column.
        n_priors (int): The number of priors to select.

    Returns:
        np.ndarray: An array of selected observation indices with a balanced distribution between two systems.
    """
    system_keys = adata.obs[SYSTEM_KEY].unique()
    assert len(system_keys) == 2
    assert 0 in system_keys
    assert 1 in system_keys

    obs = adata.obs.copy()
    obs['uid'] = np.arange(obs.shape[0])

    return np.where(
        np.arange(N_PRIOR_COMPONENTS) % 2 == 0,
        obs[obs[SYSTEM_KEY] == 0]['uid'].sample(N_PRIOR_COMPONENTS, replace=False).to_numpy(),
        obs[obs[SYSTEM_KEY] == 1]['uid'].sample(N_PRIOR_COMPONENTS, replace=False).to_numpy(),
    )
    

INIT_ALGORITHMS = {
    'default': lambda adata, n_priors: None,
    'random': random_init_algorithm,
    'system_0': lambda adata, n_priors: random_from_system_i(adata, n_priors, 0),
    'system_1': lambda adata, n_priors: random_from_system_i(adata, n_priors, 1),
    'most_balanced': most_balanced_algorithm,
}

# %%
adata=sc.read(path_data)
adata

# %% [markdown]
# ## Train model (inlc. saving prior components positions)

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
    
    model = XXJointModel(adata=adata_training, prior='vamp', 
                         n_prior_components=N_PRIOR_COMPONENTS,
                         pseudoinputs_data_init=True,
                         pseudoinputs_data_indices=INIT_ALGORITHMS[INIT_METHOD](adata_training, N_PRIOR_COMPONENTS),
                         trainable_priors=TRAINABLE_PRIORS,
                         encode_pseudoinputs_on_eval_mode=True,)
    # Inspect prior component movement during training
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
    model.save(output_filename, overwrite=True)

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
plt.savefig(os.path.join(output_filename, 'losses.png'))
fig.tight_layout()

# %% [markdown]
# ## Latent data representation

# %%
# Get latent rep
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
sc.tl.pca(embed)
sc.tl.umap(embed)
embed.write(os.path.join(output_filename, 'embed.h5ad'))

# %%
embed = sc.read(os.path.join(output_filename, 'embed.h5ad'))

# %%
# Most probable prior
prior_probs = ad.AnnData(
    model.get_prior_probs(
        adata=adata_training,
        indices=None,
        batch_size=None,
        as_numpy=True),
    obs=adata_training.obs
)
prior_probs

# %%
embed.obsm['prior_probs'] = prior_probs[embed.obs.index].X
embed.obs['most_probable_prior_p'] = embed.obsm['prior_probs'].max(axis=1)
embed.obs['most_probable_prior_id'] = pd.Categorical(embed.obsm['prior_probs'].argmax(axis=1))

# %%
embed.write(os.path.join(output_filename, 'embed.h5ad'))

# %%
rcParams['figure.figsize']=(8,8)
sc.pl.umap(embed, color=[CT_KEY, *BATCH_KEYS, 'species', 'most_probable_prior_p', 'most_probable_prior_id'], s=10, wspace=0.5, 
           save='_umap_cells.png', ncols=1)

# %%
rcParams['figure.figsize']=(5,3)
_=plt.violinplot(embed.to_df())
plt.savefig(os.path.join(output_filename, 'latent_violin.png'))
plt.show()

# %%
rcParams['figure.figsize']=(8,8)
sc.pl.pca(embed, color=[CT_KEY, *BATCH_KEYS, 'species','most_probable_prior_p', 'most_probable_prior_id',], 
          s=10, wspace=0.5, components=['1,2', '3,4', '5,6', '7,8'], ncols=4,
          save='_pca_all.png'
         )

# %% [markdown]
# ## Pseudoinputs embedding

# %%
# Encode pseudoinputs
prior_history = model.train_prior_history_
n_steps = len(prior_history)
n_points = prior_history[0][0].shape[0]

prior_x = np.concatenate([x[0] for x in prior_history])
prior_cov = np.concatenate([x[1] for x in prior_history])

embed_pseudoinputs = model.module.encoder(x=torch.tensor(prior_x, device=model.module.device),
                                          cov=torch.tensor(prior_cov, device=model.module.device))['y_m'].detach().cpu().numpy()
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
embed_all.write(os.path.join(output_filename, 'embed_all.h5ad'))

# %%
embed_all = sc.read(os.path.join(output_filename, 'embed_all.h5ad'))

# %%
embed_final = embed_all[embed_all.obs['pseudoinput_time'].isna() | (embed_all.obs['pseudoinput_time'] == n_steps - 1)]

# %%
rcParams['figure.figsize']=(8,8)
sc.pl.pca(embed_final, color=['input_type', CT_KEY, *BATCH_KEYS, 'species',
                              'most_probable_prior_p', 'most_probable_prior_id',
                             ], s=10, wspace=0.5, components=['1,2', '3,4', '5,6', '7,8'], ncols=4,
          save='_pca_all.png'
         )

# %%
rcParams['figure.figsize']=(8,8)
sc.pl.umap(embed_final, color=['input_type', CT_KEY, *BATCH_KEYS, 'species',
                               'most_probable_prior_p', 'most_probable_prior_id',
                              ], s=10, wspace=0.5,
           save='_umap_all.png'
          )

# %%
rcParams['figure.figsize']=(8,8)
sc.pl.pca(embed_all, color=['input_type', 'pseudoinput_time', 'pseudoinput_id',
                            'most_probable_prior_p', 'most_probable_prior_id',], s=10, wspace=0.5, components=['1,2', '3,4', '5,6', '7,8'], ncols=4,
          save='_pca_pseudoinput_time.png'
         )

# %%
rcParams['figure.figsize']=(8,8)
sc.pl.umap(embed_all, color=['input_type', 'pseudoinput_time', 'pseudoinput_id',
                             'most_probable_prior_p', 'most_probable_prior_id'], s=10, wspace=0.5,
           save='_umap_pseudoinput_time.png'
          )

# %% [raw]
#

# %%
