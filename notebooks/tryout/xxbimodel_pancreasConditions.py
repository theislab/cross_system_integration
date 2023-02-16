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
path_data='/lustre/groups/ml01/workspace/karin.hrovatin/data/cross_species_prediction/pancreas_example/'

# %%
adata=sc.read(path_data+'combined_conditions_orthologues_full.h5ad')

# %%
adata_training = XXJointModel.setup_anndata(
    adata=adata,
    system_key='system',
    #group_key='CellType',
    categorical_covariate_keys=['batch'],
)

# %% [markdown]
# ## cVAE + VampPrior (n=500)
# Integration works better with higher VampPrior n components

# %%
model = XXJointModel(adata=adata_training, 
                     prior='vamp', 
                     n_prior_components=500,
                     pseudoinputs_data_init=True)
model.train(max_epochs=25,
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
np.random.seed(0)
random_indices=np.random.permutation(list(range(embed_full.shape[0])))[:50000]
embed=sc.AnnData(embed_full[random_indices,:],obs=adata_training[random_indices,:].obs)
embed.obs['system_name']=embed.obs.system.map({0:'mm',1:'hs'})
embed.obs['hs_age']=embed.obs['hs_age'].astype(float)

# %%
sc.pp.neighbors(embed, use_rep='X')
sc.tl.umap(embed)

# %%
rcParams['figure.figsize']=(6,6)
sc.pl.umap(embed,color=['cell_type_eval','system_name',
                        'mm_leiden_r1.5_parsed', 'hs_disease_state'],
           s=10,wspace=0.8,ncols=3)

# %%
embed_beta=embed[embed.obs.cell_type_eval=='beta',:].copy()
sc.pp.neighbors(embed_beta, use_rep='X',n_neighbors=10)
sc.tl.umap(embed_beta, min_dist=0.1, spread=1)

# %%
embed_beta.uns['hs_disease_state_colors']=['tab:blue','tab:olive','tab:cyan','tab:brown']

# %%
rcParams['figure.figsize']=(3,3)
sc.pl.umap(embed_beta,
           color=['system_name','mm_leiden_r1.5_parsed', 'hs_disease_state',
                  'hs_age','hs_sex','hs_race'],
           s=15,wspace=0.6,ncols=3)

# %%
adata_sub_plot=adata_training[embed_beta.obs_names,:].copy()
adata_sub_plot.obsm['X_umap']=embed_beta.obsm['X_umap']

# %%
rcParams['figure.figsize']=(3,3)
sc.pl.umap(adata_sub_plot,
           # Sst, Gsg, G6pc2 (beta)
           color=['Sst','Gcg','G6pc2'],
           s=10,ncols=3,gene_symbols='gs_mm')

# %% [markdown]
# C: What separates in human T2D may just be doublets with delta cells

# %%
rcParams['figure.figsize']=(3,3)
sc.pl.umap(adata_sub_plot[adata_sub_plot.obs.system==1,],
           # Mouse/human markers
           color=['Gc','Etv1','Fkbp11','Ucn3','Mafa',
                  # DE below
                  'Rbp4','Iapp','Pcsk1','Slc7a2','Vgf',
                  # Human T2D markers
                 'Rbp1','Dlk1','Onecut2'],
           s=50,ncols=5,gene_symbols='gs_mm',sort_order=False)

# %%
rcParams['figure.figsize']=(3,3)
sc.pl.umap(adata_sub_plot[adata_sub_plot.obs.system==0,],
           # Mouse/human markers
           color=['Gc','Etv1','Fkbp11','Ucn3','Mafa',
                  # DE below
                  'Rbp4','Iapp','Pcsk1','Slc7a2','Vgf',
                  # Human T2D markers
                 'Rbp1','Dlk1','Onecut2'],
           s=30,ncols=5,gene_symbols='gs_mm',sort_order=False)

# %% [markdown]
# C: Mouse cells have localised marker expression, but not human data or at least not consistently within species (e.g. markers of same tyope coexpressed) and between species (accroding to mm). Maybe we would need to find new markers based on clustering.
#
# C:: Some genes DE in human (below) have oposite pattern in mouse - e.g. Vgf - is this artefact of normalisation or real?

# %% [markdown]
# Find clusters on all beta cells, select healthy/diabetic based on mouse, perform DE on human cells from these clusters only.

# %%
sc.tl.leiden(embed_beta,resolution=0.3)

# %%
sc.pl.umap(embed_beta,color='leiden')

# %%
# Subset to human data for DE
adata_sub_plot.obs['leiden']=embed_beta.obs['leiden']
adata_sub_plot_hs=adata_sub_plot[adata_sub_plot.obs.system==1,:].copy()


# %%
# DE on human data - up diabetes
sc.tl.rank_genes_groups(
    adata_sub_plot_hs, groupby='leiden', groups=['5'], reference='1')
sc.tl.filter_rank_genes_groups(adata_sub_plot_hs)
rcParams['figure.figsize']=(5,3)
sc.pl.rank_genes_groups(adata_sub_plot_hs,gene_symbols='gs_mm')

# %%
# DE on human data - down diabetes
sc.tl.rank_genes_groups(
    adata_sub_plot_hs, groupby='leiden', groups=['1'], reference='5')
sc.tl.filter_rank_genes_groups(adata_sub_plot_hs)
rcParams['figure.figsize']=(5,3)
sc.pl.rank_genes_groups(adata_sub_plot_hs,gene_symbols='gs_mm')

# %% [markdown]
# C: Some diabetes genes are nevertheless DE between the two clusters that correspond to mouse healthy and t2D cells.

# %% [markdown]
# ## cVAE + z_dist_cycle

# %%
model = XXJointModel(adata=adata_training,
                    z_dist_metric = 'MSE_standard')
model.train(max_epochs=25,
            check_val_every_n_epoch=1,
           plan_kwargs={'loss_weights':dict(
            kl_weight= 1.0,
            kl_cycle_weight = 0,
            reconstruction_weight= 1.0,
            reconstruction_mixup_weight = 0,
            reconstruction_cycle_weight= 0,
            z_distance_cycle_weight = 10,
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
np.random.seed(0)
random_indices=np.random.permutation(list(range(embed_full.shape[0])))[:50000]
embed=sc.AnnData(embed_full[random_indices,:],obs=adata_training[random_indices,:].obs)
embed.obs['system_name']=embed.obs.system.map({0:'mm',1:'hs'})
embed.obs['hs_age']=embed.obs['hs_age'].astype(float)

# %%
sc.pp.neighbors(embed, use_rep='X')
sc.tl.umap(embed)

# %%
rcParams['figure.figsize']=(6,6)
sc.pl.umap(embed,color=['cell_type_eval','system_name',
                        'mm_leiden_r1.5_parsed', 'hs_disease_state'],
           s=10,wspace=0.8,ncols=3)

# %%
embed_beta=embed[embed.obs.cell_type_eval=='beta',:].copy()
sc.pp.neighbors(embed_beta, use_rep='X',n_neighbors=10)
sc.tl.umap(embed_beta, min_dist=0.1, spread=1)

# %%
embed_beta.uns['hs_disease_state_colors']=['tab:blue','tab:olive','tab:cyan','tab:brown']

# %%
rcParams['figure.figsize']=(3,3)
sc.pl.umap(embed_beta,
           color=['system_name','mm_leiden_r1.5_parsed', 'hs_disease_state',
                  'hs_age','hs_sex','hs_race'],
           s=15,wspace=0.6,ncols=3)

# %%
adata_sub_plot=adata_training[embed_beta.obs_names,:].copy()
adata_sub_plot.obsm['X_umap']=embed_beta.obsm['X_umap']

# %%
rcParams['figure.figsize']=(3,3)
sc.pl.umap(adata_sub_plot,
           # Sst, Gsg, G6pc2 (beta)
           color=['Sst','Gcg','G6pc2'],
           s=10,ncols=3,gene_symbols='gs_mm')

# %% [markdown]
# C: What separates in human T2D may just be doublets with delta cells

# %%
rcParams['figure.figsize']=(3,3)
sc.pl.umap(adata_sub_plot[adata_sub_plot.obs.system==1,],
           # Mouse/human markers
           color=['Gc','Etv1','Fkbp11','Ucn3','Mafa',
                  # DE below
                  'Rbp4','Iapp','Pcsk1','Slc7a2','Vgf',
                  # Human T2D markers
                 'Rbp1','Dlk1','Onecut2'],
           s=50,ncols=5,gene_symbols='gs_mm',sort_order=False)

# %%
rcParams['figure.figsize']=(3,3)
sc.pl.umap(adata_sub_plot[adata_sub_plot.obs.system==0,],
           # Mouse/human markers
           color=['Gc','Etv1','Fkbp11','Ucn3','Mafa',
                  # DE below
                  'Rbp4','Iapp','Pcsk1','Slc7a2','Vgf',
                  # Human T2D markers
                 'Rbp1','Dlk1','Onecut2'],
           s=30,ncols=5,gene_symbols='gs_mm',sort_order=False)

# %% [markdown]
# ## cVAE

# %%
model = XXJointModel(adata=adata_training,)
model.train(max_epochs=25,
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
np.random.seed(0)
random_indices=np.random.permutation(list(range(embed_full.shape[0])))[:50000]
embed=sc.AnnData(embed_full[random_indices,:],obs=adata_training[random_indices,:].obs)
embed.obs['system_name']=embed.obs.system.map({0:'mm',1:'hs'})
embed.obs['hs_age']=embed.obs['hs_age'].astype(float)

# %%
sc.pp.neighbors(embed, use_rep='X')
sc.tl.umap(embed)

# %%
rcParams['figure.figsize']=(6,6)
sc.pl.umap(embed,color=['cell_type_eval','system_name',
                        'mm_leiden_r1.5_parsed', 'hs_disease_state'],
           s=10,wspace=0.8,ncols=3)

# %%
embed_beta=embed[embed.obs.cell_type_eval=='beta',:].copy()
sc.pp.neighbors(embed_beta, use_rep='X',n_neighbors=10)
sc.tl.umap(embed_beta, min_dist=0.1, spread=1)

# %%
embed_beta.uns['hs_disease_state_colors']=['tab:blue','tab:olive','tab:cyan','tab:brown']

# %%
rcParams['figure.figsize']=(3,3)
sc.pl.umap(embed_beta,
           color=['system_name','mm_leiden_r1.5_parsed', 'hs_disease_state',
                  'hs_age','hs_sex','hs_race'],
           s=15,wspace=0.6,ncols=3)

# %%
adata_sub_plot=adata_training[embed_beta.obs_names,:].copy()
adata_sub_plot.obsm['X_umap']=embed_beta.obsm['X_umap']

# %%
rcParams['figure.figsize']=(3,3)
sc.pl.umap(adata_sub_plot,
           # Sst, Gsg, G6pc2 (beta)
           color=['Sst','Gcg','G6pc2'],
           s=10,ncols=3,gene_symbols='gs_mm')

# %% [markdown]
# C: What separates in human T2D may just be doublets with delta cells

# %%
rcParams['figure.figsize']=(3,3)
sc.pl.umap(adata_sub_plot[adata_sub_plot.obs.system==1,],
           # Mouse/human markers
           color=['Gc','Etv1','Fkbp11','Ucn3','Mafa',
                  # DE below
                  'Rbp4','Iapp','Pcsk1','Slc7a2','Vgf',
                  # Human T2D markers
                 'Rbp1','Dlk1','Onecut2'],
           s=50,ncols=5,gene_symbols='gs_mm',sort_order=False)

# %%
rcParams['figure.figsize']=(3,3)
sc.pl.umap(adata_sub_plot[adata_sub_plot.obs.system==0,],
           # Mouse/human markers
           color=['Gc','Etv1','Fkbp11','Ucn3','Mafa',
                  # DE below
                  'Rbp4','Iapp','Pcsk1','Slc7a2','Vgf',
                  # Human T2D markers
                 'Rbp1','Dlk1','Onecut2'],
           s=30,ncols=5,gene_symbols='gs_mm',sort_order=False)

# %%
