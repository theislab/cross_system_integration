# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: csi
#     language: python
#     name: csi
# ---

# %%
import scanpy as sc
import pickle as pkl
import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix

from matplotlib.pyplot import rcParams
import matplotlib.pyplot as plt
import seaborn as sb

import scvi

#import torch
#torch.cuda.is_available()


# %%
TESTING=False

# %%
path_adata='/om2/user/khrovati/data/cross_system_integration/retina_adult_organoid/combined_HVG.h5ad'

# %%
group_key='cell_type'
batch_key='sample_id'
system_key='system'

# %%
adata=sc.read(path_adata)

# %%
# revert log
# Need to exp-1 only non-0 elements (others will result in 0 anyways)
adata.layers['normalized']=csr_matrix(
    (np.exp(adata.X.data)-1,adata.X.indices,adata.X.indptr),shape=adata.shape)

# %% [markdown]
# ## Counts

# %%
adata_training = adata.copy()
scvi.model.SCVI.setup_anndata(
    adata_training, 
    layer="counts", 
    batch_key=system_key,
    categorical_covariate_keys=[batch_key])

# %%
model = scvi.model.SCVI(adata_training, 
                        n_layers=2, n_hidden=256, n_latent=15, 
                        gene_likelihood="nb")
#model.to_device(0) # For some reason this does not work here automatically

max_epochs= 100 if not TESTING else 2
model.train(
    max_epochs = max_epochs,
)

# %%
cells_eval=adata_training.obs_names if not TESTING else \
    np.random.RandomState(seed=0).permutation(adata_training.obs_names)[:1000]
print('N cells for eval:',cells_eval.shape[0])
embed = model.get_latent_representation(
    adata=adata_training[cells_eval,:],
    indices=None,
    batch_size=None, )
embed=sc.AnnData(embed,obs=adata_training[cells_eval,:].obs)
# Make system categorical for metrics and plotting
embed.obs[system_key]=embed.obs[system_key].astype(str)
# neigh
sc.pp.neighbors(embed, use_rep='X')
sc.tl.umap(embed)

# %%
rcParams['figure.figsize']=(8,8)
cols=[system_key,group_key,batch_key]
np.random.seed(0)
random_idx=np.random.permutation(embed.obs_names)
fig,axs=plt.subplots(len(cols),1,figsize=(8,8*len(cols)))
for col,ax in zip(cols,axs):
    sc.pl.embedding(embed[random_idx],'X_umap',color=col,s=10,ax=ax,show=False,sort_order=False)

# %% [markdown]
# ## Normalized

# %%
adata_training = adata.copy()
scvi.model.SCVI.setup_anndata(
    adata_training, 
    layer="normalized", 
    batch_key=system_key,
    categorical_covariate_keys=[batch_key])

# %%
model = scvi.model.SCVI(adata_training, 
                        n_layers=2, n_hidden=256, n_latent=15, 
                        gene_likelihood="nb")
#model.to_device(0) # For some reason this does not work here automatically

max_epochs= 100 if not TESTING else 2
model.train(
    max_epochs = max_epochs,
)

# %%
cells_eval=adata_training.obs_names if not TESTING else \
    np.random.RandomState(seed=0).permutation(adata_training.obs_names)[:1000]
print('N cells for eval:',cells_eval.shape[0])
embed = model.get_latent_representation(
    adata=adata_training[cells_eval,:],
    indices=None,
    batch_size=None, )
embed=sc.AnnData(embed,obs=adata_training[cells_eval,:].obs)
# Make system categorical for metrics and plotting
embed.obs[system_key]=embed.obs[system_key].astype(str)
# neigh
sc.pp.neighbors(embed, use_rep='X')
sc.tl.umap(embed)

# %%
rcParams['figure.figsize']=(8,8)
cols=[system_key,group_key,batch_key]
np.random.seed(0)
random_idx=np.random.permutation(embed.obs_names)
fig,axs=plt.subplots(len(cols),1,figsize=(8,8*len(cols)))
for col,ax in zip(cols,axs):
    sc.pl.embedding(embed[random_idx],'X_umap',color=col,s=10,ax=ax,show=False,sort_order=False)

# %% [markdown]
# C: Seems like normalization does not make a difference in scIV

# %%
