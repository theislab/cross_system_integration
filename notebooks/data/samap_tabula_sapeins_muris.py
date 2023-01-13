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
#     display_name: SAMap
#     language: python
#     name: samap
# ---

# %% [markdown]
# # SAMap mapping of tabula sapiens/muris 
# From: Deep learning of cross-species single-cell landscapes identifies conserved regulatory programs underlying cell types

# %%
import pickle as pkl
import scanpy as sc
import numpy as np
import pandas as pd

from matplotlib import rcParams
import matplotlib.pyplot as plt

# %%
path='/lustre/groups/ml01/workspace/karin.hrovatin/data/cross_species_prediction/samap_tabula/'

# %% [markdown]
# ## Extract integrated adata from samap object

# %%
samap_res=pkl.load(open(path+'H_M_1101_new.pkl','rb'))

# %%
adata=samap_res.smap.samap.adata

# %%
adata.obs

# %%
# Embedding
adata.obs['cell_type']=adata.obs.Cellcluster.str.replace('hu_','').str.replace('mo_','')
np.random.seed(0)
random_indices=np.random.permutation(list(range(adata.shape[0])))
with plt.rc_context({'figure.figsize':(10,10)}):
    sc.pl.umap(adata[random_indices,:],color=['species','cell_type'])
adata.obs.drop('cell_type',axis=1,inplace=True)

# %%
# Save adata with mapping information
if False:
    adata.write(path+'H_M_1101_new.h5ad')

# %% [markdown]
# ## Mutual NN

# %%
# Reload adata
adata=sc.read(path+'H_M_1101_new.h5ad')

# %%
# Mutual NN matrix
mnn=adata.uns['mdata']['knn_2v1'].multiply(adata.uns['mdata']['knn_1v2'].T)
# Mnn numbers
print(f'Has MNN in hs: {((mnn>0).sum(axis=0)>0).sum()}/{mnn.shape[1]} and '+\
      f'in mm: {((mnn>0).sum(axis=1)>0).sum()}/{mnn.shape[0]}')
# Save mnn
adata.uns['mdata']['mnn']=mnn

# %%
# Save adata with mnn information
if False:
    adata.write(path+'H_M_1101_new.h5ad')

# %% [markdown]
# Are mnn cells all ower the embedding?

# %%
# Which cells on embedding have a mnn? 
adata.obs.loc[
    adata.obs[adata.obs.species=='hu'].index[np.array((mnn>0).sum(axis=0)>0).ravel()],
    'has_mnn']='True'
adata.obs.loc[
    adata.obs[adata.obs.species=='mo'].index[np.array((mnn>0).sum(axis=1)>0).ravel()],
    'has_mnn']='True'
adata.obs['has_mnn'].fillna('False',inplace=True)
np.random.seed(0)
random_indices=np.random.permutation(list(range(adata.shape[0])))
with plt.rc_context({'figure.figsize':(10,10)}):
    sc.pl.umap(adata[random_indices,:],color=['has_mnn'],groups=['True'],s=10)
adata.obs.drop('has_mnn',axis=1,inplace=True)

# %% [markdown]
# C: Most clusters have MNN cells, but some not. Some of the uncovered ones are species specific (see above).

# %% [markdown]
# ### Beta cells
# Try to find beta cells for analysis

# %%
with plt.rc_context({'figure.figsize':(10,10)}):
    sc.pl.umap(adata[random_indices,:],color=['hu_INS','mo_Ins1'],s=20,cmap='viridis_r')

# %% [markdown]
# Ins distn accross cells

# %%
plt.hist(adata[adata.obs.query('species=="hu"').index,'hu_INS'].to_df(),bins=50)
plt.yscale('log')

# %%
plt.hist(adata[adata.obs.query('species=="mo"').index,'mo_Ins1'].to_df(),bins=50)
plt.yscale('log')

# %% [markdown]
# C: Some cells have no ins (non-pancreatic tissue) while others have some ins or may be ambient. The ambient thr seems quite clear. 
#
# Probably best to go with less strict thr to remove all cells from training and more strict thr to retain only true beta cells for eval. Can also use per-species cl info (have leiden clusters which were computed per species as dont match on umap (not shown)).

# %%
thr=2
adata.obs['ins_high']=(adata[:,'mo_Ins1'].to_df()>thr).values.ravel() | \
    (adata[:,'hu_INS'].to_df()>thr).values.ravel()
ins_top_cls=pd.crosstab(adata.obs.leiden_clusters,adata.obs.ins_high,normalize='index'
           ).sort_values(True,ascending=False).head(10)
display(ins_top_cls)
adata.obs.drop('ins_high',axis=1,inplace=True)

# %%
# Location of clusters with high ins+ ratio on integrated umap
np.random.seed(0)
random_indices=np.random.permutation(list(range(adata.shape[0])))
with plt.rc_context({'figure.figsize':(10,10)}):
    colors=dict(zip(adata.obs.leiden_clusters.unique(),
                    ['gray']*adata.obs.leiden_clusters.nunique()))
    colors.update(dict(zip(ins_top_cls.index.values[:4],['r']*4)))
    for group in ins_top_cls.index.values[:4]:
        sc.pl.umap(adata[random_indices,:],
               color=['leiden_clusters'],groups=group,s=40, palette=colors)

# %% [markdown]
# C: Interetsingly, cells within clusters got mapped to different locations. Thus using thr>x is probably safest for removing all ins+ cells.

# %%

# %%

# %%
