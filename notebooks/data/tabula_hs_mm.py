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
# # TODO!!! ERROR: oto_orthologues are computed wrongly, need to use keep=False in duplicated

# %%
import scanpy as sc
import pandas as pd
import pickle as pkl
import gc
import numpy as np
from scipy.sparse import csr_matrix

# %%
path_ds='/lustre/groups/ml01/workspace/karin.hrovatin/data/datasets/'
path_hs=path_ds+'tabula_sapiens/'
path_mm=path_ds+'tabula_muris_senis/'
path_genes='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/gene_lists/'
path_save='/lustre/groups/ml01/workspace/karin.hrovatin/data/cross_species_prediction/tabula/'

# %%
# orthologues
orthologues=pd.read_table(path_genes+'orthologues_ORGmus_musculus_ORG2homo_sapiens_V103.tsv'
                         ).rename(
    {'Gene name':'gs_mm','Human gene name':'gs_hs',
     'Gene stable ID':'eid_mm','Human gene stable ID':'eid_hs'},axis=1)
# One to one orthologues - dont have same mm/hs gene in the table 2x
oto_orthologues=orthologues[~orthologues.duplicated('eid_mm').values & 
               ~orthologues.duplicated('eid_hs').values]
print(oto_orthologues.shape[0])

# %% [markdown]
# ## Prepare data with shared HVGs based on one-to-one orthologues

# %% [markdown]
# ### Tabula muris

# %%
adata_mm=sc.read(path_mm+'local.h5ad')

# %%
n=100
cells=[]
for group,data in adata_mm.obs.groupby(['mouse.id','tissue','cell_type']):
    cells.extend(data.index[:n])
print(len(cells))
adata_mm=adata_mm[cells,:]

# %%
# Add raw expression to X, remove lowly expr genes, subset to O-to-O orthologues, and normalise
adata_mm=adata_mm.raw.to_adata()
adata_mm=adata_mm[:,np.array((adata_mm.X>0).sum(axis=0)>20).ravel()]
adata_mm=adata_mm[:,[g for g in oto_orthologues.eid_mm if g in adata_mm.var_names]]
sc.pp.normalize_total(adata_mm)
sc.pp.log1p(adata_mm)
adata_mm.shape

# %%
# HVGs on the final cell subset
sc.pp.highly_variable_genes(
     adata=adata_mm, n_top_genes=5000, flavor='cell_ranger', batch_key='mouse.id',
    subset=True)
adata_mm.shape

# %%
gc.collect()

# %%
adata_mm.obs['batch']=adata_mm.obs['mouse.id']

# %% [markdown]
# ### Tabula sapiens

# %%
adata_hs=sc.read(path_hs+'local.h5ad')

# %%
n=100
cells=[]
for group,data in adata_hs.obs.groupby(['donor_id','tissue','cell_type']):
    cells.extend(data.index[:n])
print(len(cells))
adata_hs=adata_hs[cells,:]

# %%
# Add raw expression to X, remove lowly expr genes, subset to O-to-O orthologues, and normalise
adata_hs=adata_hs.raw.to_adata()
adata_hs=adata_hs[:,np.array((adata_hs.X>0).sum(axis=0)>20).ravel()]
adata_hs=adata_hs[:,[g for g in oto_orthologues.eid_hs if g in adata_hs.var_names]]
sc.pp.normalize_total(adata_hs)
sc.pp.log1p(adata_hs)
adata_hs.shape

# %%
# HVGs on the final cell subset
sc.pp.highly_variable_genes(
     adata=adata_hs, n_top_genes=5000, flavor='cell_ranger', batch_key='donor_id',
    subset=True)
adata_hs.shape

# %%
gc.collect()

# %%
adata_hs.obs['batch']=adata_hs.obs['donor_id']

# %% [markdown]
# ### Shared HVGs

# %%
# Find shared HVGs
eids_hs=set(adata_hs.var_names)
eids_mm=set(adata_mm.var_names)
shared_orthologues=oto_orthologues.query('eid_hs in @eids_hs')
shared_orthologues=shared_orthologues.query('eid_mm in @eids_mm')
print('N shared:',shared_orthologues.shape[0])

# %%
# Subset adatas to shared HVGs
# This already ensures same gene order
adata_hs=adata_hs[:,shared_orthologues.eid_hs]
adata_mm=adata_mm[:,shared_orthologues.eid_mm]


# %% [markdown]
# ### Combine adatas

# %%
# Prepare adatas for concat and concat
cols=['cell_type','tissue','batch']
# Human
adata_hs_sub=adata_hs.copy()
adata_hs_sub.obs=adata_hs_sub.obs[cols]
adata_hs_sub.obs['system']=1
adata_hs_sub.var['EID']=adata_hs_sub.var_names
adata_hs_sub.var_names=adata_mm.var_names
del adata_hs_sub.obsm
# Mouse
adata_mm_sub=adata_mm.copy()
adata_mm_sub.obs=adata_mm_sub.obs[cols]
adata_mm_sub.obs['system']=0
del adata_mm_sub.obsm
# Concat
adata=sc.concat([adata_mm_sub,adata_hs_sub])

del adata_mm_sub
del adata_hs_sub
gc.collect()

display(adata)

# %%
adata.write(path_save+'combined_orthologues.h5ad')

# %%
