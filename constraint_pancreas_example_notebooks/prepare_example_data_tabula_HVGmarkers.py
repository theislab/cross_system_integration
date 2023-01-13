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
# Prepare mouse and human data from tabula muris and sapiens for training of the model. Use HVGs and endocrine markers to enable their later analysis.

# %%
import scanpy as sc
import pandas as pd
import pickle as pkl
import gc
import numpy as np

# %%
path_ds='/lustre/groups/ml01/workspace/karin.hrovatin/data/datasets/'
path_hs=path_ds+'tabula_sapiens/'
path_mm=path_ds+'tabula_muris_senis/'
path_genes='/lustre/groups/ml01/workspace/karin.hrovatin/data/cross_species_prediction/pancreas_ct_example_constraint/'


# %% [markdown]
# ### Prepare mouse data

# %%
markers_mm=list(set(
    [g for gs in pkl.load(open(path_genes+'endo_markers_set_mm.pkl','rb')).values() 
          for g in gs]))

# %%
adata_mm=sc.read(path_mm+'local.h5ad')

# %%
sc.pp.highly_variable_genes(adata_mm, n_top_genes=2000,batch_key='mouse.id')

# %% [markdown]
# Merge HVG and markers

# %%
markers_mm=set([m for m in markers_mm if m in adata_mm.var_names])
print('N selected markers:',len(markers_mm))

# %%
hvgs_mm=set(adata_mm.var_names[adata_mm.var['highly_variable']])
print('N selected HVGs:',len(hvgs_mm))

# %%
genes_mm=list(markers_mm|hvgs_mm)
print('N selected genes:',len(genes_mm))

# %%
# Subset to selected genes
adata_mm=adata_mm[:,genes_mm].copy()
gc.collect()

# %%
adata_mm

# %%
adata_mm.obs['tissue_cell_type']=adata_mm.obs.apply(
    lambda x: x['tissue']+'_'+x['cell_type'], axis=1)

# %% [markdown]
# ### Prepare human data

# %%
markers_hs=list(set(
    [g for gs in pkl.load(open(path_genes+'endo_markers_set_hs.pkl','rb')).values() 
          for g in gs]))

# %%
adata_hs=sc.read(path_hs+'local.h5ad')

# %%
sc.pp.highly_variable_genes(adata_hs, n_top_genes=2000,batch_key='donor_id')

# %% [markdown]
# Merge HVG and markers

# %%
markers_hs=set(adata_hs.var.query('feature_name in @markers_hs').index)
print('N selected markers:',len(markers_hs))

# %%
hvgs_hs=set(adata_hs.var_names[adata_hs.var['highly_variable']])
print('N selected HVGs:',len(hvgs_hs))

# %%
genes_hs=list(markers_hs|hvgs_hs)
print('N selected genes:',len(genes_hs))

# %%
# Subset to selected genes
adata_hs=adata_hs[:,genes_hs].copy()
gc.collect()

# %%
adata_hs

# %%
adata_hs.obs['tissue_cell_type']=adata_hs.obs.apply(
    lambda x: x['tissue']+'_'+x['cell_type'], axis=1)

# %% [markdown]
# ## Combine data

# %%
cts_mm=set(adata_mm.obs.tissue_cell_type.unique())
cts_hs=set(adata_hs.obs.tissue_cell_type.unique())
print('N cell groups both:',len(cts_mm&cts_hs),
      'N cell groups mm unique:',len(cts_mm-cts_hs),
      'N cell groups hs unique:',len(cts_hs-cts_mm),
     )

# %%
cts_mm=set(adata_mm.obs.cell_type.unique())
cts_hs=set(adata_hs.obs.cell_type.unique())
print('N cell groups both:',len(cts_mm&cts_hs),
      'N cell groups mm unique:',len(cts_mm-cts_hs),
      'N cell groups hs unique:',len(cts_hs-cts_mm),
     )

# %% [markdown]
# Problem that categories can not be easily mateched by names alone. Maybe graph parsing would help.

# %%
