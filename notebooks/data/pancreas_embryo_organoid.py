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
import pandas as pd
import numpy as np
from collections import defaultdict

import gc

from matplotlib import rcParams

# %%
path_organoid='/lustre/groups/ml01/workspace/karin.hrovatin/data/datasets/pancreas_organoid/'
path_embryo='/lustre/groups/ml01/workspace/karin.hrovatin/data/pigE_cross_species/datasets/hs/'
path_save='/lustre/groups/ml01/workspace/karin.hrovatin/data/cross_species_prediction/pancreas_organoid_embryo/'

# %%
adata_s5=sc.read(path_organoid+'scRNA-seq_Integrated_S5_adata_rmDoublets_normalized_integrated_annotated.h5ad',
                 backed='r')
adata_s6=sc.read(path_organoid+'scRNA-seq_iPSC_IIR-KO_S6_adata_rmDoublets_normalized_integrated_annotated.h5ad',
                backed='r')

# %%
adata_e=sc.read(path_embryo+'HFP_10x_ep_nodoub.h5ad',backed='r')

# %% [markdown]
# ### S5

# %% [markdown]
# C: TODO correct All samples are S5 (no S6!!!) and need to be renamed

# %%
adata_s5

# %%
for col in adata_s5.obs.columns:
    print('\n',col,adata_s5.obs[col].nunique())
    if adata_s5.obs[col].nunique()<30:
        print(adata_s5.obs[col].unique())
        display(adata_s5.obs[col].value_counts())

# %%
adata_s5.obs.groupby(['sample','stage'],dropna=False,observed=True).size()

# %%
pd.crosstab(adata_s5.obs['sample'],adata_s5.obs['reporter'])

# %%
pd.crosstab(adata_s5.obs['cell_type'],adata_s5.obs['sample'])

# %%
sc.pl.umap(adata_s5,color=['sample','initial_cell_type','cell_type'],wspace=0.4)

# %% [markdown]
# ### S6

# %%
adata_s6

# %%
for col in adata_s6.obs.columns:
    print('\n',col,adata_s6.obs[col].nunique())
    if adata_s6.obs[col].nunique()<30:
        print(adata_s6.obs[col].unique())
        display(adata_s6.obs[col].value_counts())

# %%
sc.pl.umap(adata_s6,color=['sample',
                           'initial_cell_type','wt_cell_type',
                           'predicted_cell_type','cell_type'],
           wspace=0.4)

# %% [markdown]
# ## Embryo

# %%
adata_e

# %%
for col in adata_e.obs.columns:
    print('\n',col,adata_e.obs[col].nunique())
    if adata_e.obs[col].nunique()<30:
        print(adata_e.obs[col].unique())
        display(adata_e.obs[col].value_counts(dropna=False))

# %% [markdown]
# annotation_nodoub_2 was used in many of their notebooks

# %%
sc.pl.embedding(adata_e,'scVI_nodoub_umap_1',
                color=['Sample','annotation_1','annotation_1_new','annotation_2',
                         'BigGroups','annotation_nodoub','annotation_nodoub_2'],
           wspace=0.4)

# %%
sc.pl.embedding(adata_e,'scVI_nodoub_umap_1',color=['Batch'],wspace=0.4)

# %%
adata_e.obs.groupby(['Batch','Sample'],observed=True).size()

# %%
pd.crosstab(adata_e.obs['annotation_nodoub_2'],adata_e.obs.Batch)

# %% [markdown]
# ## Combine objects

# %% [markdown]
# Will work with gene names as dont have EIDs for human embryo genes. Potential issue of some duplicated EIDs/scanpy appends numbers, but the model should be rebust enough to work even with  few mismatches.
#
# Also dont have all genes for organoid data - already filtered

# %%
list(adata_e6.obs_names)+list(adata_e5.obs_names)

# %%
adatas={}
for name,adata,layer,batch,cell_type in [
    ('organoid_s5',adata_s5,'raw_counts','sample','cell_type'),
    ('organoid_s6',adata_s6,'raw_counts','sample','cell_type'),
    ('embryo',adata_e,'counts','Sample','annotation_nodoub_2')]:
    print(name)
    adata=sc.AnnData(adata.layers[layer],
                     obs=adata.obs, var=adata.var)
    adata=adata[:,np.array((adata.X>0).sum(axis=0)>20).ravel()].copy()
    if name!='embryo':
        adata.obs['system']=0
    else:
        adata.obs['system']=1
    adata.obs['dataset']=name
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    adata.obs['sample']=adata.obs[batch].astype('category')
    adata.obs['cell_type']=adata.obs[cell_type].astype('category')
    adata.obs=adata.obs[['system','dataset','sample','cell_type']]
    sc.pp.highly_variable_genes(
     adata=adata, n_top_genes=3000, flavor='cell_ranger', batch_key='sample', subset=True)
    adatas[name]=adata
    print(adata.shape)

# %%
# Shared HVGs
shared_hvgs=list(set.intersection(*[set(adata.var_names) for adata in adatas.values()]))
len(shared_hvgs)

# %%
# Subset to shraed HVGs and concat
adata=sc.concat([adata[:,shared_hvgs] for adata in adatas.values()],join='outer',
                index_unique='_',
                keys=[adata.obs['dataset'][0] for adata in adatas.values()])
adata

# %%
adata.write(path_save+'organoid_embryo.h5ad')

# %% [markdown]
# ## TODO - Match ct names accross datasets

# %%
pd.crosstab(adata_s5.obs['cell_type'],adata_s5.obs['initial_cell_type'])

# %%
for stage,adata in [('s5',adata_s5),('s6',adata_s6)]:
    print('\n',stage)
    for col in [c for c in adata.obs.columns if 'cell_type' in c]:
        print('\n',col)
        print(sorted(adata.obs[col].unique()))

# %%
