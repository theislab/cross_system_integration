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
from scipy.io import mmread
from scipy.sparse import csr_matrix

import gc

from matplotlib import rcParams

# %%
path_mm='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/combined/submission/geo/'
path_genes='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/gene_lists/'
path_hs='/lustre/groups/ml01/workspace/karin.hrovatin/data/datasets/d10_1101_2023_02_03_526994/'
path_save='/lustre/groups/ml01/workspace/karin.hrovatin/data/cross_species_prediction/pancreas_example/'

# %%
# Orthologues
orthology_info=pd.read_table(path_genes+'orthologues_ORGmus_musculus_ORG2homo_sapiens_V103.tsv'
                         ).rename(
    {'Gene name':'gs_mm','Human gene name':'gs_hs',
     'Gene stable ID':'eid_mm','Human gene stable ID':'eid_hs'},axis=1)

# %%
# One to one orthologues - dont have same mm/hs gene in the table 2x
oto_orthologues=orthology_info[~orthology_info.duplicated('eid_mm',keep=False).values & 
               ~orthology_info.duplicated('eid_hs',keep=False).values]

# %%
oto_orthologues.shape

# %% [markdown]
# ### Mouse

# %%
adata_mm=sc.read(path_mm+'adata.h5ad')

# %%
adata_mm=adata_mm[adata_mm.obs.study!='embryo',:]

# %%
# Add raw expression to X, remove lowly expr genes, and normalise
adata_mm_raw=adata_mm.raw.to_adata()
adata_mm=adata_mm_raw.copy()
adata_mm=adata_mm[:,np.array((adata_mm.X>0).sum(axis=0)>20).ravel()]
adata_mm=adata_mm[:,[g for g in oto_orthologues.eid_mm if g in adata_mm.var_names]]
sc.pp.normalize_total(adata_mm)
sc.pp.log1p(adata_mm)

# %%
# HVGs on the final cell subset
sc.pp.highly_variable_genes(
     adata=adata_mm, n_top_genes=3000, flavor='cell_ranger', batch_key='study_sample',
    subset=True)
adata_mm.shape

# %%
adata_mm

# %% [markdown]
# ### Human

# %%
adata_hs=sc.read(path_hs+'hpap_islet_scRNAseq.h5ad')

# %%
counts=mmread(path_hs+'hpap_islet_scRNAseq_counts.mtx')
adata_hs_raw=adata_hs.raw.to_adata()
adata_hs_raw.X=csr_matrix(counts.T)

# %%
adata_hs

# %%
# Add raw expression to X, remove lowly expr genes, and normalise
adata_hs=adata_hs_raw.copy()
adata_hs=adata_hs[:,np.array((adata_hs.X>0).sum(axis=0)>20).ravel()]
gs=set(adata_hs.var_names)
adata_hs=adata_hs[:,[g for g in set(oto_orthologues.gs_hs.values) if g in gs]]
sc.pp.normalize_total(adata_hs)
sc.pp.log1p(adata_hs)

# %%
# HVGs on the final cell subset
sc.pp.highly_variable_genes(
     adata=adata_hs, n_top_genes=3000, flavor='cell_ranger', batch_key='Library',
    subset=True)
adata_hs.shape

# %% [markdown]
# ### Shared genes

# %%
# Find shared HVGs
gs_hs=set(adata_hs.var_names)
eids_mm=set(adata_mm.var_names)
shared_orthologues=oto_orthologues.query('gs_hs in @gs_hs')
shared_orthologues=shared_orthologues.query('eid_mm in @eids_mm')
print('N shared:',shared_orthologues.shape[0])

# %%
# Subset adatas to shared HVGs
# This already ensures same gene order
adata_hs=adata_hs[:,shared_orthologues.gs_hs]
adata_mm=adata_mm[:,shared_orthologues.eid_mm]


# %% [markdown]
# ### Combine adatas of mm and hs

# %%
pd.crosstab(adata_hs.obs['Cell Type'],adata_hs.obs['Cell Type Grouped'])

# %%
sorted(adata_hs.obs['Cell Type'].str.lower().unique())

# %%
sorted(adata_mm.obs.cell_type_integrated_v1_parsed.unique())

# %%
adata_mm.obs['cell_type_eval']=adata_mm.obs.cell_type_integrated_v1_parsed.replace({
     'E endo.':np.nan,
     'E non-endo.':np.nan,
     'lowQ':np.nan,
})
adata_hs.obs['cell_type_eval']=adata_hs.obs['Cell Type'].str.lower().replace({
 'active stellate':'stellate a.',
 'cycling alpha':'endo. prolif.',
 'gamma+epsilon':'gamma',
 'macrophage':'immune',
 'mast':'immune',
 'muc5b+ ductal':'ductal',
 'quiescent stellate':'stellate q.'
})

# %%
list(adata_mm.obs['cell_type_eval'].unique())

# %%
list(adata_hs.obs['cell_type_eval'].unique())

# %%
# Prepare adatas for concat and concat
# Human
obs_keep_hs=['Library', 'Sex', 'Diabetes Status','Cell Type','cell_type_eval']
adata_hs_sub=adata_hs.copy()
adata_hs_sub.obs=adata_hs_sub.obs[obs_keep_hs]
adata_hs_sub.obs.rename({'Library':'batch'}, axis=1, inplace=True)
adata_hs_sub.obs.rename({c:'hs_'+c for c in adata_hs_sub.obs.columns 
                         if c not in ['cell_type_eval','batch']}, 
                         axis=1, inplace=True)
adata_hs_sub.obs['system']=1
adata_hs_sub.var['gene_symbol']=adata_hs_sub.var_names
adata_hs_sub.var_names=adata_mm.var_names # Can do as matched subsetting
del adata_hs_sub.obsm
del adata_hs_sub.uns
# Mouse
adata_mm_sub=adata_mm.copy()
obs_keep_mm=['study_sample', 'study', 'sex','age','study_sample_design', 
              'cell_type_integrated_v1_parsed','BETA-DATA_hc_gene_programs_parsed',
              'BETA-DATA_leiden_r1.5_parsed', 'cell_type_eval']
adata_mm_sub.obs=adata_mm_sub.obs[obs_keep_mm]
adata_mm_sub.obs.rename({'study_sample':'batch'}, axis=1, inplace=True)
adata_mm_sub.obs.rename({c:'mm_'+c.replace('BETA-DATA_','') for c in adata_mm_sub.obs.columns 
                         if c not in ['cell_type_eval','batch']}, 
                         axis=1, inplace=True)
adata_mm_sub.obs['system']=0
del adata_mm_sub.obsm
del adata_hs_sub.uns
# Concat
adata=sc.concat([adata_mm_sub,adata_hs_sub],join='outer')

#del adata_mm_sub
#del adata_hs_sub
gc.collect()

# %%
gs_df=shared_orthologues.copy()
gs_df.index=shared_orthologues['eid_mm']
adata.var[['gs_mm','gs_hs']]=gs_df[['gs_mm','gs_hs']]

# %% [markdown]
# ### Add raw counts

# %%
# Subset raw
adata_mm_raw=adata_mm_raw[adata[adata.obs.system==0].obs_names,adata.var_names]
adata_hs_raw=adata_hs_raw[adata[adata.obs.system==1].obs_names,adata.var.gs_hs]
adata_hs_raw.var_names=adata.var_names
# make single raw
adata_raw=sc.concat([adata_mm_raw,adata_hs_raw])

# %%
# Add counts
adata.layers['counts']=adata_raw[adata.obs_names,adata.var_names].X

# %% [markdown]
# ### Save

# %%
display(adata)

# %%
adata.write(path_save+'combined_conditions_orthologues_full_hpap2nd.h5ad')

# %%
