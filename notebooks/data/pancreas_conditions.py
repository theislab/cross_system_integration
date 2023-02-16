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
path_mm='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/combined/submission/geo/'
path_genes='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/gene_lists/'
path_hs='/lustre/groups/ml01/workspace/karin.hrovatin/data/datasets/d10_1101_2023_01_03_522578/'
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
adata_mm=adata_mm.raw.to_adata()
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
adata_hs=sc.read(path_hs+'T1D_T2D_public.h5ad')

# %%
adata_hs

# %%
# Add raw expression to X, remove lowly expr genes, and normalise
adata_hs=adata_hs.raw.to_adata()
adata_hs=adata_hs[:,np.array((adata_hs.X>0).sum(axis=0)>20).ravel()]
adata_hs.var_names=adata_hs.var['features']
gs=set(adata_hs.var_names)
adata_hs=adata_hs[:,[g for g in set(oto_orthologues.gs_hs.values) if g in gs]]
sc.pp.normalize_total(adata_hs)
sc.pp.log1p(adata_hs)

# %%
# HVGs on the final cell subset
sc.pp.highly_variable_genes(
     adata=adata_hs, n_top_genes=3000, flavor='cell_ranger', batch_key='donor_id',
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
sorted(adata_hs.obs.cell_type.unique())

# %%
sorted(adata_mm.obs.cell_type_integrated_v1_parsed.unique())

# %%
adata_mm.obs['cell_type_eval']=adata_mm.obs.cell_type_integrated_v1_parsed.replace({
     'E endo.':np.nan,
     'E non-endo.':np.nan,
     'lowQ':np.nan,
})
adata_hs.obs['cell_type_eval']=adata_hs.obs.cell_type.replace({
     'PP cell':'gamma',
     'acinar cell':'acinar',
     'alpha cell':'alpha',
     'beta cell':'beta',
     'delta cell':'delta',
     'duct epithelial cell':'ductal',
     'endothelial cell': 'endothelial',
     'epsilon cell':'epsilon',
     'leukocyte':'immune',
     'mesenchymal cell':'mesenchymal',
     'unknown':np.nan
})

# %%
list(adata_mm.obs['cell_type_eval'].unique())

# %%
list(adata_hs.obs['cell_type_eval'].unique())

# %%
# Prepare adatas for concat and concat
# Human
obs_keep_hs=['donor_id', 'cell_type', 'sex', 'age', 'race', 'disease_state',
             'cell_type_eval']
adata_hs_sub=adata_hs.copy()
adata_hs_sub.obs=adata_hs_sub.obs[obs_keep_hs]
adata_hs_sub.obs.rename({'donor_id':'batch'}, axis=1, inplace=True)
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
adata_mm_raw=sc.read(path_mm+'adata.h5ad').raw.to_adata()
adata_hs_raw=sc.read(path_hs+'T1D_T2D_public.h5ad').raw.to_adata()
adata_hs_raw.var_names=adata_hs_raw.var['features'].values

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
adata.write(path_save+'combined_conditions_orthologues_full.h5ad')

# %%
#adata=sc.read(path_save+'combined_conditions_orthologues_full.h5ad')

# %%
