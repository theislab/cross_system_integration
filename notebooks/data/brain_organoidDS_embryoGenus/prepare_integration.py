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
#     display_name: analysis
#     language: python
#     name: analysis
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
path_data='/net/bmc-lab6/data/lab/kellis/users/khrovati/data/'
path_mm=path_data+'datasets/mouse_brain_devel_genus/'
path_hs=path_data+'datasets/human_brain_organoid_DS/'
path_genes=path_data+'gene_info/'
path_save=path_data+'cross_species_prediction/brain_mmEmbryo_hsOrganoid/'

# %%
# Orthologues
orthology_info=pd.read_table(path_genes+'orthologues_ORGmus_musculus_ORG2homo_sapiens_V109.tsv'
                         ).rename(
    {'Gene name':'gs_mm','Human gene name':'gs_hs',
     'Gene stable ID':'eid_mm','Human gene stable ID':'eid_hs'},axis=1)

# %%
# One to one orthologues - dont have same mm/hs gene in the table 2x
# USe here human gs as dont have EIDs
oto_orthologues=orthology_info[~orthology_info.duplicated('eid_mm',keep=False).values & 
               ~orthology_info.duplicated('gs_hs',keep=False).values]

# %%
oto_orthologues

# %%
oto_orthologues['gs_mm'].nunique()==oto_orthologues.shape[0]

# %%
oto_orthologues['gs_mm'].value_counts().head(3)

# %%
oto_orthologues['gs_hs'].nunique()==oto_orthologues.shape[0]

# %% [markdown]
# C: Mouse symbols are not unique but for now it may be resolved after HVG selection.

# %% [markdown]
# ## Mouse

# %%
adata=sc.read(path_mm+'filtered_celltype_subset_labelled.h5ad')

# %%
adata.shape

# %%
# normalise
adata.layers['counts']=adata.X.copy()
sc.pp.normalize_total(adata,target_sum =1e4)
sc.pp.log1p(adata)

# %%
# Subset to orthologues
oto=set(oto_orthologues['gs_mm'])
adata=adata[:,[v for v in adata.var_names if v in oto]]
adata.shape

# %%
n_hvg=4000
sc.pp.highly_variable_genes(
     adata=adata, n_top_genes=n_hvg, flavor='cell_ranger', subset=True,batch_key='ID')

# %%
# Parse obs
del adata.uns
del adata.obsm
del adata.var
obs=adata.obs
del adata.obs
adata.obs['cell_type']=obs['celltype']
adata.obs['condition']=obs['Genotype_Treatment']
adata.obs['sample']=obs['ID']
adata.obs['system']=0

# %%
adata

# %%
adata_mm=adata

# %% [markdown]
# ## Human

# %%
adata=sc.read(path_hs+'all_combined_cycling.h5ad')

# %%
adata.shape

# %%
# normalise
adata.layers['counts']=adata.X.copy()
sc.pp.normalize_total(adata,target_sum =1e4)
sc.pp.log1p(adata)

# %%
# Subset to orthologues and change to mouse symbols
oto=set(oto_orthologues['gs_hs'])
adata=adata[:,[v for v in adata.var_names if v in oto]]
oto_orthologues.index=oto_orthologues.gs_hs
adata.var_names=oto_orthologues.loc[adata.var_names,'gs_mm']
adata.shape

# %%
n_hvg=4000
sc.pp.highly_variable_genes(
     adata=adata, n_top_genes=n_hvg, flavor='cell_ranger', subset=True,batch_key='Sample')

# %%
# Parse obs
del adata.uns
del adata.obsm
del adata.var
obs=adata.obs
del adata.obs
adata.obs['cell_type']=obs['LineComp']
adata.obs['cell_type_fine']=obs['FullLineage']
adata.obs['condition']=obs.apply(lambda x: str(x['Genotype'])+'_'+str(x['Time']),axis=1)
adata.obs['sample']=obs['Sample']
adata.obs['system']=1

# %%
adata

# %%
adata_hs=adata

# %% [markdown]
# ## Combine adatas

# %%
genes=list(set(adata_mm.var_names)&set(adata_hs.var_names))
adata=sc.concat([adata_mm[:,genes],adata_hs[:,genes]],join='outer')

# %%
adata

# %%
adata.write(path_save+'combined_orthologuesHVG.h5ad')
