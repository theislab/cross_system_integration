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

# %%
path='/net/bmc-lab6/data/lab/kellis/users/khrovati/data/'
path_data=path+'datasets/10_1016_j_cell_2020_08_013/'
path_train=path+'cross_system_integration/retina_adult_organoid/'

# %%
adata=sc.concat([
    sc.concat(
        [sc.read(path_data+'periphery.h5ad'),sc.read(path_data+'fovea.h5ad')],
        label='region',keys =['periphery','fovea'],index_unique='-',join='outer'
    ),sc.read(path_data+'organoid.h5ad')],
    label='material',keys =['adult','organoid'],index_unique='-',join='outer'
)

# %%
adata

# %% [markdown]
# C: Unclear what batch may be

# %%
adata.obs.filtered_out_cells.sum()

# %%
pd.crosstab(adata.obs.cell_type,adata.obs.material)

# %%
adata.obs.cell_type.isna().sum()

# %%
sc.pl.umap(adata,color=['cell_type','material','region','sample_id'])

# %% [markdown]
# ## Prepare data for training

# %%
adata_sub=adata.copy()

# %%
# Remove unnanotated cells
adata_sub=adata_sub[adata_sub.obs.cell_type!='native cell',:]

# %%
# Keep not too lowly expressed genes as intersection of the two systems
adata_sub=adata_sub[:,
                    np.array((adata_sub[adata_sub.obs.material=="adult",:].X>0).sum(axis=0)>20).ravel()&\
                    np.array((adata_sub[adata_sub.obs.material=="organoid",:].X>0).sum(axis=0)>20).ravel()
                   ]

# %%
adata_sub.shape

# %%
# Normalize and log scale
# Can normalize together as just CPM
sc.pp.normalize_total(adata_sub, target_sum=1e4)
sc.pp.log1p(adata_sub)

# %%
hvgs=set(sc.pp.highly_variable_genes(
    adata_sub[adata_sub.obs.material=="adult",:], 
    n_top_genes=4000, flavor='cell_ranger', inplace=False, batch_key='sample_id').query('highly_variable==True').index)&\
set(sc.pp.highly_variable_genes(
    adata_sub[adata_sub.obs.material=="organoid",:], 
    n_top_genes=4000, flavor='cell_ranger', inplace=False, batch_key='sample_id').query('highly_variable==True').index)
print(len(hvgs))

# %%
adata_sub=adata_sub[:,list(hvgs)]

# %%
del adata_sub.uns
del adata_sub.obsm
adata_sub.obs=adata_sub.obs[[
    'cell_type','cell_type_group','author_cell_type','condition', 
    'dataset','sample_id','ega_sample_alias', 'hca_data_portal_donor_uuid', 'hca_data_portal_cellsuspension_uuid', 
    'region',  'material', 'system']]

# %%
adata_sub.obs['system']=adata_sub.obs['material'].map({"organoid":0,'adult':1})

# %%
adata_sub.layers['counts']=adata[adata_sub.obs_names,adata_sub.var_names].X.copy()

# %%
adata_sub

# %%
adata_sub.write(path_train+'combined_HVG.h5ad')

# %%
