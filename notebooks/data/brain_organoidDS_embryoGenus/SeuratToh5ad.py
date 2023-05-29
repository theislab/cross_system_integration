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
from scipy.sparse import csr_matrix


# %%
#R interface
# %load_ext rpy2.ipython

import rpy2.robjects as ro

from rpy2.robjects import pandas2ri
pandas2ri.activate()


# %% language="R"
# library('Seurat')
# library(Matrix)
# library(qs)

# %%
dir_data='/net/bmc-lab6/data/lab/kellis/users/khrovati/data/datasets/'

# %% [markdown]
# ## human_brain_organoid_DS

# %%
dir_ds=dir_data+'human_brain_organoid_DS/'
fn='all_combined_cycling.qs'

# %% magic_args="-i dir_ds -i fn -o obs -o var -o shape -o count_data_i -o count_data_j -o count_data_x" language="R"
# obj<-qread(paste0(dir_ds,fn))
# shape<-dim(obj)
# counts<-GetAssayData(object = obj, slot = "counts")
# counts<-as(counts, "TsparseMatrix")
# # Exporting like vectors is fastest (not exporting like df, etc)
# # Matrix starts indexing at 0
# count_data_i<-counts@i
# count_data_j<-counts@j
# count_data_x<-counts@x
# obs<-obj@meta.data
# var<-obj[['RNA']][[]]

# %%
# %R dim(obj)

# %%
counts=csr_matrix((count_data_x,(count_data_j,count_data_i)), shape=shape[::-1])

# %%
adata=sc.AnnData(counts,obs=obs,var=var)

# %%
adata

# %%
for col in adata.obs.columns:
    print('****\n',col)
    if adata.obs[col].nunique()<30:
        print(sorted(adata.obs[col].unique()))

# %%
pd.crosstab(adata.obs.FullLineage,adata.obs.LineComp)

# %%
pd.crosstab(adata.obs.LineComp,adata.obs.group_comb)

# %%
adata.write(dir_ds+'.'.join(fn.split('.')[:-1])+'.h5ad')

# %% [markdown]
# ## mouse_brain_devel_genus

# %%
dir_ds=dir_data+'mouse_brain_devel_genus/'
fn='filtered_celltype_subset_labelled.rds'

# %% magic_args="-i dir_ds -i fn -o obs -o var -o shape -o count_data_i -o count_data_j -o count_data_x" language="R"
# obj<-readRDS(paste0(dir_ds,fn))
# shape<-dim(obj)
# counts<-GetAssayData(object = obj, slot = "counts")
# counts<-as(counts, "TsparseMatrix")
# # Exporting like vectors is fastest (not exporting like df, etc)
# # Matrix starts indexing at 0
# count_data_i<-counts@i
# count_data_j<-counts@j
# count_data_x<-counts@x
# obs<-obj@meta.data
# var<-obj[['RNA']][[]]

# %%
# %R dim(obj)

# %%
counts=csr_matrix((count_data_x,(count_data_j,count_data_i)), shape=shape[::-1])

# %%
adata=sc.AnnData(counts,obs=obs,var=var)

# %%
adata

# %%
for col in adata.obs.columns:
    print('****\n',col)
    if adata.obs[col].nunique()<30:
        print(sorted(adata.obs[col].unique()))

# %%
adata.write(dir_ds+'.'.join(fn.split('.')[:-1])+'.h5ad')

# %%
