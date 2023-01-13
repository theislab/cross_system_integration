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
from scipy.io import mmread

import gc

# %%
path_data='/lustre/groups/ml01/workspace/karin.hrovatin/data/datasets/'

# %% [markdown]
# ## Lee

# %%
path_data_lee=path_data+'d10_1016_j_neuron_2020_06_021/'

# %%
prefixes=['GSE152058_human_snRNA_processed',
               'GSE152058_R62_snRNA_processed',
               'GSE152058_zQ175_snRNA_processed']

# %%
for prefix in prefixes:
    prefix=prefix+'_'
    print(prefix)
    x=mmread(path_data_lee+prefix+'counts.mtx.gz').tocsr()
    var=pd.read_table(path_data_lee+prefix+'rowdata.tsv.gz', compression='gzip',index_col=0)
    obs=pd.read_table(path_data_lee+prefix+'coldata.tsv.gz', compression='gzip',index_col=0)
    adata=sc.AnnData(x.astype(np.float32).T,obs=obs,var=var)
    display(adata)
    adata.write(path_data+prefix.rstrip('_')+'.h5ad')

# %%
for prefix in prefixes:
    print('\n****',prefix)
    obs=sc.read(path_data+prefix+'.h5ad',backed='r').obs
    for col in obs.columns:
        print(col)
        if obs[col].nunique()<30:
            print(sorted(obs[col].unique()))
gc.collect()

# %% [markdown]
# ## Sun

# %%
# TODO read object - they are rds but all sparse matrices so can read with rpy2 easily
