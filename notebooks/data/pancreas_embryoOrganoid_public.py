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
import glob
from scipy.io import mmread
from scipy.sparse import csr_matrix

import gc

from matplotlib import rcParams

# %%
path_data='/lustre/groups/ml01/workspace/karin.hrovatin/data/'
path_e=''
path_o=path_data+'datasets/d10_1038_s41587-022-01219-z/'
path_save=path_data+'cross_species_prediction/pancreas_organoid_embryo/'

# %% [markdown]
# ## Organoid - pp to adata

# %%
# Load genes - all same
features=pd.read_table(path_o+'GSE167880_features.tsv',index_col=0,header=None)
features.columns=['gene_symbol','feature_type']
features.index.name="EID"

# %%
adata=[]
samples=[]
for fn in glob.glob(path_o+'*_matrix.mtx.gz'):
    fn_obs=fn.replace('matrix.mtx','barcodes.tsv')
    mtx=mmread(fn)
    cells=pd.read_table(fn_obs,index_col=0,header=None)
    cells.index.name=None
    samples.append(fn.split('/')[-1].split('_matrix')[0])
    adata.append(sc.AnnData(csr_matrix(mtx.T),obs=cells,var=features))
adata=sc.concat(adata,label='sample',keys =samples,index_unique='-',merge='same')

# %%
adata

# %%
adata.X[:10,:10].todense()

# %%
adata.write(path_o+'adata.h5ad')

# %%
