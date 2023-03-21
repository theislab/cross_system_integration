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
import seaborn as sb

# %%
path_data='/lustre/groups/ml01/workspace/karin.hrovatin/data/'
path_sub=path_data+'datasets/d10_1038_s41586-022-04518-2/SCP1376/'
path_save=path_data+'cross_species_prediction/adipose_sc_sn/'

# %% [markdown]
# ## Metadata of cells

# %%
# Metadata for both sc and sn
meta=pd.read_table(path_sub+'metadata/metadata.scp.tsv',index_col=0,skiprows=[1])

# %%
for col in meta.columns:
    print('\n***',col,meta[col].nunique())
    if meta[col].nunique()<30:
        print(meta[col].unique())
        display(meta[col].value_counts())

# %% [markdown]
# ## Sc data

# %%
fn=path_sub+'expression/618069d6771a5b396cca7a7d/HsDrop.counts'

# %%
x=mmread(fn+'.mtx.gz')
features=pd.DataFrame(pd.read_table(fn+'.features.tsv.gz',header=None,index_col=0)[1])
features.index.name=None
features.columns=['gene_symbol']
barcodes=pd.read_table(fn+'.barcodes.tsv.gz',header=None,index_col=0)
barcodes.index.name=None

# %%
adata=sc.AnnData(csr_matrix(x.T),var=features,obs=barcodes)
adata

# %%
cols=['depot__ontology_label','donor_id','sex','disease__ontology_label',
    'fat__type','cell_cycle__phase','cluster','subcluster']
adata.obs[cols]=meta.loc[adata.obs_names,cols]

# %%
for col in cols:
    print('\n***',col,adata.obs[col].nunique())
    if adata.obs[col].nunique()<30:
        print(adata.obs[col].unique())
        display(adata.obs[col].value_counts())

# %%
pd.crosstab(adata.obs['subcluster'],adata.obs['cluster'])

# %%
adata

# %%
adata.write(path_sub+'HsDrop.h5ad')

# %% [markdown]
# ## Sn data

# %%
fn=path_sub+'expression/618065be771a5b54fcddaed6/Hs10X.counts'

# %%
x=mmread(fn+'.mtx.gz')
features=pd.DataFrame(pd.read_table(fn+'.features.tsv.gz',header=None,index_col=0)[1])
features.index.name=None
features.columns=['gene_symbol']
barcodes=pd.read_table(fn+'.barcodes.tsv.gz',header=None,index_col=0)
barcodes.index.name=None

# %%
adata=sc.AnnData(csr_matrix(x.T),var=features,obs=barcodes)
adata

# %%
cols=['depot__ontology_label','donor_id','sex','disease__ontology_label',
    'fat__type','cell_cycle__phase','cluster','subcluster']
adata.obs[cols]=meta.loc[adata.obs_names,cols]

# %%
for col in cols:
    print('\n***',col,adata.obs[col].nunique())
    if adata.obs[col].nunique()<30:
        print(adata.obs[col].unique())
        display(adata.obs[col].value_counts())

# %%
pd.crosstab(adata.obs['subcluster'],adata.obs['cluster'])

# %%
adata

# %%
adata.write(path_sub+'Hs10X.h5ad')

# %% [markdown]
# ## Combien adatas

# %% [markdown]
# Sn need to subset to same type of fat as sc as else unmatched cts. In cs dont need to remove cancer partients as this does not seem to affect fat.

# %%
# PP sc
adata_sc=sc.read(path_sub+'HsDrop.h5ad')
# Subset to fat type and annotated cells
adata_sc=adata_sc[adata_sc.obs.fat__type=="SAT",:]
adata_sc=adata_sc[~adata_sc.obs.cluster.isna(),:]
# metadata
adata_sc.obs['system']=0
adata_sc.obs=adata_sc.obs[['system','donor_id','cluster','subcluster']]


# Subset to expr genes and normalise
adata_sc=adata_sc[:,np.array((adata_sc.X>0).sum(axis=0)>20).ravel()].copy()
adata_sc.layers['counts']=adata_sc.X.copy()
sc.pp.normalize_total(adata_sc)
sc.pp.log1p(adata_sc)
# HVG
sc.pp.highly_variable_genes(
     adata=adata_sc, n_top_genes=3000, flavor='cell_ranger', batch_key='donor_id', subset=True)

display(adata_sc)

# %%
# PP sn
adata_sn=sc.read(path_sub+'Hs10X.h5ad')
# Subset to fat type and annotated cells
adata_sn=adata_sn[adata_sn.obs.fat__type=="SAT",:]
adata_sn=adata_sn[~adata_sn.obs.cluster.isna(),:]
# metadata
adata_sn.obs['system']=1
adata_sn.obs=adata_sn.obs[['system','donor_id','cluster','subcluster']]


# Subset to expr genes and normalise
adata_sn=adata_sn[:,np.array((adata_sn.X>0).sum(axis=0)>20).ravel()].copy()
adata_sn.layers['counts']=adata_sn.X.copy()
sc.pp.normalize_total(adata_sn)
sc.pp.log1p(adata_sn)
# HVG
sc.pp.highly_variable_genes(
     adata=adata_sn, n_top_genes=3000, flavor='cell_ranger', batch_key='donor_id', subset=True)

display(adata_sn)

# %%
# Shared HVGs
shared_hvgs=list(set(adata_sc.var_names) & set(adata_sn.var_names))
len(shared_hvgs)

# %%
sorted(adata_sc.obs.cluster.unique())

# %%
sorted(adata_sc.obs.cluster.unique())

# %%
# Subset to shraed HVGs and concat
adata=sc.concat([adata_sc[:,shared_hvgs], adata_sn[:,shared_hvgs]],
                join='outer',
                index_unique='_', keys=['sc','sn'])
adata

# %%
adata.write(path_save+'adiposeHsSAT_sc_sn.h5ad')

# %%
