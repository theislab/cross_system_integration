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
path_sn=path_data+'datasets/d10_1038_s41586-022-04521-7/'
path_sc=path_data+'datasets/'
path_save=path_data+'cross_species_prediction/brainVasculature_sc_sn/'

# %% [markdown]
# ## SN data - pp to adata

# %% [markdown]
# Control

# %%
# Load matrix
mtx=mmread(path_sn+'brain.BBB.human.counts.GEOsubmission.mtx')
# Make adata
adata=sc.AnnData(
    csr_matrix(mtx.T),
    obs=pd.read_table(path_sn+'brain.BBB.human.droplets.GEOsubmission.tsv',
                      index_col=0,header=None),
    var=pd.read_table(path_sn+'brain.BBB.human.features.GEOsubmission.tsv',
                      index_col=0,header=None),
)
# Add obs
obs=pd.read_table(path_sn+'brain.BBB.human.meta.txt',index_col=0)
print(obs.shape)
for col in obs.columns:
    adata.obs[col]=obs[col]
adata.obs['cellsubtype']=pd.read_table(
    path_sn+'brain.BBB.human.vascular.final.Jan2022.metadata.txt',index_col=0)['cellsubtype']
# Needed for saving
adata.obs.index.name=None
adata.var.index.name=None

display(adata)

# %%
adata.obs

# %%
adata.obs['celltype'].value_counts()

# %%
adata.obs['cellsubtype'].value_counts()

# %%
adata.var

# %%
adata.X[:20,:15].todense()

# %%
# Save
adata.write(path_sn+'brain.BBB.human.counts.GEOsubmission.h5ad')

# %%
#adata=sc.read(path_sn+'brain.BBB.human.counts.GEOsubmission.h5ad')

# %% [markdown]
# HD

# %%
# Load matrix
#mtx=mmread(path_sn+'brain.BBB.humanHD.counts.GEOsubmission.mtx')
# Make adata
adata=sc.AnnData(
    csr_matrix(mtx.T),
    obs=pd.read_table(path_sn+'brain.BBB.humanHD.droplets.GEOsubmission.tsv',
                      index_col=0,header=None),
    var=pd.read_table(path_sn+'brain.BBB.humanHD.features.GEOsubmission.tsv',
                      index_col=0,header=None),
)
# Add obs
obs=pd.read_table(path_sn+'brain.HD.snRNAseq.metadata.txt',index_col=0)
print(obs.shape)
for col in obs.columns:
    adata.obs[col]=obs[col]
adata.obs['cellsubtype']=pd.read_table(
    path_sn+'brain.human.HD.vascular.seurat.harmony.metadata.celltype.txt',index_col=0
)['sub.celltype']
# Needed for saving
adata.obs.index.name=None
adata.var.index.name=None

display(adata)

# %%
adata.obs

# %%
adata.obs['celltype'].value_counts()

# %%
adata.obs['cellsubtype'].value_counts()

# %%
adata.obs.Condition.unique()

# %%
adata.var

# %%
adata.X[:20,20:35].todense()

# %%
# Save
adata.write(path_sn+'brain.BBB.humanHD.counts.GEOsubmission.h5ad')

# %%
#adata=sc.read(path_sn+'brain.BBB.humanHD.counts.GEOsubmission.h5ad')

# %% [markdown]
# ## SN - pp for integration

# %%
# Load data
adata_sn_cont=sc.read(path_sn+'brain.BBB.human.counts.GEOsubmission.h5ad')
adata_sn_hd=sc.read(path_sn+'brain.BBB.humanHD.counts.GEOsubmission.h5ad')

# %%
# Format obs
adata_sn_cont.obs['dataset']='control'
adata_sn_hd.obs['dataset']='HD'
adata_sn_cont.obs=adata_sn_cont.obs[['SampleID', 'PatientID','snRNAPreparation',
                                    'celltype','cellsubtype','dataset']]
adata_sn_hd.obs['SampleID']=adata_sn_hd.obs['Sample.ID']
adata_sn_hd.obs=adata_sn_hd.obs[['SampleID', 'Condition','Brain.Region',
                                 'celltype','cellsubtype','dataset']]

# %%
# Concat and subset to cells with anno
adata_sn=sc.concat([adata_sn_cont,adata_sn_hd],join='outer')
adata_sn=adata_sn[~adata_sn.obs.celltype.isna(),:]

# %%
adata_sn

# %%
# Add raw expression to X, remove lowly expr genes, and normalise
adata_sn_raw=adata_sn.copy()
adata_sn=adata_sn[:,np.array((adata_sn.X>0).sum(axis=0)>20).ravel()]
sc.pp.normalize_total(adata_sn)
sc.pp.log1p(adata_sn)

# %%
# HVGs on the final cell subset
sc.pp.highly_variable_genes(
     adata=adata_sn, n_top_genes=3000, flavor='cell_ranger', batch_key='SampleID',
    subset=True)
adata_sn.shape

# %%
adata_sn

# %%
