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
path_e=path_data+'datasets/d10_1038_s41422-021-00486-w/'
path_e_carina='/lustre/groups/ml01/workspace/karin.hrovatin/data/pigE_cross_species/datasets/hs/'
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
obs_info=pd.read_table(path_o+'endocrine.cluster.txt',index_col=0,skiprows=[1]
                      ).query('Dataset=="GSE167880 (Balboa 2021)"')

# %%
obs_info

# %%
obs_info['Developmental stage'].unique()

# %%
obs_info['Cell type'].value_counts()

# %%
# Parse obs_info index partially
obs_info.index=obs_info.index.str.replace('d18','D18').str.replace('d20','D20').str.replace('d25','D25')

# %%
# Map obs info index with adata obs names
cell_names=dict(zip(adata.obs_names.map(
    lambda x: x.split('_')[1]+'.'+x.split('_')[2]+'_'+x.split('-')[0]),adata.obs_names))
obs_cells=set(obs_info.index)
cells={c_parsed:c for c_parsed,c in cell_names.items() if c_parsed in obs_cells}

# %%
# Sample counts of parsed adata
pd.Series([c.split('_')[0] for c in cells ]).value_counts()

# %%
# Sample counts of obs_info
pd.Series([c.split('_')[0] for c in obs_info.index]).value_counts()

# %%
# Replace obs_info index
obs_info.index=obs_info.index.map(cells)

# %%
# N cells that are not in adata
obs_info.index.isna().sum()

# %%
adata.obs['cell_type']=obs_info['Cell type'].apply(lambda x: x.split(') ')[1])

# %%
(~adata.obs['cell_type'].isna()).sum()

# %%
adata.obs['stage']=obs_info['Developmental stage'].apply(lambda x: x.split(') ')[1])

# %%
adata.obs['sample'].unique()

# %%
adata.obs['design']=adata.obs['sample'].apply(
    lambda x: '_'.join(x.split('_')[1:]))

# %%
adata

# %%
adata.X[:10,:10].todense()

# %%
adata.write(path_o+'adata.h5ad')

# %%
#adata=sc.read(path_o+'adata.h5ad')

# %% [markdown]
# ## Embryo - pp to adata
# Get annotation from carina

# %%
counts=pd.read_table(path_e+'OMIX236-20-02_10x.txt',index_col=0)

# %%
adata=sc.AnnData(counts.T)
adata.X=csr_matrix(adata.X)

# %%
obs=sc.read(path_e_carina+'HFP_10x_ep_nodoub.h5ad',backed='r').obs.query('Batch=="Wang"').copy()
obs.index=obs.apply(
    lambda x: x['Sample']+'_'+x.name.split('-')[0],axis=1)

# %%
for col in ['annotation_nodoub_2']:
    adata.obs[col]=obs[col].astype(str) # Removes unused categ

# %%
# Obs cells missing from adata
pd.Series(['_'.join(c.split('_')[:-1]) for c in obs.index if c not in adata.obs_names]
         ).value_counts(sort=False)

# %%
# Obs cells in adata
pd.Series(['_'.join(c.split('_')[:-1]) for c in obs.index if c  in adata.obs_names]
         ).value_counts(sort=False)

# %%
# all adata cells
pd.Series(['_'.join(c.split('_')[:-1]) for c in adata.obs_names]).value_counts(sort=False)

# %%
# all obs cells, some from other datasets (removed)
obs.Sample.value_counts(sort=False)

# %% [markdown]
# C: Some cells just dont overlap

# %%
adata.obs['sample']=['_'.join(c.split('_')[:-1]) for c in adata.obs_names]
adata.obs['age']=[c.split('.')[0].split('_')[0] for c in adata.obs_names]

# %%
adata.obs

# %%
adata.obs.age.value_counts()

# %%
adata.write(path_e+'adata.h5ad')

# %% [markdown]
# ## Combine adatas

# %%
# PP embryo
adata_e=sc.read(path_e+'adata.h5ad')
adata_e.obs['cell_type']=adata_e.obs['annotation_nodoub_2'].astype('category')
adata_e.obs['system']=1
adata_e.obs=adata_e.obs[['system','sample','cell_type']]

# Subset to annotated cells
adata_e=adata_e[~adata_e.obs.cell_type.isna(),:]

# Subset to expr genes and normalise
adata_e=adata_e[:,np.array((adata_e.X>0).sum(axis=0)>20).ravel()].copy()
adata_e.layers['counts']=adata_e.X.copy()
sc.pp.normalize_total(adata_e)
sc.pp.log1p(adata_e)
# HVG
sc.pp.highly_variable_genes(
     adata=adata_e, n_top_genes=3000, flavor='cell_ranger', batch_key='sample', subset=True)

# %%
# PP organoid
adata_o=sc.read(path_o+'adata.h5ad')
adata_o.obs['system']=0
adata_o.obs=adata_o.obs[['system','sample','cell_type']]
adata_o.var_names=adata_o.var.gene_symbol.astype(str)
# Since mapping from EID to symbol some names are duplicated - remove such genes
dup_var=adata_o.var_names.value_counts()
dup_var=set(dup_var[dup_var>1].index)
adata_o=adata_o[:,[v not in dup_var for v in adata_o.var_names]]

# Subset to annotated cells
adata_o=adata_o[~adata_o.obs.cell_type.isna(),:]

# Subset to expr genes and normalise
adata_o=adata_o[:,np.array((adata_o.X>0).sum(axis=0)>20).ravel()].copy()
adata_o.layers['counts']=adata_o.X.copy()
sc.pp.normalize_total(adata_o)
sc.pp.log1p(adata_o)
# HVG
sc.pp.highly_variable_genes(
     adata=adata_o, n_top_genes=3000, flavor='cell_ranger', batch_key='sample', subset=True)

# %%
# Shared HVGs
shared_hvgs=list(set(adata_e.var_names) & set(adata_o.var_names))
len(shared_hvgs)

# %%
sorted(adata_o.obs.cell_type.unique())

# %%
sorted(adata_e.obs.cell_type.unique())

# %%
adata_e.obs['cell_type_original']=adata_e.obs['cell_type']
adata_e.obs['cell_type']=adata_e.obs['cell_type'].replace({
 'EP_0':'EP_early',
 'EP_1':"EP",
 'FEV':"EP",
 'PP':'Gamma',
 'Pre_Alpha':'Alpha',
 'Pre_Beta':'Beta',
 'Pre_Delta':'Delta',
 'Pre_Epsilon':'Epsilon'
})

# %%
adata_o.obs['cell_type_original']=adata_o.obs['cell_type']
adata_o.obs['cell_type']=adata_o.obs['cell_type'].replace({
 'Adult Alpha':'Alpha',
 'Adult Beta':'Beta',
 'Early SC-Beta':'Beta',
 'Endocrine Prog.':"EP",
 'Late SC-Beta':'Beta',
 'SC-Alpha':'Alpha',
 'SC-EC':"EP"
})

# %%
sorted(adata_o.obs.cell_type.unique())

# %%
sorted(adata_e.obs.cell_type.unique())

# %%
# Subset to shraed HVGs and concat
adata=sc.concat([adata_o[:,shared_hvgs], adata_e[:,shared_hvgs]],
                join='outer',
                index_unique='_', keys=['organoid','embryo'])
adata

# %%
adata.write(path_save+'organoid_embryo_public.h5ad')

# %%
