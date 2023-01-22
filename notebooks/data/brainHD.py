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
from collections import defaultdict

import gc

from matplotlib import rcParams

# %%
path_data='/lustre/groups/ml01/workspace/karin.hrovatin/data/datasets/'
path_data_lee=path_data+'d10_1016_j_neuron_2020_06_021/'
path_genes='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/gene_lists/'
path_save='/lustre/groups/ml01/workspace/karin.hrovatin/data/cross_species_prediction/brainHD_example/'

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

# %% [markdown]
# ## Lee

# %% [markdown]
# ### Data from GEO

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
    adata.write(path_data_lee+prefix.rstrip('_')+'.h5ad')

# %%
for prefix in prefixes:
    print('\n****',prefix)
    obs=sc.read(path_data_lee+prefix+'.h5ad',backed='r').obs
    for col in obs.columns:
        print(col)
        if obs[col].nunique()<30:
            print(sorted(obs[col].unique()))
gc.collect()

# %% [markdown]
# ### Data from Sebastian

# %%
adatas={}
for name in ['Human','R62','zQ175']:
    adatas[name]=sc.read(path_data_lee+f'sebastian_data/ace_hd_data_{name}.h5ad',backed='r')
    print(name)
    print(adatas[name])

# %%
for name,adata in adatas.items():
    print('\n****',name)
    for col in adata.obs.columns:
        print(col,adata.obs[col].nunique())
        if adata.obs[col].nunique()<30:
            print(sorted(adata.obs[col].unique()))
gc.collect()

# %%
rcParams['figure.figsize']=(6,6)
np.random.seed(0)
random_indices=np.random.permutation(adatas['Human'].obs_names)
# # ! Need to first run the plot without cekll ordering so that uns colors are set 
# before making a view (reordering cells) - then the plotting works as else tries
# to copy adata in backed mode to set uns of copied view, which fails
sc.pl.embedding(adatas['Human'][random_indices]
                ,'ACTIONet2D',
                color=['Batch','Region','Condition', 'Grade', 
                       'PMI','Sex', 'Age','SubType','CellType'],wspace=0.7)

# %%
rcParams['figure.figsize']=(6,6)
np.random.seed(0)
random_indices=np.random.permutation(adatas['zQ175'].obs_names)
# # ! Need to first run the plot without cekll ordering so that uns colors are set 
# before making a view (reordering cells) - then the plotting works as else tries
# to copy adata in backed mode to set uns of copied view, which fails
sc.pl.embedding(adatas['zQ175'][random_indices]
                ,'ACTIONet2D',
                color=['Batch','Condition','SubType','CellType'],wspace=0.7)

# %%
rcParams['figure.figsize']=(6,6)
np.random.seed(0)
random_indices=np.random.permutation(adatas['R62'].obs_names)
# # ! Need to first run the plot without cekll ordering so that uns colors are set 
# before making a view (reordering cells) - then the plotting works as else tries
# to copy adata in backed mode to set uns of copied view, which fails
sc.pl.embedding(adatas['R62'][random_indices]
                ,'ACTIONet2D',
                color=['Batch','Condition','SubType','CellType'],wspace=0.7)

# %% [markdown]
# ## Sun

# %%
# TODO read object - they are rds but all sparse matrices so can read with rpy2 easily

# %% [markdown]
# ## Model data
# Prepare data for model input

# %%
names=['human','R62','zQ175']

# %%
adatas={}
for name in names:
    adatas[name]=sc.read(path_data_lee+f'GSE152058_{name}_snRNA_processed.h5ad')

# %%
# Subset to N cells per ct and sample - select cells
n=500
cells=defaultdict(list)
for name, adata in adatas.items():
    print(name)
    for ct in adata.obs['CellType'].unique():
        for sample in adata.obs['Batch'].unique():
            cells[name].extend(
                adata.obs.query('CellType==@ct & Batch==@sample').index[:n])
    print(len(cells[name]))


# %%
# Subset to cells and filter genes,  subset to orthologues and rename human genes, 
# add system info, normalize, compute HVGs
for name,adata in adatas.items():
    print(name)
    adata=adata[cells[name],np.array((adata.X>0).sum(axis=0)>20).ravel()]
    if name!='human':
        adata=adata[:,[g for g in oto_orthologues.eid_mm if g in adata.var_names]].copy()
        adata.obs['system']=0
    else:
        adata=adata[:,[g for g in oto_orthologues.eid_hs if g in adata.var_names]].copy()
        adata.var['EID']=adata.var_names
        oto_orthologues_copy=oto_orthologues.copy()
        oto_orthologues_copy.index=oto_orthologues_copy.eid_hs
        adata.var_names=oto_orthologues_copy.loc[adata.var_names,'eid_mm']
        del oto_orthologues_copy
        adata.obs['system']=1
    adata.obs['dataset']=name
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    adata.obs['Batch']=adata.obs['Batch'].astype('category')
    sc.pp.highly_variable_genes(
     adata=adata, n_top_genes=3000, flavor='cell_ranger', batch_key='Batch', subset=True)
    adatas[name]=adata
    print(adata.shape)

# %%
# Shared HVGs
shared_hvgs=list(set.intersection(*[set(adata.var_names) for adata in adatas.values()]))
len(shared_hvgs)

# %%
# Subset to shraed HVGs and concat
adata=sc.concat([adata[:,shared_hvgs] for adata in adatas.values()])

# %%
adata.obs.loc[adatas['human'].obs_names,'Grade']=adatas['human'].obs['Grade']

# %%
adata

# %%
adata.write(path_save+'combined_orthologues.h5ad')

# %%
