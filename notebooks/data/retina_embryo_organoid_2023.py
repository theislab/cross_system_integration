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
from scipy.io import mmread
from scipy.sparse import csr_matrix
import glob

# %%
path_ds='/net/bmc-lab6/data/lab/kellis/users/khrovati/data/datasets/d10_1038_s41598-023-28429-y/'

# %%
path_train='/net/bmc-lab6/data/lab/kellis/users/khrovati/data/cross_system_integration/retina_embryo_organoid_2023/'

# %% [markdown]
# ## Make AnnData

# %% [markdown]
# Load data

# %%
# Get all samples, file names based on them
samples=[]
for f in sorted(glob.glob(path_ds+'Seurat_velocyto_monocle*matrix.mtx')):
    samples.append(f.split('Seurat_velocyto_monocle :')[1].replace(':matrix.mtx',''))
print(samples)

# %%
# Mapping between file name samples and annotation samples
sample_map={
    'IPS1 ':'DD13',
    'IPS2':'DD21',
    'IPS3':'DD25',
    'IPS4':'DD29',
    'RE1':'E13',
    'RE2':'P0',
    'RE3':'P5',
    'RE4':'P9',
}

# %%
# Load data
adata={}
for fn,sample in sample_map.items():
    #print(sample)
    var=pd.read_table(path_ds+'Seurat_velocyto_monocle :'+fn+':genes.tsv',header=None,index_col=0)
    var.columns=['gene_symbol']
    obs=pd.read_table(path_ds+'Seurat_velocyto_monocle :'+fn+':barcodes.tsv',header=None,index_col=0)
    obs.index.name=None
    var.index.name=None
    fn_mtx=path_ds+'Seurat_velocyto_monocle :'+fn+':matrix.mtx'
    try:
        mtx=csr_matrix(mmread(fn_mtx)).T
    except ValueError: 
        fn_mtx=path_ds+'Seurat_velocyto_monocle :'+fn+':matrix.mtx'
        mtx=pd.read_table(fn_mtx,header=None,sep=' ').values
        mtx=csr_matrix((mtx[:,2],(mtx[:,1]-1,mtx[:,0]-1)),shape=(obs.shape[0],var.shape[0]))
    adata[sample]=sc.AnnData(mtx,obs=obs,var=var)
adata=sc.concat(adata.values(),label='sample',keys=adata.keys(),index_unique ='-',merge ='same')

# %%
adata

# %% [markdown]
# Add annotation

# %%
anno=pd.read_table(path_ds+'Annotations_cells.csv',sep=',',index_col=0)

# %%
anno.index.map(lambda x: x.split('_')[:-1]).value_counts()

# %%
adata.obs.index.map(lambda x: x.split('-')[1:]).value_counts()

# %%
anno.index=anno.index.map(lambda x: '-'.join(
    [x.split('_')[-1],x.split('_')[1] if len(x.split('_'))==3 else '1',x.split('_')[0]]))

# %%
anno.index.map(lambda x: x.split('-')[1:]).value_counts()

# %%
# Find cells that are probably not correct from anno
adata_cells=set(adata.obs_names)
anno_cells=set(anno.index)
print(anno_cells-adata_cells)

# %%
cols=['TimePoint','Model','Annotations']
cells=list(anno_cells&adata_cells)
adata.obs.loc[cells,cols]=anno.loc[cells,cols]

# %%
# Fix condition and sample (bio replicates)
adata.obs['sample']=adata.obs.index.map(
    lambda x: x.split('-')[2] +'_'+x.split('-')[1] if len(x.split('-'))==3 else x.split('-')[1]+'_1')

# %%
adata.obs['sample'].value_counts(sort=False)

# %%
adata.obs['batch']=adata.obs['sample'].map(lambda x: x.split('_')[0])

# %% [markdown]
# There are two bio samples per condition, except for the 4th stage in both models. But they were sequenced together:
#
# We generated two biological replicate samples (NaR and RO) for stages I to III and one biological replicate for stage IV. We loaded ~ 15,700 cells for biological replicate 1 (stage I–IV) and ~ 10,000 cells for biological replicate 2 (stage I–III) in a Chromium Controller instrument (10× Genomics, Pleasanton, CA).

# %%
sorted(adata.obs['TimePoint'].dropna().unique())

# %%
adata.obs['stage']=adata.obs['TimePoint'].map({
    'DD13':1, 
    'DD21':2, 
    'DD25':3, 
    'DD29':4, 
    'E13':1, 
    'P0':2, 
    'P5':3, 
    'P9':4
})

# %%
adata

# %% [markdown]
# Save data

# %%
adata.write(path_ds+'adata_Seurat_velocyto_monocle.h5ad')

# %% [markdown]
# ## Training data

# %%
#adata=sc.read(path_ds+'adata_Seurat_velocyto_monocle.h5ad')

# %%
adata

# %%
# Subset to annotated cells
adata_sub=adata[~adata.obs.Annotations.isna(),:].copy()

# %%
# Keep not too lowly expressed genes as intersection of the two systems
adata_sub=adata_sub[:,
                    np.array((adata_sub[adata_sub.obs.Model=="iPS",:].X>0).sum(axis=0)>20).ravel()&\
                    np.array((adata_sub[adata_sub.obs.Model=="Retinal",:].X>0).sum(axis=0)>20).ravel()
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
    adata_sub[adata_sub.obs.Model=="iPS",:], 
    n_top_genes=3000, flavor='cell_ranger', inplace=False, batch_key='sample').query('highly_variable==True').index)&\
set(sc.pp.highly_variable_genes(
    adata_sub[adata_sub.obs.Model=="Retinal",:], 
    n_top_genes=3000, flavor='cell_ranger', inplace=False, batch_key='sample').query('highly_variable==True').index)
print(len(hvgs))

# %%
adata_sub=adata_sub[:,list(hvgs)]

# %%
del adata_sub.uns

# %%
adata_sub.obs['system']=adata_sub.obs['Model'].map({"iPS":0,'Retinal':1})

# %%
adata_sub.layers['counts']=adata[adata_sub.obs_names,adata_sub.var_names].X.copy()

# %%
adata_sub

# %%
adata_sub.write(path_train+'combined_HVG.h5ad')

# %%
#adata_sub=sc.read(path_train+'combined_HVG.h5ad')

# %%
