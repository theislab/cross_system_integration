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

# %% [markdown]
# # Prepare pancreas data example

# %%
import scanpy as sc
import pandas as pd
import pickle as pkl
import gc
import numpy as np
from scipy.sparse import csr_matrix

import seaborn as sb
import matplotlib.pyplot as plt
from matplotlib import rcParams

# %%
path_mm='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/combined/'
fn_hs='/lustre/groups/ml01/workspace/eva.lavrencic/data/pancreas/combined/adata_integrated_log_normalised_reann.h5ad'
path_save='/lustre/groups/ml01/workspace/karin.hrovatin/data/cross_species_prediction/pancreas_example/'
path_hs_symbols='/lustre/groups/ml01/projects/2020_pancreas_karin.hrovatin/data/pancreas/scRNA/P21002/rev7/cellranger/MUC26033/count_matrices/filtered_feature_bc_matrix/features.tsv'
path_genes='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/gene_lists/'

# %% [markdown]
# ## HVGs

# %% [markdown]
# ### Load and subset mouse and human data

# %% [markdown]
# #### Mouse

# %%
adata_mm=sc.read(path_mm+'data_integrated_analysed.h5ad')

# %%
# Add cell types
adata_mm.obs['cell_type_integrated_v1']=sc.read(
    path_mm+'data_integrated_analysed.h5ad',backed='r'
).obs.loc[adata_mm.obs_names,'cell_type_integrated_v1'].copy()
adata_mm.obs['cell_type_final']=adata_mm.obs['cell_type_integrated_v1']

# %%
# Subset to N cells per ct and sample - use only healthy adults
n=1000
cells=[]
for ct in ['acinar', 'alpha', 'beta','delta', 'ductal', 'endothelial','gamma', 'immune',
           'schwann','stellate_activated','stellate_quiescent']:
    for sample in ['STZ_G1_control','VSG_MUC13633_chow_WT','VSG_MUC13634_chow_WT',
         'Fltp_adult_mouse1_head_Fltp-','Fltp_adult_mouse2_head_Fltp+',
         'Fltp_adult_mouse4_tail_Fltp+','Fltp_adult_mouse3_tail_Fltp-']:
        cells.extend(
            adata_mm.obs.query('cell_type_final==@ct & study_sample_design==@sample').index[:n])
print(len(cells))
adata_mm=adata_mm[cells,:]

# %%
# Add raw expression to X, remove lowly expr genes, and normalise
adata_mm=adata_mm.raw.to_adata()
adata_mm=adata_mm[:,np.array((adata_mm.X>0).sum(axis=0)>20).ravel()]
sc.pp.normalize_total(adata_mm)
sc.pp.log1p(adata_mm)

# %%
# HVGs on the final cell subset
sc.pp.highly_variable_genes(
     adata=adata_mm, n_top_genes=1000, flavor='cell_ranger', batch_key='study_sample',
    subset=True)
adata_mm.shape

# %%
adata_mm

# %%
adata_mm.write(path_save+'mouse.h5ad')

# %%
del adata_mm

# %%
gc.collect()

# %% [markdown]
# #### Human

# %%
adata_hs=sc.read(fn_hs)

# %%
# Healthy adult
adata_hs.obs['age_years']=adata_hs.obs['age'].str.replace(' y','').astype(float)
adata_hs=adata_hs[(adata_hs.obs['age_years']>=19) & (adata_hs.obs['age_years']<=64)  &\
    (adata_hs.obs['disease']=='healthy') & (adata_hs.obs['study']!='human_spikein_drug'),
                  :].copy()

# %%
adata_hs.obs['cell_type_final']=adata_hs.obs['cell_type_integrated_subcluster']

# %%
# Subset to N cells per ct and sample - use only healthy adults
n=1000
cells=[]
for ct in ['acinar', 'alpha', 'beta','delta', 'ductal', 'epsilon','endothelial','gamma', 
           'immune', 'schwann','stellate_activated','stellate_quiescent']:
    for sample in adata_hs.obs.study_sample.unique():
        cells.extend(
            adata_hs.obs.query('cell_type_final==@ct & study_sample==@sample').index[:n])
print(len(cells))
adata_hs=adata_hs[cells,:]

# %%
# Add raw expression to X, remove lowly expr genes, and normalise
adata_hs=adata_hs.raw.to_adata()
adata_hs=adata_hs[:,np.array((adata_hs.X>0).sum(axis=0)>20).ravel()]
sc.pp.normalize_total(adata_hs)
sc.pp.log1p(adata_hs)

# %%
# HVGs on the final cell subset
sc.pp.highly_variable_genes(
     adata=adata_hs, n_top_genes=1000, flavor='cell_ranger', batch_key='study_sample',
    subset=True)
adata_hs.shape

# %%
# Add gene symbols to adata
adata_hs.var['gene_symbol']=pd.read_table(path_hs_symbols,index_col=0,header=None
                          ).rename(
    {1:'gene_symbol'},axis=1).loc[adata_hs.var_names,'gene_symbol']

# %%
adata_hs

# %%
adata_hs.write(path_save+'human.h5ad')

# %%
del adata_hs

# %%
gc.collect()

# %% [markdown]
# ## One-to-one orthologue HVGs

# %%
# Orthologues
orthology_info=pd.read_table(path_genes+'orthologues_ORGmus_musculus_ORG2homo_sapiens_V103.tsv'
                         ).rename(
    {'Gene name':'gs_mm','Human gene name':'gs_hs',
     'Gene stable ID':'eid_mm','Human gene stable ID':'eid_hs'},axis=1)

# %%
# One to one orthologues - dont have same mm/hs gene in the table 2x
oto_orthologues=orthology_info[~orthology_info.duplicated('eid_mm').values & 
               ~orthology_info.duplicated('eid_hs').values]

# %% [markdown]
# ### Load and subset mouse and human data

# %% [markdown]
# #### Mouse

# %%
adata_mm=sc.read(path_mm+'data_integrated_analysed.h5ad')

# %%
# Add cell types
adata_mm.obs['cell_type_integrated_v1']=sc.read(
    path_mm+'data_integrated_analysed.h5ad',backed='r'
).obs.loc[adata_mm.obs_names,'cell_type_integrated_v1'].copy()
adata_mm.obs['cell_type_final']=adata_mm.obs['cell_type_integrated_v1']

# %%
# Subset to N cells per ct and sample - use only healthy adults
n=1000
cells=[]
for ct in ['acinar', 'alpha', 'beta','delta', 'ductal', 'endothelial','gamma', 'immune',
           'schwann','stellate_activated','stellate_quiescent']:
    for sample in ['STZ_G1_control','VSG_MUC13633_chow_WT','VSG_MUC13634_chow_WT',
         'Fltp_adult_mouse1_head_Fltp-','Fltp_adult_mouse2_head_Fltp+',
         'Fltp_adult_mouse4_tail_Fltp+','Fltp_adult_mouse3_tail_Fltp-']:
        cells.extend(
            adata_mm.obs.query('cell_type_final==@ct & study_sample_design==@sample').index[:n])
print(len(cells))
adata_mm=adata_mm[cells,:]

# %%
# Add raw expression to X, remove lowly expr genes, subset to O-to-O orthologues, and normalise
adata_mm=adata_mm.raw.to_adata()
adata_mm=adata_mm[:,np.array((adata_mm.X>0).sum(axis=0)>20).ravel()]
adata_mm=adata_mm[:,[g for g in oto_orthologues.eid_mm if g in adata_mm.var_names]]
sc.pp.normalize_total(adata_mm)
sc.pp.log1p(adata_mm)

# %%
# HVGs on the final cell subset
sc.pp.highly_variable_genes(
     adata=adata_mm, n_top_genes=2000, flavor='cell_ranger', batch_key='study_sample',
    subset=True)
adata_mm.shape

# %%
adata_mm

# %%
#adata_mm.write(path_save+'mouse.h5ad')

# %%
#del adata_mm

# %%
#gc.collect()

# %% [markdown]
# #### Human

# %%
adata_hs=sc.read(fn_hs)

# %%
# Healthy adult
adata_hs.obs['age_years']=adata_hs.obs['age'].str.replace(' y','').astype(float)
adata_hs=adata_hs[(adata_hs.obs['age_years']>=19) & (adata_hs.obs['age_years']<=64)  &\
    (adata_hs.obs['disease']=='healthy') & (adata_hs.obs['study']!='human_spikein_drug'),
                  :].copy()

# %%
adata_hs.obs['cell_type_final']=adata_hs.obs['cell_type_integrated_subcluster']

# %%
# Subset to N cells per ct and sample - use only healthy adults
n=1000
cells=[]
for ct in ['acinar', 'alpha', 'beta','delta', 'ductal', 'epsilon','endothelial','gamma', 
           'immune', 'schwann','stellate_activated','stellate_quiescent']:
    for sample in adata_hs.obs.study_sample.unique():
        cells.extend(
            adata_hs.obs.query('cell_type_final==@ct & study_sample==@sample').index[:n])
print(len(cells))
adata_hs=adata_hs[cells,:]

# %%
# Add raw expression to X, remove lowly expr genes, and normalise
adata_hs=adata_hs.raw.to_adata()
adata_hs=adata_hs[:,np.array((adata_hs.X>0).sum(axis=0)>20).ravel()]
adata_hs=adata_hs[:,[g for g in oto_orthologues.eid_hs if g in adata_hs.var_names]]
sc.pp.normalize_total(adata_hs)
sc.pp.log1p(adata_hs)

# %%
# HVGs on the final cell subset
sc.pp.highly_variable_genes(
     adata=adata_hs, n_top_genes=2000, flavor='cell_ranger', batch_key='study_sample',
    subset=True)
adata_hs.shape

# %%
# Add gene symbols to adata
adata_hs.var['gene_symbol']=pd.read_table(path_hs_symbols,index_col=0,header=None
                          ).rename(
    {1:'gene_symbol'},axis=1).loc[adata_hs.var_names,'gene_symbol']

# %%
adata_hs

# %%
#adata_hs.write(path_save+'human.h5ad')

# %%
#del adata_hs

# %%
#gc.collect()

# %% [markdown]
# ### Shared genes

# %%
# Find shared HVGs
eids_hs=set(adata_hs.var_names)
eids_mm=set(adata_mm.var_names)
shared_orthologues=oto_orthologues.query('eid_hs in @eids_hs')
shared_orthologues=shared_orthologues.query('eid_mm in @eids_mm')
print('N shared:',shared_orthologues.shape[0])

# %%
# Subset adatas to shared HVGs
# This already ensures same gene order
adata_hs=adata_hs[:,shared_orthologues.eid_hs]
adata_mm=adata_mm[:,shared_orthologues.eid_mm]


# %%
print(adata_hs)
print(adata_mm)

# %%
adata_hs.var

# %%
adata_mm.var

# %%
adata_hs.write(path_save+'human_orthologues.h5ad')
adata_mm.write(path_save+'mouse_orthologues.h5ad')

# %% [markdown]
# ### Prepare data nfor model eval

# %%
adata_hs=sc.read(path_save+'human_orthologues.h5ad')
adata_mm=sc.read(path_save+'mouse_orthologues.h5ad')

# %%
# Prepare adatas for concat and concat
# Human
adata_hs_sub=adata_hs.copy()
adata_hs_sub.obs=adata_hs_sub.obs[['cell_type_final','study_sample']]
adata_hs_sub.obs['system']=1
adata_hs_sub.var['EID']=adata_hs_sub.var_names
adata_hs_sub.var_names=adata_mm.var_names
del adata_hs_sub.obsm
# Mouse
adata_mm_sub=adata_mm.copy()
adata_mm_sub.obs=adata_mm_sub.obs[['cell_type_final','study_sample']]
adata_mm_sub.obs['system']=0
del adata_mm_sub.obsm
# Concat
adata=sc.concat([adata_mm_sub,adata_hs_sub])

del adata_mm_sub
del adata_hs_sub
gc.collect()

display(adata)

# %%
adata.write(path_save+'combined_orthologues.h5ad')

# %% [markdown]
# ### Human mouse correlation on integration genes
# Are respective cell types in mouse and human correlated based on the orthologues as expected?

# %% [markdown]
# #### Correlation on cell type means

# %%
# Mean expression per ct in mm and hs
# Subset to orthologue genes
x_mm=adata_mm.to_df()
x_mm['cell_type']=adata_mm.obs['cell_type_final']
x_mm=x_mm.groupby('cell_type').mean()

x_hs=adata_hs.to_df()
x_hs['cell_type']=adata_hs.obs['cell_type_final']
x_hs=x_hs.groupby('cell_type').mean()

# %%
# Correlations
cors=pd.DataFrame(index=x_mm.index,columns=x_hs.index)
for ct_mm in x_mm.index:
    for ct_hs in x_hs.index:
        cors.at[ct_mm,ct_hs]=np.corrcoef(x_mm.loc[ct_mm,:],x_hs.loc[ct_hs,:])[0,1]        
cors.index.name='mm'
cors.columns.name='hs'
cors=cors.astype(float)

# %%
sb.clustermap(cors,vmin=-1,vmax=1,cmap='coolwarm',figsize=(5,5))

# %% [markdown]
# What if we only use genes expressed in both ocompared cell groups? Maybe drouput is problematic for corr (although unlikely as here we use pseudobulk)

# %%
# Correlations
cors=pd.DataFrame(index=x_mm.index,columns=x_hs.index)
for ct_mm in x_mm.index:
    for ct_hs in x_hs.index:
        genes=list(set(np.argwhere((x_mm.loc[ct_mm,:]>0).values).ravel()) & 
                   set(np.argwhere((x_hs.loc[ct_hs,:]>0).values).ravel()))
        cors.at[ct_mm,ct_hs]=np.corrcoef(x_mm.loc[ct_mm,:].iloc[genes],
                                         x_hs.loc[ct_hs,:].iloc[genes])[0,1]        
cors.index.name='mm'
cors.columns.name='hs'
cors=cors.astype(float)

# %%
sb.clustermap(cors,vmin=-1,vmax=1,cmap='coolwarm',figsize=(5,5))

# %% [markdown]
# C: Selecting only shared expressed genes does not help

# %% [markdown]
# Another similar analysis in the xybiomodel notebook

# %%
