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
# # TODO!!! ERROR: oto_orthologues are computed wrongly, need to use keep=False in duplicated

# %%
import scanpy as sc
import pandas as pd
import pickle as pkl
import gc
import numpy as np
from scipy.sparse import csr_matrix

# %%
fn_pancreas_mm='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/combined/data_integrated_analysed'
fn_pancreas_hs='/lustre/groups/ml01/workspace/eva.lavrencic/data/pancreas/combined/adata_integrated_log_normalised_reann.h5ad'
fn_tabula_mm='/lustre/groups/ml01/workspace/karin.hrovatin/data/datasets/tabula_muris_senis/local.h5ad'
fn_tabula_hs='/lustre/groups/ml01/workspace/karin.hrovatin/data/datasets/tabula_sapiens/local.h5ad'
path_csp='/lustre/groups/ml01/workspace/karin.hrovatin/data/cross_species_prediction/'
fn_tabula_pp=path_csp+'tabula/combined_orthologues.h5ad'
fn_pancreas_pp=path_csp+'pancreas_example/combined_orthologues.h5ad'
path_save=path_csp+'pancreas_example/'
path_genes='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/gene_lists/'

# %%
# orthologues
orthologues=pd.read_table(path_genes+'orthologues_ORGmus_musculus_ORG2homo_sapiens_V103.tsv'
                         ).rename(
    {'Gene name':'gs_mm','Human gene name':'gs_hs',
     'Gene stable ID':'eid_mm','Human gene stable ID':'eid_hs'},axis=1)
# One to one orthologues - dont have same mm/hs gene in the table 2x
oto_orthologues=orthologues[~orthologues.duplicated('eid_mm').values & 
               ~orthologues.duplicated('eid_hs').values]
print(oto_orthologues.shape[0])

# %%
adata_pancreas_pp=sc.read(fn_pancreas_pp,backed='r')
adata_tabula_pp=sc.read(fn_tabula_pp,backed='r')

# %%
# Genes present in either pancreas or tabula datasets
genes=list(set(adata_pancreas_pp.var_names)|set(adata_tabula_pp.var_names))
print(len(genes))

# %%
# Get orthologues of genes to be used in both species
genes_ortho=oto_orthologues.query('eid_mm in @genes')

# %% [markdown]
# Load individual datasets, subset to relevant cells, normalize, subset to relevant genes, edit metadata

# %%
# Ct names used in pancreatic tissue in tabula 
sorted(adata_tabula_pp.obs.query('tissue=="pancreas"')['cell_type'].unique())

# %%
# Save all adata objects to be concatenated
adatas={}

# %%
# Mouse pancreas data
# Load data
adata=sc.read(fn_pancreas_mm)[adata_pancreas_pp.obs.query('system==0').index,:]
# Edit obs
adata.obs.drop(adata.obs.columns,axis=1,inplace=True)
adata.obs['batch']=adata_pancreas_pp.obs.loc[adata.obs_names,'study_sample']
adata.obs['system']=adata_pancreas_pp.obs.loc[adata.obs_names,'system']
adata.obs['cell_type']=adata_pancreas_pp.obs.loc[adata.obs_names,'cell_type_final'].map({
         'acinar':'pancreatic acinar cell',
         'alpha':'pancreatic A cell',
         'beta':'type B pancreatic cell',
         'delta':'pancreatic D cell',
         'ductal':'pancreatic ductal cell',
         'endothelial':'endothelial cell',
         'epsilon':'pancreatic epsilon cell',
         'gamma':'pancreatic PP cell',
         'immune':'leukocyte',
         'schwann':'Schwann cell',
         'stellate_activated':'pancreatic stellate cell',
         'stellate_quiescent':'pancreatic stellate cell'
    })
# Eval only pancreas data cells to make comparable to the pancreas dataset
adata.obs['eval_cells']=True
adata.obs['dataset']='pancreas'

# Get all genes
adata=adata.raw.to_adata()
# Select genes for normalisation - expressed genes + genes for integration
genes_use=set(genes)|set(adata.var_names[np.array((adata.X>0).sum(axis=0)>20).ravel()])
# Add genes missing in adata but used of integration as 0-s
genes_add=genes_use-set(adata.var_names)
adata=sc.concat([adata,
                 sc.AnnData(pd.DataFrame(
                     np.zeros((adata.shape[0],len(genes_add))).astype(np.float32),
                     index=adata.obs_names,columns=genes_add))
                ],axis=1, merge='first')
# Normalize
adata=adata[:,list(genes_use)]
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
# Subset genes
adata=adata[:,genes_ortho.eid_mm]
gc.collect()
adatas['pancreas_mm']=adata

# %%
# Human pancreas data
# Load data
adata=sc.read(fn_pancreas_hs)[adata_pancreas_pp.obs.query('system==1').index,:]
# Edit obs
adata.obs.drop(adata.obs.columns,axis=1,inplace=True)
adata.obs['batch']=adata_pancreas_pp.obs.loc[adata.obs_names,'study_sample']
adata.obs['system']=adata_pancreas_pp.obs.loc[adata.obs_names,'system']
adata.obs['cell_type']=adata_pancreas_pp.obs.loc[adata.obs_names,'cell_type_final'].map({
         'acinar':'pancreatic acinar cell',
         'alpha':'pancreatic A cell',
         'beta':'type B pancreatic cell',
         'delta':'pancreatic D cell',
         'ductal':'pancreatic ductal cell',
         'endothelial':'endothelial cell',
         'epsilon':'pancreatic epsilon cell',
         'gamma':'pancreatic PP cell',
         'immune':'leukocyte',
         'schwann':'Schwann cell',
         'stellate_activated':'pancreatic stellate cell',
         'stellate_quiescent':'pancreatic stellate cell'
    })
adata.obs['eval_cells']=True
adata.obs['dataset']='pancreas'

# Get all genes
adata=adata.raw.to_adata()
# Select genes for normalisation - expressed genes + genes for integration
genes_use=set(orthologues.query('eid_mm in @genes')['eid_hs'])|\
    set(adata.var_names[np.array((adata.X>0).sum(axis=0)>20).ravel()])
# Add genes missing in adata but used of integration as 0-s
genes_add=genes_use-set(adata.var_names)
adata=sc.concat([adata,
                 sc.AnnData(pd.DataFrame(
                     np.zeros((adata.shape[0],len(genes_add))).astype(np.float32),
                     index=adata.obs_names,columns=genes_add))
                ],axis=1, merge='first')
# Normalize
adata=adata[:,list(genes_use)]
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
# Subset genes
adata=adata[:,genes_ortho.eid_hs]
adata.var['EID']=adata.var_names
adata.var_names=genes_ortho.eid_mm
gc.collect()
adatas['pancreas_hs']=adata

# %%
# Mouse tabula data
# Load data
adata=sc.read(fn_tabula_mm)[adata_tabula_pp.obs.query('system==0').index,:]
# Edit obs
adata.obs.drop(adata.obs.columns,axis=1,inplace=True)
adata.obs=adata_tabula_pp.obs.loc[adata.obs_names,['batch','system','cell_type']]
adata.obs['eval_cells']=False
adata.obs['dataset']='tabula'
# Get all genes
adata=adata.raw.to_adata()
# Select genes for normalisation - expressed genes + genes for integration
genes_use=set(genes)|set(adata.var_names[np.array((adata.X>0).sum(axis=0)>20).ravel()])
# Add genes missing in adata but used of integration as 0-s
genes_add=genes_use-set(adata.var_names)
adata=sc.concat([adata,
                 sc.AnnData(pd.DataFrame(
                     np.zeros((adata.shape[0],len(genes_add))).astype(np.float32),
                     index=adata.obs_names,columns=genes_add))
                ],axis=1, merge='first')
# Normalize
adata=adata[:,list(genes_use)]
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
# Subset genes
adata=adata[:,genes_ortho.eid_mm]
gc.collect()
adatas['tabula_mm']=adata

# %%
# Human tabula data
# Load data
adata=sc.read(fn_tabula_hs)[adata_tabula_pp.obs.query('system==1').index,:]
# Edit obs
adata.obs.drop(adata.obs.columns,axis=1,inplace=True)
adata.obs=adata_tabula_pp.obs.loc[adata.obs_names,['batch','system','cell_type']]
adata.obs['eval_cells']=False
adata.obs['dataset']='tabula'

# Get all genes
adata=adata.raw.to_adata()
# Select genes for normalisation - expressed genes + genes for integration
genes_use=set(orthologues.query('eid_mm in @genes')['eid_hs'])|\
    set(adata.var_names[np.array((adata.X>0).sum(axis=0)>20).ravel()])
# Add genes missing in adata but used of integration as 0-s
genes_add=genes_use-set(adata.var_names)
adata=sc.concat([adata,
                 sc.AnnData(pd.DataFrame(
                     np.zeros((adata.shape[0],len(genes_add))).astype(np.float32),
                     index=adata.obs_names,columns=genes_add))
                ],axis=1, merge='first')
# Normalize
adata=adata[:,list(genes_use)]
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
# Subset genes
adata=adata[:,genes_ortho.eid_hs]
adata.var['EID']=adata.var_names
adata.var_names=genes_ortho.eid_mm
gc.collect()
adatas['tabula_hs']=adata

# %%
adatas

# %%
adata=sc.concat(list(adatas.values()))

# %%
# Mark genes to be used for eval - matching the pancreas data without tabula
adata.var.loc[adata_pancreas_pp.var_names,'eval_genes']=True
adata.var['eval_genes'].fillna(False, inplace=True)

# %%
adata

# %%
print(adata.obs.eval_cells.sum(),adata.var.eval_genes.sum())

# %%
adata.write(path_save+'combined_tabula_orthologues.h5ad')

# %%
#adata=sc.read(path_save+'combined_tabula_orthologues.h5ad')

# %%
