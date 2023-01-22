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
# Find and analyse correlation between human endocrine markers in Tabula Sapiens, excludin gendocrine cell types (to ensure OOD for later).

# %%
import scanpy as sc
import pandas as pd
import pickle as pkl
import numpy as np

from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
import seaborn as sb
from matplotlib import rcParams

# %%
path_genes='/lustre/groups/ml01/workspace/karin.hrovatin/data/cross_species_prediction/pancreas_ct_example_constraint/'
path_data='/lustre/groups/ml01/workspace/karin.hrovatin/data/datasets/tabula_sapiens/'

# %%
genes_ct=pkl.load(open(path_genes+'endo_markers_set_hs.pkl','rb'))
genes=set([g for gs in genes_ct.values() for g in gs])

# %% [markdown]
# Load tabula sapiens

# %%
adata_full=sc.read(path_data+'local.h5ad')

# %%
adata_full

# %%
adata_full.uns['X_normalization']

# %% [markdown]
# Subset to endocrine marker genes

# %%
# Subset adata to only marker genes
print('N markers:',len(genes))
adata_full=adata_full[:,adata_full.var.query('feature_name in @genes').index].copy()
print('N marker genes in adata:',adata_full.shape[1])
print('N markers in adata:',adata_full.var.feature_name.nunique())

# %% [markdown]
# Remove endocrine pancreas cell types

# %%
# All cell types in pancreas
sorted(adata_full.obs.query('tissue_in_publication =="Pancreas"')['cell_type'].unique())

# %%
# Cell types to remove, subset adata to exclude them
ct_rm=['pancreatic A cell','pancreatic D cell',
       'pancreatic PP cell','type B pancreatic cell']
adata=adata_full[adata_full.obs.query('cell_type not in @ct_rm').index,:].copy()
adata.shape

# %% [markdown]
# Group adata by cell type, tissue (both as published - not ontology), and donor to make pseudobulk for correlation

# %%
# Make pb
x=adata.to_df()
group_cols=['tissue_in_publication','donor_id', 'free_annotation', ]
x[group_cols]=adata.obs[group_cols]
x=x.groupby(group_cols,observed=True).apply(lambda x:x.drop(group_cols,axis=1).mean(axis=0))

# %%
# Remove all 0 expressed genes
print('N all 0 genes:',(x==0).all().sum())
x=x.loc[:,(x>0).any()]
x.shape

# %%
# Find correlated genes
corrs=pd.DataFrame(np.corrcoef(x,rowvar=False),index=x.columns,columns=x.columns)
# Set self corr to 0
np.fill_diagonal(corrs.values,0)

# %% [markdown]
# Annotate which gene marks which ct

# %%
# Which gene marks which ct
gene_ct_anno=pd.DataFrame(index=genes,columns=genes_ct.keys())
for ct,gs in genes_ct.items():
    gene_ct_anno.loc[gs,ct]=True
gene_ct_anno.fillna(False,inplace=True)

# Stringt of all cts it marks
gene_ct_anno['marks']=gene_ct_anno.apply(lambda x:'_'.join(x.index[x]),axis=1)
display(gene_ct_anno['marks'].value_counts())

# Asign colors
palette=dict(zip(gene_ct_anno['marks'].value_counts().index,
                 ['#1f77b4','#ff7f0e','#bcbd22','#7f7f7f','#000000']))
anno=gene_ct_anno['marks'].map(palette)
# Add EIDs
eid_gs={g:adata.var.at[g,'feature_name'] for g in x.columns}
anno=anno[[eid_gs[g] for g in x.columns]]
anno.index=x.columns
anno.name='marks'

# %%
sb.clustermap(corrs,vmin=-1,vmax=1,cmap='coolwarm',
              xticklabels=False,yticklabels=False,col_colors=anno)

# %% [markdown]
# ### Correlations on endocrine cells

# %% [markdown]
# Keep only endocrine pancreas cell types

# %%
# Cell types to remove, subset adata to exclude them
ct_rm=['pancreatic A cell','pancreatic D cell',
       'pancreatic PP cell','type B pancreatic cell']
adata_endo=adata_full[adata_full.obs.query('cell_type in @ct_rm').index,:].copy()
adata_endo.shape

# %% [markdown]
# Group adata by cell type, tissue (both as published - not ontology), and donor to make pseudobulk for correlation

# %%
# Make pb
x_endo=adata_endo.to_df()
group_cols=['tissue_in_publication','donor_id', 'free_annotation', ]
x_endo[group_cols]=adata_endo.obs[group_cols]
x_endo=x_endo.groupby(group_cols,observed=True
                     ).apply(lambda x:x.drop(group_cols,axis=1).mean(axis=0))

# %%
# Remove all 0 expressed genes
print('N all 0 genes:',(x_endo==0).all().sum())
x_endo=x_endo.loc[:,(x_endo>0).any()]
x_endo.shape

# %%
# Find correlated genes
corrs_endo=pd.DataFrame(np.corrcoef(x_endo,rowvar=False),
                        index=x_endo.columns,columns=x_endo.columns)
np.fill_diagonal(corrs_endo.values,0)

# %% [markdown]
# Annotate which gene marks which ct

# %%
# Which gene marks which ct
gene_ct_anno=pd.DataFrame(index=genes,columns=genes_ct.keys())
for ct,gs in genes_ct.items():
    gene_ct_anno.loc[gs,ct]=True
gene_ct_anno.fillna(False,inplace=True)

# Stringt of all cts it marks
gene_ct_anno['marks']=gene_ct_anno.apply(lambda x:'_'.join(x.index[x]),axis=1)
display(gene_ct_anno['marks'].value_counts())

# Asign colors
palette=dict(zip(gene_ct_anno['marks'].value_counts().index,
                 ['#1f77b4','#ff7f0e','#bcbd22','#7f7f7f','#000000']))
anno=gene_ct_anno['marks'].map(palette)
# Add EIDs
eid_gs={g:adata.var.at[g,'feature_name'] for g in x_endo.columns}
anno=anno[[eid_gs[g] for g in x_endo.columns]]
anno.index=x_endo.columns
anno.name='marks'

# %%
sb.clustermap(corrs_endo,vmin=-1,vmax=1,cmap='coolwarm',
              xticklabels=False,yticklabels=False,col_colors=anno,row_colors=anno)

# %%
idx=np.triu_indices_from(corrs,k=1)
corrs_triu=pd.Series(corrs.values[idx],name='corr',
          index=['_'.join(sorted([corrs.columns[i],corrs.columns[j]])) for i,j in zip(*idx)])
idx=np.triu_indices_from(corrs_endo,k=1)
corrs_endo_triu=pd.Series(corrs_endo.values[idx],name='corr_endo',
          index=['_'.join(sorted([corrs_endo.columns[i],
                                  corrs_endo.columns[j]])) for i,j in zip(*idx)])

# %%
corrs_compare=pd.concat([corrs_triu,corrs_endo_triu],axis=1).dropna()

# %%
sb.scatterplot(x='corr',y='corr_endo',data=corrs_compare,s=1)

# %% [markdown]
# C: many genes loose correlation when doing it on non-endo cts rather than endo cts. But some genes retain the correlation.
#
# C: It seems that highly corr genes on non-endo are also highly corr in endo.

# %%
# N genes that have another gene pair with high corr
(corrs>0.7).any().sum()

# %%
# N high corr pairs
((corrs* np.tri(*corrs.shape))>0.7).sum().sum()

# %%
sb.clustermap(corrs>0.7,vmin=0,vmax=1,cmap='cividis',
              xticklabels=False,yticklabels=False,col_colors=anno,row_colors=anno)

# %% [markdown]
# C: Most high corrs are probably form a single cluster of genes

# %% [markdown]
# Expression in non-beta - maybe not expressed so cant se correl
# TODO

# %%
np.fill_diagonal(corrs.values,0)

# %%
plt.scatter(x['ENSG00000225972'],x['ENSG00000198886'])

# %% [markdown]
# C: In some cases would be better to do corr only on pbs that express both genes (e.g. at least 10% of max expression)). NO: gene pair bimodal behaviour

# %% [markdown]
# TODO: More analysis on how correlations look, maybe modelling of multiple genes (see bimodal example above)

# %% [markdown]
# ### Regression coefficients for correlated pairs

# %%
# Fill one half of matrix with 0 not to repeat pairs, the diagonal was filled in before
corr_pairs=np.where((corrs* np.tri(*corrs.shape))>0.7)
coefs={}
for i,j in zip(*corr_pairs):
    g1=corrs.index[i]
    g2=corrs.columns[j]
    lr=LinearRegression(normalize=False)
    lr.fit(x[g1].values.reshape(-1,1),x[g2].values.reshape(-1,1))
    # Save (x,y)=(intercept,coef)
    coefs[(g1,g2)]=(lr.intercept_[0],lr.coef_[0,0])

# %%
rcParams['figure.figsize']=(3,3)
plt.scatter(np.array(list(coefs.values()))[:,0],np.array(list(coefs.values()))[:,1],s=0.5)
plt.xlabel('intercept')
plt.ylabel('coef')

# %%
coefs_df=pd.concat([pd.DataFrame(list(coefs.keys()),columns=['gx','gy']),
pd.DataFrame(list(coefs.values()),columns=['intercept','coef'])],axis=1)

# %%
coefs_df.to_csv(path_genes+'coefs_hs.tsv',sep='\t',index=False)

# %%
