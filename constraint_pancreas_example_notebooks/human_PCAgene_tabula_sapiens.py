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
# Tryu to use tabula sapiens for gene embedding calculation (expression accross cts and donors).

# %%
import scanpy as sc
import pandas as pd
import numpy as np
import pickle as pkl
from sklearn.preprocessing import StandardScaler, scale, minmax_scale
from sklearn.decomposition import PCA
import gc
import random

import seaborn as sb
import matplotlib.pyplot as plt
from matplotlib import rcParams

# %%
path_genes='/lustre/groups/ml01/workspace/karin.hrovatin/data/cross_species_prediction/pancreas_ct_example_constraint/'
path_data='/lustre/groups/ml01/workspace/karin.hrovatin/data/datasets/tabula_sapiens/'

# %%
markers_ct=pkl.load(open(path_genes+'endo_markers_set_hs.pkl','rb'))
markers=set([g for gs in markers_ct.values() for g in gs])

# %% [markdown]
# Load tabula sapiens

# %%
adata_full=sc.read(path_data+'local.h5ad')

# %%
adata_full

# %%
adata_full.uns['X_normalization']

# %%
# The above takes too much memory
#adata=adata_full.copy()
adata=adata_full

# %% [markdown]
# Find HVG data

# %%
sc.pp.highly_variable_genes(adata, n_top_genes=2000)

# %% [markdown]
# Merge HVG and markers

# %%
markers_eid=set(adata.var.query('feature_name in @markers').index)
print('N selected markers:',len(markers_eid))

# %%
hvgs_eid=set(adata.var_names[adata.var['highly_variable']])
print('N selected HVGs:',len(hvgs_eid))

# %%
genes_eid=list(markers_eid|hvgs_eid)
print('N selected genes:',len(genes_eid))

# %% [markdown]
# Subset to endocrine marker genes+HVGs

# %%
# Subset adata to only marker genes
adata=adata[:,genes_eid].copy()

# %%
gc.collect()

# %% [markdown]
# For now gene coexpression information from all cell types will be used

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
print('N all-zero genes:',(x==0).all().sum())
x=x.loc[:,(x>0).any()]
x.shape

# %%
x_pb=x.copy()

# %% [markdown]
# Mean and variance of unscaled cells

# %%
plt.scatter(x.mean(axis=1),x.std(axis=1),s=1)
plt.ylabel('std cells')
plt.xlabel('mean cells')

# %%
plt.hist(x.mean(axis=1),bins=20)
plt.xlabel('mean cells')

# %%
plt.hist(x.std(axis=1),bins=20)
plt.xlabel('std cells')

# %% [markdown]
# Mean and variance of unscaled cells but with gene scaling

# %%
plt.hist(scale(x).mean(axis=1),bins=20)
plt.xlabel('mean cells after gene scaling')

# %%
plt.hist(scale(x).std(axis=1),bins=20)
plt.xlabel('std cells after gene scaling')

# %% [markdown]
# C: After gene scaling there seems to be some outliers, but relatively less variability accross cells.

# %% [markdown]
# Standardise expression cell-wise (features)

# %%
# Transpose to genes*cell features
x=x.T
print(x.shape)

# %%
# Save index and columns
x_genes=x.index
x_cts=x.columns

# %%
scaler=StandardScaler().fit(x)
x=scaler.transform(x)

# %% [markdown]
# PCA

# %%
pca=PCA(n_components=50).fit(x)

# %%
plt.scatter(range(len(pca.explained_variance_)),pca.explained_variance_)

# %% [markdown]
# PC1 explains most variance, thus remove from plot to analyse also other PCs

# %%
# As above but without PC1
plt.scatter(range(1,len(pca.explained_variance_)),pca.explained_variance_[1:])

# %% [markdown]
# C: Odd that single PC explains so much variance. Try to figure out what it may be connected to (anything technical?)

# %% [markdown]
# Loadings distn for first few components

# %%
# Loadings for 1st PC
plt.hist(pca.components_[0],bins=20)

# %%
# Loadings for 2nd PC
plt.hist(pca.components_[1],bins=20)

# %%
# Loadings for 3th PC
plt.hist(pca.components_[2],bins=20)

# %%
# PC1 vs PC2 loadings
plt.scatter(pca.components_[0],pca.components_[1])

# %% [markdown]
# Visualise gene embedding

# %%
adata_genes=sc.AnnData(x)
adata_genes.obsm['X_pca']=pca.transform(x)
adata_genes

# %%
adata_genes.obs_names=x_genes
adata_genes.var_names=[i[2] for i in x_cts]

# %%
sc.pl.embedding(adata_genes,'X_pca')

# %%
n_pcs=20
sc.pp.neighbors(adata_genes,n_pcs=n_pcs)
sc.tl.umap(adata_genes)

# %%
sc.pl.umap(adata_genes)

# %%
adata_genes.obs['n_cts']=(adata_genes.X>0).sum(axis=1)

# %%
sc.pl.umap(adata_genes,color='n_cts')

# %%
sc.pl.embedding(adata_genes,'X_pca',color='n_cts')

# %% [markdown]
# C: N cell types expressing a gene affects the PCA strongly (1st component which explains most var) since genes were not scaled

# %%
# N genes expressed in ct
adata_genes.var['n_genes']=(adata_genes.X>0).sum(axis=0)

# %%
plt.scatter(adata_genes.var['n_genes'],pca.components_[0])

# %%
plt.scatter(adata_genes.var['n_genes'],pca.components_[1])

# %%
plt.scatter(adata_genes.var['n_genes'],pca.components_[2])

# %% [markdown]
# Where are markers in the embedding?

# %%
for ct,gs in markers_ct.items():
    eids=set(adata.var.query('feature_name in @gs').index)
    adata_genes.obs['marker_'+ct]=[int(o in eids) for o in adata_genes.obs_names]

# %%
sc.pl.umap(adata_genes,color=[c for c in adata_genes.obs.columns if 'marker_' in c],s=100)

# %% [markdown]
# C: Markers do not have very specific location, much seems to depend on N cts they are expressed in

# %% [markdown]
# Check if embedings of markers cluster together

# %%
marker_anno=pd.Series(index=adata_genes.obs_names)
colors={'beta':'r','alpha':'y','gamma':'g','delta':'b'}
for ct_col in [col for col in adata_genes.obs.columns if 'marker_' in col]:
    eids=adata_genes.obs_names[adata_genes.obs[ct_col].astype('bool')]
    marker_anno.loc[eids]=colors[ct_col.split('_')[1]]
marker_anno.fillna('lightgray',inplace=True)
sb.clustermap(pd.DataFrame(adata_genes.obsm['X_pca'][:,:n_pcs],
                           index=adata_genes.obs_names),yticklabels=False,
             row_colors=marker_anno)


# %% [markdown]
# C: It seems that most genes do not show much variability? Markers show more...

# %%
eids_plot=marker_anno.index[marker_anno!='lightgray']
sb.clustermap(pd.DataFrame(adata_genes[eids_plot,:].obsm['X_pca'][:,:n_pcs],
                           index=eids_plot),yticklabels=False,
             row_colors=marker_anno[eids_plot])

# %%
eids_plot=marker_anno.index[marker_anno!='lightgray']
sb.clustermap(pd.DataFrame(minmax_scale(adata_genes[eids_plot,:].obsm['X_pca'][:,:n_pcs]),
                           index=eids_plot),yticklabels=False,
             row_colors=marker_anno[eids_plot])

# %%
eids_plot=marker_anno.index[marker_anno!='lightgray']
sb.clustermap(pd.DataFrame(minmax_scale(adata_genes[eids_plot,:].obsm['X_pca'][:,:n_pcs],axis=1),
                           index=eids_plot),yticklabels=False,
             row_colors=marker_anno[eids_plot])

# %% [markdown]
# C: It seems that gene embedding does not capture the markers well. Could be that markers are non-specific for other cts, but in either case probably not great for decoding.

# %% [markdown]
# Check if plotting just accross pseudobulks also has such poor structure or sth went wrong with PCA

# %%
sb.clustermap(x_pb.T,yticklabels=False,xticklabels=False,
             row_colors=marker_anno)

# %% [markdown]
# C: Seems that selected HVGs are lowly expresssed mainly, potentially not useful

# %%
# Scaling cells
sb.clustermap(pd.DataFrame(minmax_scale(x_pb.T),index=x_pb.columns),
              yticklabels=False,xticklabels=False,
             row_colors=marker_anno)

# %%
# Scaling genes
sb.clustermap(pd.DataFrame(minmax_scale(x_pb.T,axis=1),index=x_pb.columns),
              yticklabels=False,xticklabels=False,
             row_colors=marker_anno)

# %%
# Scaling genes
sb.clustermap(pd.DataFrame(minmax_scale(x_pb.T,axis=1),
                           index=x_pb.columns,columns=x_pb.index).loc[eids_plot,:],
              yticklabels=False,
             row_colors=marker_anno[eids_plot])

# %% [markdown]
# C: Most of the markers are expressed accross all cts and donors?

# %%
plt.hist(x_pb[adata.var.query('feature_name=="INS"').index[0]],bins=100)
plt.yscale('log')

# %%
expr_gene=x_pb[adata.var.query('feature_name=="INS"').index[0]]
expr_gene[expr_gene>9]

# %% [markdown]
# C: Ambience is not reason as e.g. INS looks ok.

# %% [markdown]
# Potential problems with embedding: 
# - Low expression of some genes in many cell types (although most markers seem not to be lowly expressed)
# - Hard to capture patterns?

# %% [markdown]
# How do pseudobulk tissues and donors look like; can anything be merged instead of dim reduction

# %%
adata_pb=sc.AnnData(x_pb)
obs=adata_pb.obs_names.to_frame()
for col in obs.columns:
    adata_pb.obs[col]=obs[col]
adata_pb.obs_names=[str(i) for i in range(adata_pb.shape[0])]

# %%
sc.pp.scale(adata_pb)
sc.pp.pca(adata_pb,10)
sc.pp.neighbors(adata_pb,n_pcs=10)
sc.tl.umap(adata_pb)

# %%
adata_pb

# %%
adata_pb.uns['free_annotation_colors']=[]
for i in range(adata_pb.obs.free_annotation.nunique()):
    adata_pb.uns['free_annotation_colors'].append(
        "#"+''.join([random.choice('ABCDEF0123456789') for i in range(6)]))
sc.pl.umap(adata_pb, color=['tissue_in_publication', 'donor_id', 'free_annotation'],
           wspace=0.7)

# %%
adata.obs.cell_type.nunique()

# %% [markdown]
# Save

# %%
pd.DataFrame(adata_genes.obsm['X_pca'][:,:n_pcs], index=adata_genes.obs_names
            ).to_csv(path_genes+'pca_hs.tsv',sep='\t')

# %% [markdown]
# PCA not going to work as decomposes var of whole matrix and most of var is in genes not interesting - overall, batch effect. Non linear model needed.

# %%
