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
from collections import defaultdict
from scipy.io import mmread
from scipy.sparse import csr_matrix
import pickle as pkl

import gc

from matplotlib import rcParams
import matplotlib.pyplot as plt
import seaborn as sb

# %%
#path_data='/lustre/groups/ml01/workspace/karin.hrovatin/data/'
path_data='/net/bmc-lab6/data/lab/kellis/users/khrovati/data/'
path_sub=path_data+'datasets/d10_1038_s41586-022-04518-2/SCP1376/'
path_save=path_data+'cross_system_integration/adipose_sc_sn_updated/'

# %% [markdown]
# # Prepare adatas from mtx

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
# # Combien adatas for training

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
sc.pp.normalize_total(adata_sc, target_sum=1e4)
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
sc.pp.normalize_total(adata_sn, target_sum=1e4)
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

# %% [markdown]
# # Moran's I for eval
# Find genes that would be appropriate for computing Moran's I on for evaluation in every sample-cell type group (of appropriate size) by computing Moran's I on per-sample non integrated data. This I can then also be used as a reference later on to compute relative preservation of Moran's I.
#

# %%
#adata=sc.read(path_save+'adiposeHsSAT_sc_sn.h5ad')

# %%
# Potential groups to compute Moran's I on (batch-system and group)
pd.crosstab(adata.obs.donor_id,adata.obs.cluster)

# %%
# Filtered groups based on N cells
groups=adata.obs.groupby(['cluster','system','donor_id']).size()
groups=groups[groups>=500]
display(groups)
print('N cell types',groups.index.get_level_values('cluster').nunique())

# %%
# Compute Moran's I per group
data=[]
for group in groups.index:
    # Group adata
    print(group)
    adata_sub=adata[
        (adata.obs.cluster==group[0]).values&\
        (adata.obs.system==group[1]).values&\
        (adata.obs.donor_id==group[2]).values,:].copy()
    # Remove lowly expr genes before Moran's I computation as they will be less likely relevant
    # As this is done per small cell group within sample+cell type there is not many genes (200-500)
    # so all can be used for Moran's I computation
    sc.pp.filter_genes(adata_sub, min_cells=adata_sub.shape[0]*0.1) 
    # Compute embedding of group
    sc.pp.pca(adata_sub, n_comps=15)
    sc.pp.neighbors(adata_sub, n_pcs=15)
    # Compute I
    morans_i=sc.metrics._morans_i._morans_i(
        g=adata_sub.obsp['connectivities'],
        vals=adata_sub.X.T)
    # Save data
    morans_i=pd.DataFrame({'morans_i':morans_i},index=adata_sub.var_names)
    morans_i['group']=group[0]
    morans_i['system']=group[1]
    morans_i['batch']=group[2]
    data.append(morans_i)
data=pd.concat(data,axis=0)

# %%
# Moran's I distn accross groups
sb.catplot(x='batch',y='morans_i',hue='system',row='group',data=data,kind='violin',
           inner=None,height=3,aspect=2)

# %%
# Moran's I thr
thr_mi=0.2

# %%
# N genes per group at certain thr
data.groupby(['group','system','batch']).apply(lambda x: (x['morans_i']>=thr_mi).sum())

# %%
# N genes vs N cells in group
rcParams['figure.figsize']=(4,4)
sb.scatterplot(x='N cells',y='N genes',hue='level_0',style='system',
           data=pd.concat(
    [data.groupby(['group','system','batch']).apply(lambda x: (x['morans_i']>=thr_mi).sum()).rename('N genes'),
    groups.rename('N cells')],axis=1).reset_index())
plt.legend(bbox_to_anchor=(1,1))
plt.xscale('log')

# %% [markdown]
# C: Thr of 0.2 seems to separate in general approximately between highly and lowly variable genes and has at least some genes for every group and not too many in any of the groups.
#
# C: There is no clear bias between N cells in group and N genes, although such bias was observed within a cell type accross genes. Likely due to sample/cell type specific effects.
#
# C: Selected genes may not be diverse though - they may capture the same pattern and maybe more subtle patterns are at lower Moran's I.

# %%
# Prepare selected genes for saving (fileterd genes&I per group)
selected=list()
for group,data_sub in data.groupby(['group','system','batch']):
    group=dict(zip(['group','system','batch'],group))
    group['genes']=data_sub.query('morans_i>=@thr_mi')['morans_i']
    selected.append(group)

# %%
# Save
pkl.dump(selected,open(path_save+'adiposeHsSAT_sc_sn_moransiGenes.pkl','wb'))

# %%
