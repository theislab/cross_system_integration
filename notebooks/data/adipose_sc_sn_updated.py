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

from sklearn.metrics.pairwise import euclidean_distances
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

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

# %% [markdown]
# Add PCA for scGLUE 

# %%
# PCA and clusters per system
n_pcs=15
X_pca_system=[]
for system in adata.obs.system.unique():
    adata_sub=adata[adata.obs.system==system,:].copy()
    sc.pp.scale(adata_sub)
    sc.pp.pca(adata_sub, n_comps=n_pcs)
    X_pca_system.append(pd.DataFrame(adata_sub.obsm['X_pca'],index=adata_sub.obs_names))
del adata_sub
X_pca_system=pd.concat(X_pca_system)
adata.obsm['X_pca_system']=X_pca_system.loc[adata.obs_names,:].values

# %% [markdown]
# ### Save

# %%
adata

# %%
adata.write(path_save+'adiposeHsSAT_sc_sn.h5ad')

# %%
#adata=sc.read(path_save+'adiposeHsSAT_sc_sn.h5ad')

# %% [markdown]
# # Non-integrated embedding

# %%
# Non-integrated embedding
n_pcs=15
cells_eval=np.random.RandomState(seed=0).permutation(adata.obs_names)[:100000]
adata_temp=adata[cells_eval,:].copy()
sc.pp.scale(adata_temp)
sc.pp.pca(adata_temp, n_comps=n_pcs)
sc.pp.neighbors(adata_temp, use_rep='X_pca')
sc.tl.umap(adata_temp)

# %%
# Slimmed down data for saving
adata_embed=sc.AnnData(adata_temp.obsm['X_pca'],obs=adata_temp.obs)
for k in ['pca','neighbors','umap']:
    adata_embed.uns[k]=adata_temp.uns[k]
adata_embed.obsm['X_umap']=adata_temp.obsm['X_umap']
for k in ['distances', 'connectivities']:
    adata_embed.obsp[k]=adata_temp.obsp[k]
display(adata_embed)

# %%
# Save
adata_embed.write(path_save+'adiposeHsSAT_sc_sn_embed.h5ad')

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
    group['genes']=(data_sub.query('morans_i>=@thr_mi')['morans_i']+1)/2
    selected.append(group)

# %%
# Save
pkl.dump(selected,open(path_save+'adiposeHsSAT_sc_sn_moransiGenes.pkl','wb'))

# %% [markdown]
# # Batch effects within and between systems

# %%
#adata=sc.read(path_save+'adiposeHsSAT_sc_sn.h5ad')

# %%
# Compute PCA on the whole data
adata_scl=adata.copy()
sc.pp.scale(adata_scl)
n_pcs=15
sc.pp.pca(adata_scl, n_comps=n_pcs)
pca=pd.DataFrame(adata_scl.obsm['X_pca'],index=adata_scl.obs_names)
del adata_scl

# %%
# Average PCA accross system-batch-group pseudobulks. 
# Only use pseudobulks with at least 50 cells
# Only use cell types with at least 3 samples per system
pca[['system','batch','group']]=adata.obs[['system', 'donor_id', 'cluster']]
pca_pb=pca.groupby(['system','batch','group'])
pca_mean=pca_pb.mean()
pb_size=pca_pb.size()
# Remove samples with too little cells
filtered_pb=pb_size.index[pb_size>=50]
# Get pbs/cts where both systems have enough samples
n_samples_system=filtered_pb.to_frame().rename({'group':'group_col'},axis=1).groupby(
    'group_col',observed=True)['system'].value_counts().rename('n_samples').reset_index()
cts=set(n_samples_system.query('system==0 & n_samples>=3').group_col)&\
    set(n_samples_system.query('system==1 & n_samples>=3').group_col)
filtered_pb=filtered_pb[filtered_pb.get_level_values(2).isin(cts)]
pca_mean=pca_mean.loc[filtered_pb,:]

# %%
# Compute per-ct distances of samples within and between systems
distances={}
for ct in cts:
    pca_s0=pca_mean[(pca_mean.index.get_level_values(0)==0) &
                    (pca_mean.index.get_level_values(2)==ct)]
    pca_s1=pca_mean[(pca_mean.index.get_level_values(0)==1) &
                    (pca_mean.index.get_level_values(2)==ct)]
    d_s0=euclidean_distances(pca_s0)[np.triu_indices(pca_s0.shape[0],k=1)]
    d_s1=euclidean_distances(pca_s1)[np.triu_indices(pca_s1.shape[0],k=1)]
    d_s0s1=euclidean_distances(pca_s0,pca_s1).ravel()
    distances[ct]={'s0':d_s0,'s1':d_s1,'s0s1':d_s0s1}

# %%
# Save distances
pkl.dump(distances,open(path_save+'adiposeHsSAT_sc_sn_PcaSysBatchDist.pkl','wb'))

# %%
#distances=pkl.load(open(path_save+'adiposeHsSAT_sc_sn_PcaSysBatchDist.pkl','rb'))

# %%
# Prepare df for plotting
plot=[]
for ct,dat in distances.items():
    for comparison,dist in dat.items():
        dist=pd.DataFrame(dist,columns=['dist'])
        dist['group']=ct
        dist['comparison']=comparison
        plot.append(dist)
plot=pd.concat(plot)

# %%
# Plot distances
sb.catplot(x='comparison',y='dist',col='group',
           data=plot.reset_index(drop=True),kind='swarm',
           sharey=False, height=2.5,aspect=1 )

# %% [markdown]
# Evaluate statisticsal significance

# %%
# Compute significance of differences within and accross systems
signif=[]
for ct,dat in distances.items():
    for ref in ['s0','s1']:
        u,p=mannwhitneyu( dat[ref],dat['s0s1'],alternative='less')
        signif.append(dict( cell_type=ct,system=ref, u=u,pval=p,
                           n_system=dat[ref].shape[0],n_crossystem=dat['s0s1'].shape[0]))
signif=pd.DataFrame(signif)
signif['padj']=multipletests(signif['pval'],method='fdr_bh')[1]

# %%
signif

# %%
# Save signif
signif.to_csv(path_save+'adiposeHsSAT_sc_sn_PcaSysBatchDist_Signif.tsv',sep='\t',index=False)

# %%
