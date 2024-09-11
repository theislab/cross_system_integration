# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.3
#   kernelspec:
#     display_name: csi
#     language: python
#     name: csi
# ---

# %%
# %load_ext autoreload
# %autoreload 2

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

import scib_metrics as sm
import sys
import os
sys.path.append('/'.join(os.getcwd().split('/')[:-1]+['eval','cleaned','']))
from metrics import ilisi, asw_batch


# %%
path_data = '/home/moinfar/data/human_retina_atlas/'
path_sc_data = path_data + 'human_retina_atlas_scrna_all.h5ad'
path_sn_data = path_data + 'human_retina_atlas_snrna_all.h5ad'
path_save = path_data + 'human_retina_atlas_sc_sn_hvg.h5ad'

# %% [markdown]
# # Load data

# %% [markdown]
# # Combien adatas for training

# %% [markdown]
# Sn need to subset to same type of fat as sc as else there are a lot of unmatched cts. In cs dont need to remove cancer partients as this does not seem to affect fat.

# %%
# PP sc
adata_sc=sc.read(path_sc_data)
adata_sc.X = adata_sc.raw.X
del adata_sc.raw
adata_sc.obs['system'] = 0
adata_sc

# %%
# Subset to expr genes and normalise
adata_sc = adata_sc[:,np.array((adata_sc.X>0).sum(axis=0)>20).ravel()].copy()
adata_sc.layers['counts']=adata_sc.X.copy()
sc.pp.normalize_total(adata_sc, target_sum=1e4)
sc.pp.log1p(adata_sc)
# HVG
sc.pp.highly_variable_genes(
     adata=adata_sc, n_top_genes=3000, flavor='cell_ranger', batch_key='donor_id', subset=True)

adata_sc

# %%
adata_sc.write(path_data + '_tmp_sc_hvg.h5ad')

# %%
adata_sc = sc.read(path_data + '_tmp_sc_hvg.h5ad')
adata_sc

# %%

# %%
# PP sn
adata_sn=sc.read(path_sn_data)
adata_sn.X = adata_sn.raw.X
del adata_sn.raw
adata_sn.obs['system'] = 1
adata_sn

# %%
# Subset to expr genes and normalise
adata_sn=adata_sn[:,np.array((adata_sn.X>0).sum(axis=0)>20).ravel()].copy()
adata_sn.layers['counts']=adata_sn.X.copy()
sc.pp.normalize_total(adata_sn, target_sum=1e4)
sc.pp.log1p(adata_sn)
# HVG
sc.pp.highly_variable_genes(
     adata=adata_sn, n_top_genes=3000, flavor='cell_ranger', batch_key='donor_id', subset=True)

adata_sn

# %%
adata_sn.write(path_data + '_tmp_sn_hvg.h5ad')

# %%
adata_sn = sc.read(path_data + '_tmp_sn_hvg.h5ad')
adata_sn

# %%

# %%
# Shared HVGs
shared_hvgs=list(set(adata_sc.var_names) & set(adata_sn.var_names))
len(shared_hvgs)

# %%

# %% [markdown]
# Match cell type names

# %%
adata_sc.obs.cell_type.value_counts(), adata_sn.obs.cell_type.value_counts()

# %%
assert len(set(adata_sn.obs.donor_id.unique()).intersection(set(adata_sc.obs.donor_id.unique()))) == 0

# %% [markdown]
# Joint adata

# %%
# Subset to shraed HVGs and concat
adata=sc.concat([adata_sc[:,shared_hvgs], adata_sn[:,shared_hvgs]],
                join='outer',
                index_unique='_', keys=['sc','sn'])
adata

# %%
pd.crosstab(adata.obs.cell_type, adata.obs.system)

# %%
# N samples and cells per system
display(adata.obs.groupby('system')['donor_id'].nunique())
display(adata.obs.groupby('system').size())

# %%

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
adata.write(path_save)

# %%

# %% [markdown]
# # Non-integrated embedding

# %%
adata = sc.read(path_save)
path_save = path_data + 'embed_pca'

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
adata_embed.write(path_save + 'hrca_sc_sn_pca_embed.h5ad')

# %% [markdown]
# # Integration metrics on non-integrated data

# %%
# Reload
adata_embed=sc.read(path_save + 'hrca_sc_sn_pca_embed.h5ad')

# %%
# Check ranges of individual PCs
rcParams['figure.figsize']=(6,2)
_=plt.boxplot(adata_embed.X)
plt.ylabel('PCA value')
plt.xlabel('PCs')

# %%
# Compute ASW
asw, asw_macro, asw_data_label=asw_batch(
    X=adata_embed.X,
    batches=adata_embed.obs['system'], 
    labels=adata_embed.obs['cell_type'])

# %%
asws={
    'asw_micro':asw,
    'asw_macro':asw_macro,
    'asw_data_label':asw_data_label
}
for k,v in asws.items():
    print(k)
    print(v)
    print('\n')

# %%
pkl.dump({'asw_batch':asws},open(path_save+'hrca_sc_sn_pca_embed_integrationMetrics.pkl','wb'))

# %% [markdown]
# # Moran's I for eval
# Find genes that would be appropriate for computing Moran's I on for evaluation in every sample-cell type group (of appropriate size) by computing Moran's I on per-sample non integrated data. This can then also be used as a reference later on to compute relative preservation of Moran's I.
#

# %%
adata_path = path_data + 'human_retina_atlas_sc_sn_hvg.h5ad'
adata=sc.read(adata_path)
adata

# %%
# Potential groups to compute Moran's I on (batch-system and group)
pd.crosstab(adata.obs.donor_id, adata.obs.cell_type)

# %%
# Filtered groups based on N cells
groups=adata.obs.groupby(['cell_type', 'system', 'donor_id']).size()
groups=groups[groups>=500]
display(groups)
print('N cell types', groups.index.get_level_values('cell_type').nunique())

# %%
# Compute Moran's I per group
data=[]
for group in groups.index:
    # Group adata
    print(group)
    adata_sub=adata[
        (adata.obs.cell_type==group[0]).values&\
        (adata.obs.system==group[1]).values&\
        (adata.obs.donor_id==group[2]).values,:].copy()
    # Remove lowly expr genes before Moran's I computation as they will be less likely relevant
    # As this is done per small cell group within sample+cell type and HVGs there is not many genes (200-500)
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
           inner=None,height=3,aspect=5)

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
# C: There is no clear bias between N cells in group and N genes.
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
pkl.dump(selected,open(path_data+'human_retina_atlas_sc_sn_hvg_moransiGenes.pkl','wb'))

# %% [markdown]
# # Batch effects within and between systems

# %%
adata_path = path_data + 'human_retina_atlas_sc_sn_hvg.h5ad'
adata=sc.read(adata_path)
adata

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
pca[['system','batch','group']]=adata.obs[['system', 'donor_id', 'cell_type']]
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
# pkl.dump(distances,open(path_save+'adiposeHsSAT_sc_sn_PcaSysBatchDist.pkl','wb'))

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
signif.to_csv(path_data + 'human_retina_atlas_sc_sn_hvg_PcaSysBatchDist_Signif.tsv',sep='\t',index=False)

# %%

# %%
