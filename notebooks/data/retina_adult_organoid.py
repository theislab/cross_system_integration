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
#     display_name: csi
#     language: python
#     name: csi
# ---

# %%
import scanpy as sc
import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import euclidean_distances
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

import pickle as pkl

from matplotlib import rcParams
import matplotlib.pyplot as plt
import seaborn as sb

import scib_metrics as sm
import sys
import os
sys.path.append('/'.join(os.getcwd().split('/')[:-1]+['eval','cleaned','']))
from metrics import ilisi,asw_batch

# %%
path='/net/bmc-lab6/data/lab/kellis/users/khrovati/data/'
path_data=path+'datasets/d10_1016_j_cell_2020_08_013/'
path_train=path+'cross_system_integration/retina_adult_organoid/'

# %%
adata=sc.concat([
    sc.concat(
        [sc.read(path_data+'periphery.h5ad'),sc.read(path_data+'fovea.h5ad')],
        label='region',keys =['periphery','fovea'],index_unique='-',join='outer'
    ),sc.read(path_data+'organoid.h5ad')],
    label='material',keys =['adult','organoid'],index_unique='-',join='outer'
)

# %%
adata.X=adata.raw.X

# %% [markdown]
# C: Unclear what batch may be

# %%
adata.obs.filtered_out_cells.sum()

# %%
pd.crosstab(adata.obs.cell_type,adata.obs.material)

# %%
adata.obs.cell_type.isna().sum()

# %%
sc.pl.umap(adata,color=['cell_type','material','region','sample_id'])

# %% [markdown]
# ## Prepare data for training

# %%
adata_sub=adata.copy()

# %%
# Remove unnanotated cells
adata_sub=adata_sub[adata_sub.obs.cell_type!='native cell',:]

# %%
# Keep not too lowly expressed genes as intersection of the two systems
adata_sub=adata_sub[:,
                    np.array((adata_sub[adata_sub.obs.material=="adult",:].X>0).sum(axis=0)>20).ravel()&\
                    np.array((adata_sub[adata_sub.obs.material=="organoid",:].X>0).sum(axis=0)>20).ravel()
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
    adata_sub[adata_sub.obs.material=="adult",:], 
    n_top_genes=4000, flavor='cell_ranger', inplace=False, batch_key='sample_id').query('highly_variable==True').index)&\
set(sc.pp.highly_variable_genes(
    adata_sub[adata_sub.obs.material=="organoid",:], 
    n_top_genes=4000, flavor='cell_ranger', inplace=False, batch_key='sample_id').query('highly_variable==True').index)
print(len(hvgs))

# %%
adata_sub=adata_sub[:,list(hvgs)]

# %%
adata_sub.obs['system']=adata_sub.obs['material'].map({"organoid":0,'adult':1})

# %%
del adata_sub.uns
del adata_sub.obsm
adata_sub.obs=adata_sub.obs[[
    'cell_type','cell_type_group','author_cell_type','condition', 
    'dataset','sample_id','ega_sample_alias', 'hca_data_portal_donor_uuid', 'hca_data_portal_cellsuspension_uuid', 
    'region',  'material', 'system']]

# %%
adata_sub.layers['counts']=adata[adata_sub.obs_names,adata_sub.var_names].X.copy()

# %%
pd.crosstab(adata_sub.obs.cell_type,adata_sub.obs.system)

# %%
# N samples and cells per system
display(adata_sub.obs.groupby('system')['sample_id'].nunique())
display(adata_sub.obs.groupby('system').size())

# %% [markdown]
# Add PCA for scGLUE

# %%
# PCA per system
n_pcs=15
X_pca_system=[]
for system in adata_sub.obs.system.unique():
    adata_temp=adata_sub[adata_sub.obs.system==system,:].copy()
    sc.pp.scale(adata_temp)
    sc.pp.pca(adata_temp, n_comps=n_pcs)
    X_pca_system.append(pd.DataFrame(adata_temp.obsm['X_pca'],index=adata_temp.obs_names))
del adata_temp
X_pca_system=pd.concat(X_pca_system)
adata_sub.obsm['X_pca_system']=X_pca_system.loc[adata_sub.obs_names,:].values

# %% [markdown]
# ### Save

# %%
adata_sub

# %%
adata_sub.write(path_train+'combined_HVG.h5ad')

# %%
#path_train='/om2/user/khrovati/data/cross_system_integration/retina_adult_organoid/'
#adata_sub=sc.read(path_train+'combined_HVG.h5ad')

# %% [markdown]
# # Non-integrated embedding

# %%
# Non-integrated embedding
n_pcs=15
cells_eval=np.random.RandomState(seed=0).permutation(adata_sub.obs_names)[:100000]
adata_temp=adata_sub[cells_eval,:].copy()
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
adata_embed.write(path_train+'combined_HVG_embed.h5ad')

# %% [markdown]
# # Integration metrics on non-integrated data

# %%
# Reload
#adata_embed=sc.read(path_train+'combined_HVG_embed.h5ad')

# %%
ilisi_system, ilisi_system_macro, ilisi_system_data_label=ilisi(
        X=adata_embed.obsp['distances'],
        batches=adata_embed.obs['system'], 
        labels=adata_embed.obs['cell_type'])

# %%
ilisi_system, ilisi_system_macro, ilisi_system_data_label

# %%
sm.graph_connectivity( X=adata_embed.obsp['distances'],
        labels=adata_embed.obs['system'])

# %%
rcParams['figure.figsize']=(6,2)
_=plt.boxplot(adata_embed.X)
plt.ylabel('PCA value')
plt.xlabel('PCs')

# %%
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
pkl.dump({'asw_batch':asws},open(path_train+'combined_HVG_embed_integrationMetrics.pkl','wb'))

# %% [markdown]
# # Moran's I for eval

# %%
#adata=adata_sub

# %%
#adata=sc.read(path_train+'combined_HVG.h5ad')

# %%
# Potential groups to compute Moran's I on (batch-system and group)
cols=pd.get_option('display.max_columns')
pd.set_option('display.max_columns', 30)
display(pd.crosstab(adata.obs.sample_id,adata.obs.cell_type))
pd.set_option('display.max_columns', cols)

# %%
# Filtered groups based on N cells
groups=adata.obs.groupby(['cell_type','system','sample_id']).size()
groups=groups[groups>=500]
rows=pd.get_option('display.max_rows')
pd.set_option('display.max_rows', 100)
display(groups)
pd.set_option('display.max_rows', rows)
print('N cell types',groups.index.get_level_values('cell_type').nunique())

# %%
# Compute Moran's I per group
data=[]
for group in groups.index:
    # Group adata
    print(group)
    adata_sub=adata[
        (adata.obs.cell_type==group[0]).values&\
        (adata.obs.system==group[1]).values&\
        (adata.obs.sample_id==group[2]).values,:].copy()
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
# I thr
thr_mi=0.15

# %%
# N genes per group at certain thr
rows=pd.get_option('display.max_rows')
pd.set_option('display.max_rows', 100)
display(data.groupby(['group','system','batch']).apply(lambda x: (x['morans_i']>=thr_mi).sum()))
pd.set_option('display.max_rows', rows)

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
# C: Thr of 0.15 has at least some genes for every group and not too many in any of the groups. Some groups would need lower/higher thr potentially.
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
pkl.dump(selected,open(path_train+'combined_HVG_moransiGenes.pkl','wb'))

# %%
path_train

# %% [markdown]
# # Batch effects within and between systems

# %%
#adata=sc.read(path_train+'combined_HVG.h5ad')

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
pca[['system','batch','group']]=adata.obs[['system', 'sample_id', 'cell_type']]
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
pkl.dump(distances,open(path_train+'combined_HVG_PcaSysBatchDist.pkl','wb'))

# %%
# Reload distances
#distances=pkl.load(open(path_train+'combined_HVG_PcaSysBatchDist.pkl','rb'))

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
           data=plot.reset_index(drop=True),kind='violin',
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
signif.to_csv(path_train+'combined_HVG_PcaSysBatchDist_Signif.tsv',sep='\t',index=False)

# %% [markdown]
# AUC based on Mann-Whitney U (using within system as one group and between system as the other)

# %%
auc_roc={}
for ct,dat in distances.items():
    x_within=np.concatenate([dat['s0'],dat['s1']])
    x_between=dat['s0s1']
    auc_roc[ct]=mannwhitneyu(x_between,x_within)[0]/(x_within.shape[0]*x_between.shape[0])
print(auc_roc)

# %%
