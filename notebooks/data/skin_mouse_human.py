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
import scanpy as sc
import pandas as pd
import numpy as np
from collections import defaultdict
from scipy.io import mmread
from scipy.sparse import csr_matrix
import pickle as pkl
from pathlib import Path

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
from metrics import ilisi,asw_batch

# %%
path_data='/home/moinfar/data/skin_mouse_human/'
path_mm=path_data+'GSE222250.h5ad'
path_hs=path_data+'HRA004936.h5ad'
path_genes=path_data
path_save=Path(path_data+'processed/')
path_save.mkdir(parents=True, exist_ok=True)

# %%
# Orthologues
orthology_info=pd.read_table(path_genes+'orthologues_ORGmus_musculus_ORG2homo_sapiens_V109.tsv'
                         ).rename(
    {'Gene name':'gs_mm','Human gene name':'gs_hs',
     'Gene stable ID':'eid_mm','Human gene stable ID':'eid_hs'},axis=1)
orthology_info[:3]

# %% [markdown]
# # One-to-one orthologues

# %% [markdown]
# ## Prepare data for integration

# %%
# One to one orthologues - dont have same mm/hs gene in the table 2x
oto_orthologues=orthology_info[~orthology_info.duplicated('eid_mm',keep=False).values & 
               ~orthology_info.duplicated('eid_hs',keep=False).values]

# %%
oto_orthologues.shape

# %% [markdown]
# ### Mouse

# %%
adata_mm=sc.read(path_mm)

# %%
adata_mm

# %%
# TODO: check what to remove

# %%
# Add raw expression to X, remove lowly expr genes, and normalise
adata_mm_raw=adata_mm.raw.to_adata()
adata_mm=adata_mm_raw.copy()
adata_mm.layers['counts']=adata_mm_raw[adata_mm.obs_names,adata_mm.var_names].X.copy()
del adata_mm_raw
adata_mm=adata_mm[:,np.array((adata_mm.X>0).sum(axis=0)>20).ravel()]
adata_mm=adata_mm[:,[g for g in oto_orthologues.eid_mm if g in adata_mm.var_names]]
sc.pp.normalize_total(adata_mm, target_sum=1e4)
sc.pp.log1p(adata_mm)

# %%
adata_mm

# %%
# HVGs on the final cell subset
# donor_id and batch are the same in this dataset
sc.pp.highly_variable_genes(
     adata=adata_mm, n_top_genes=3000, flavor='cell_ranger', batch_key='donor_id',
    subset=True)
adata_mm.shape

# %%
adata_mm

# %% [markdown]
# ### Human

# %%
adata_hs=sc.read(path_hs)

# %%
adata_hs

# %%
# Remove erythrocytes
adata_hs = adata_hs[adata_hs.obs['cell_type'] != 'erythrocyte'].copy()
adata_hs

# %%
adata_hs.X[:100, :100].max()

# %%
adata_hs.var.index[:3], oto_orthologues[:3]

# %%
# X is already count, remove lowly expr genes, and normalise
adata_hs=adata_hs[:,np.array((adata_hs.X>0).sum(axis=0)>20).ravel()]
gs=set(adata_hs.var_names)
adata_hs=adata_hs[:,[g for g in set(oto_orthologues.eid_hs.values) if g in gs]]
adata_hs.layers['counts']=adata_hs.X.copy()
sc.pp.normalize_total(adata_hs, target_sum=1e4)
sc.pp.log1p(adata_hs)

# %%
# HVGs on the final cell subset
# donor_id and sample_id are equal in this data
sc.pp.highly_variable_genes(
     adata=adata_hs, n_top_genes=3000, flavor='cell_ranger', batch_key='donor_id',
    subset=True)
adata_hs.shape

# %%
adata_hs

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


# %% [markdown]
# ### Combine adatas of mm and hs

# %% [markdown]
# Match cell types

# %%
sorted(adata_hs.obs['cell_type'].str.lower().unique())

# %%
sorted(adata_mm.obs['cell_type'].str.lower().unique())

# %%
adata_mm.obs['cell_type_eval'] = adata_mm.obs['cell_type'].str.lower().replace({
    'schwann cell': 'SOX10',
    'chondrocyte': 'Chondrocyte',
    'endothelial cell': 'Endothelial',
    'fibroblast': 'Fibroblast',
    'fibroblast of lymphatic vessel': 'Lymphatic',
    'innate lymphoid cell': 'CD3E & CPA3', # CD3E cluster
    'keratinocyte': 'Epidermal',
    'mast cell': 'CD3E & CPA3', # CPA3 cluster
    'neutrophil': 'Neutrophil', # Not sure if they should be separate (absent in human) or if in human they are present within the Immune cluster and should be merged there
    'pericyte': 'Pericyte',
    'professional antigen presenting cell': 'Immune',
})
pd.crosstab(adata_mm.obs['cell_type'], adata_mm.obs['cell_type_eval'])

# %%
adata_hs.obs['cell_type_eval']=adata_hs.obs['cell_type'].str.lower().replace({
    'langerhans cell':  'Immune', # CD74 and others cluster
    'basal cell of epidermis': 'Epidermal',
    'cytotoxic t cell': 'CD3E & CPA3', # CD3E and CPA3 are two separate cell types, but in human they are not properly subclustered (in mouse they are), so I merged the annotation
    'dendritic cell, human':  'Immune',
    'endothelial cell': 'Endothelial',
    'endothelial cell of lymphatic vessel': 'Lymphatic',
    'fibroblast': 'Fibroblast',
    'fibroblast of papillary layer of dermis': 'Fibroblast',
    'granular cell of epidermis': 'Epidermal',
    'hair follicular keratinocyte': 'Epidermal',
    'helper t cell': 'CD3E & CPA3',
    'macrophage':  'Immune',
    'melanocyte of skin': 'SOX10', # Schwann & melanocyte cluster
    'monocyte':  'Immune',
    'natural killer cell': 'CD3E & CPA3',
    'pericyte': 'Pericyte',
    'prickle cell': 'Epidermal',
    'regulatory t cell': 'CD3E & CPA3',
    'skin fibroblast': 'Fibroblast',
    'suprabasal keratinocyte': 'Epidermal',
})
pd.crosstab(adata_hs.obs['cell_type'], adata_hs.obs['cell_type_eval'])

# %%
sorted(adata_mm.obs['cell_type_eval'].unique())

# %%
sorted(adata_hs.obs['cell_type_eval'].unique())

# %%

# %%
# Prepare adatas for concat and concat
adata_hs_sub=adata_hs.copy()
adata_mm_sub=adata_mm.copy()

# Human
obs_keep_hs=['donor_id', 'disease', 'sex', 'cell_type', 'cell_type_eval', 'sample_id']
adata_hs_sub.obs=adata_hs_sub.obs[obs_keep_hs]
adata_hs_sub.obs.rename({'sample_id': 'batch'}, axis=1, inplace=True)
adata_hs_sub.obs.rename({c:'hs_'+c for c in adata_hs_sub.obs.columns 
                         if c not in ['cell_type_eval', 'batch']}, 
                         axis=1, inplace=True)
adata_hs_sub.obs['system']=1
del adata_hs_sub.obsm
del adata_hs_sub.uns

# Mouse
obs_keep_mm=['donor_id', 'disease', 'cell_type','cell_type_eval', 'batch']
adata_mm_sub.obs=adata_mm_sub.obs[obs_keep_mm]
adata_mm_sub.obs.rename({'batch': 'batch'}, axis=1, inplace=True)
adata_mm_sub.obs.rename({c:'mm_'+c for c in adata_mm_sub.obs.columns 
                         if c not in ['cell_type_eval', 'batch']}, 
                         axis=1, inplace=True)
adata_mm_sub.obs['system']=0
del adata_mm_sub.obsm
del adata_hs_sub.uns

# Concat
adata_hs_sub.var['gene_symbol'] = adata_hs_sub.var.GeneSym
adata_hs_sub.var['gs_hs'] = adata_hs_sub.var.GeneSym
adata_hs_sub.var['gs_mm'] = adata_mm_sub.var.feature_name.tolist()
adata_hs_sub.var['hs_eid'] = adata_hs_sub.var_names
adata_hs_sub.var['mm_eid'] = adata_mm_sub.var_names.tolist()
adata_hs_sub.var_names = adata_hs_sub.var['gene_symbol'].tolist() # Can do as matched subsetting
adata_hs_sub.var = adata_hs_sub.var[['gene_symbol', 'gs_hs', 'gs_mm', 'hs_eid', 'mm_eid']]
adata_mm_sub.var = adata_hs_sub.var

adata=sc.concat([adata_mm_sub, adata_hs_sub], join='outer', merge='unique')

del adata_mm_sub
del adata_hs_sub
gc.collect()

# %%
adata.var, adata.obs

# %%
pd.crosstab(adata.obs.cell_type_eval,adata.obs.system)

# %%
# N samples and cells per system
display(adata.obs.groupby('system')['batch'].nunique())
display(adata.obs.groupby('system').size())

# %% [markdown]
# Add PCA and leiden for scGLUE and Saturn

# %%
# PCA and clusters per system
n_pcs=15
X_pca_system=[]
leiden_system=[]
for system in adata.obs.system.unique():
    adata_sub=adata[adata.obs.system==system,:].copy()
    sc.pp.scale(adata_sub)
    sc.pp.pca(adata_sub, n_comps=n_pcs)
    sc.pp.neighbors(adata_sub, n_pcs=n_pcs)
    sc.tl.leiden(adata_sub)
    X_pca_system.append(pd.DataFrame(adata_sub.obsm['X_pca'],index=adata_sub.obs_names))
    leiden_system.append(adata_sub.obs.apply(lambda x: str(x['system'])+'_'+x['leiden'],axis=1))
del adata_sub
X_pca_system=pd.concat(X_pca_system)
leiden_system=pd.concat(leiden_system)
adata.obsm['X_pca_system']=X_pca_system.loc[adata.obs_names,:].values
adata.obs['leiden_system']=leiden_system.loc[adata.obs_names].values

# %% [markdown]
# ### Save

# %%
display(adata)

# %%
adata.write(path_save / 'skin_mm_hs_hvg.h5ad')

# %%
path_save / 'skin_mm_hs_hvg.h5ad'

# %% [markdown]
# #### Save in format for scGLUE/Saturn

# %%
# Save with gene symbols in var names for Saturn
adata_mm_sub=adata[adata.obs.system==0,:]
adata_mm_sub.var_names=adata_mm_sub.var['gs_mm']
adata_mm_sub.write(path_save / 'skin_mm_hs_mmPart_hvg.h5ad')
adata_hs_sub=adata[adata.obs.system==1,:]
adata_hs_sub.var_names=adata_hs_sub.var['gs_hs']
adata_hs_sub.write(path_save / 'skin_mm_hs_hsPart_hvg.h5ad')

# %%
adata_mm_sub

# %%
adata_hs_sub

# %%
adata.var.to_csv(path_save / 'combined_orthologuesHVG_geneMapping.tsv',index=False,sep='\t')

# %% [markdown]
# ## Non-integrated embedding

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
adata_embed.write(path_save / 'skin_mm_hs_hvg_embed.h5ad')

# %% [markdown]
# ## Integration metrics on non-integrated data

# %%
# Reload
adata_embed=sc.read(path_save / 'skin_mm_hs_hvg_embed.h5ad')

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
    labels=adata_embed.obs['cell_type_eval'])

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
pkl.dump({'asw_batch':asws},open(path_save / 'skin_mm_hs_hvg_embed_integrationMetrics.pkl','wb'))

# %% [markdown]
# ## Moran's I for eval
# Find genes that would be appropriate for computing Moran's I on for evaluation in every sample-cell type group (of appropriate size) by computing Moran's I on per-sample non integrated data. This can then also be used as a reference later on to compute relative preservation of Moran's I.

# %%
adata=sc.read(path_save / 'skin_mm_hs_hvg.h5ad')

# %%
# Potential groups to compute Moran's I on (batch-system and group)
cols=pd.get_option('display.max_rows')
pd.set_option('display.max_rows', 120)
display(pd.crosstab(adata.obs.batch,adata.obs.cell_type_eval))
pd.set_option('display.max_rows', cols)

# %%
# Filtered groups based on N cells and also remove the doublets
groups=adata.obs.groupby(['cell_type_eval','system','batch']).size()
groups=groups[(groups>=500).values&~groups.index.get_level_values(0).str.contains('\+')]
rows=pd.get_option('display.max_rows')
pd.set_option('display.max_rows', 250)
display(groups)
pd.set_option('display.max_rows', rows)
print('N cell types',groups.index.get_level_values('cell_type_eval').nunique())

# %%
# Compute Moran's I per group
data=[]
for group in groups.index:
    # Group adata
    print(group)
    adata_sub=adata[
        (adata.obs.cell_type_eval==group[0]).values&\
        (adata.obs.system==group[1]).values&\
        (adata.obs.batch==group[2]).values,:].copy()
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
# I thr
thr_mi=0.2

# %%
# N genes per group at certain thr
rows=pd.get_option('display.max_rows')
pd.set_option('display.max_rows', 250)
display(data.groupby(['group','system','batch']).apply(lambda x: (x['morans_i']>=thr_mi).sum()))
pd.set_option('display.max_rows', rows)

# %%
# N genes vs N cells in group
rcParams['figure.figsize']=(4,4)
sb.scatterplot(x='N cells',y='N genes',hue='level_0',style='system',
           data=pd.concat(
    [data.groupby(['group','system','batch']).apply(lambda x: (x['morans_i']>=thr_mi).sum()).rename('N genes'),
    groups.rename('N cells')],axis=1).reset_index(),s=20)
plt.legend(bbox_to_anchor=(1,1))
plt.xscale('log')

# %% [markdown]
# C: Thr of 0.2 has at least some genes for every group and not too many in any of the groups. Some groups would need lower/higher thr potentially.
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
pkl.dump(selected,open(path_save / 'skin_mm_hs_hvg_moransiGenes.pkl','wb'))

# %% [markdown]
# ## Batch effects within and between systems

# %%
adata=sc.read(path_save / 'skin_mm_hs_hvg.h5ad')

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
pca[['system','batch','group']]=adata.obs[['system','batch', 'cell_type_eval']]
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
    # Data for computing distances
    pca_s0=pca_mean[(pca_mean.index.get_level_values(0)==0) &
                    (pca_mean.index.get_level_values(2)==ct)]
    pca_s1=pca_mean[(pca_mean.index.get_level_values(0)==1) &
                    (pca_mean.index.get_level_values(2)==ct)]
    
    # Distances for s0 - within or between datasets
    d_s0=euclidean_distances(pca_s0)
    triu=np.triu_indices(pca_s0.shape[0],k=1)
    idx_map=dict(enumerate(pca_s0.index.get_level_values(1)))
    idx_within=([],[])
    idx_between=([],[])
    for i,j in zip(*triu):
        if idx_map[i]==idx_map[j]:
            idx_list=idx_within
        else:
            idx_list=idx_between
        idx_list[0].append(i)
        idx_list[1].append(j)
    d_s0_within=d_s0[idx_within]
    d_s0_between=d_s0[idx_between]
    
    # Distances for s1 and s0s1
    d_s1=euclidean_distances(pca_s1)[np.triu_indices(pca_s1.shape[0],k=1)]
    d_s0s1=euclidean_distances(pca_s0,pca_s1).ravel()
    distances[ct]={'s0_within':d_s0_within,'s0_between':d_s0_between,'s1':d_s1,'s0s1':d_s0s1}

# %%
# Save distances
pkl.dump(distances,open(path_save / 'skin_mm_hs_hvg_PcaSysBatchDist.pkl','wb'))

# %%
# Reload distances
#distances=pkl.load(open(path_save+'combined_orthologuesHVG_PcaSysBatchDist.pkl','rb'))

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
           sharey=False, height=2.5,aspect=1.5)

# %% [markdown]
# Evaluate statisticsal significance

# %%
# Compute significance of differences within and accross systems
signif=[]
for ct,dat in distances.items():
    for ref in ['s0_within','s0_between','s1']:
        n_system=dat[ref].shape[0]
        if n_system>=3:
            u,p=mannwhitneyu( dat[ref],dat['s0s1'],alternative='less')
            signif.append(dict( cell_type=ct,system=ref, u=u,pval=p,
                               n_system=n_system,n_crossystem=dat['s0s1'].shape[0]))
signif=pd.DataFrame(signif)
signif['padj']=multipletests(signif['pval'],method='fdr_bh')[1]

# %%
signif

# %%
# Save signif
signif.to_csv(path_save / 'skin_mm_hs_hvg_PcaSysBatchDist_Signif.tsv',sep='\t',index=False)

# %% [markdown]
# # Include non-one-to-one orthologues (Saturn, GLUE)

# %% [markdown]
# ## Prepare data for integration

# %%
adata = sc.read(path_save / 'skin_mm_hs_hvg.h5ad')
adata

# %%
adata[adata.obs['system'] == 1].obs['batch'].nunique()

# %% [markdown]
# ### Mouse

# %%
adata_mm=sc.read(path_mm)

# %%
# Genes with unique symbols (as Saturn embeddings are symbol based)
unique_gs=set(adata_mm.var_names[~adata_mm.var.duplicated('feature_name', keep=False)])

# %%
len(unique_gs)

# %%
# Add raw expression to X, remove lowly expr genes, and normalise
adata_mm_raw=adata_mm.raw.to_adata()
adata_mm=adata_mm_raw.copy()
adata_mm.layers['counts']=adata_mm_raw[adata_mm.obs_names,adata_mm.var_names].X.copy()
del adata_mm_raw
adata_mm=adata_mm[:,np.array((adata_mm.X>0).sum(axis=0)>20).ravel()]
adata_mm=adata_mm[:,[g for g in adata_mm.var_names if g in unique_gs]]
sc.pp.normalize_total(adata_mm, target_sum=1e4)
sc.pp.log1p(adata_mm)

# %%
# Add gene symbols as var names for Saturn
adata_mm.var['mm_eid'] = adata_mm.var_names
adata_mm.var['gs_mm'] = adata_mm.var.feature_name
adata_mm.var_names = adata_mm.var.feature_name.astype(str)

# %%
# HVGs on the final cell subset
sc.pp.highly_variable_genes(
     adata=adata_mm, n_top_genes=3000, flavor='cell_ranger', batch_key='donor_id',
    subset=True)
adata_mm.shape

# %%
adata_mm

# %% [markdown]
# ### Human

# %%
adata_hs=sc.read(path_hs)

# %%
# Remove erythrocytes
adata_hs = adata_hs[adata_hs.obs['cell_type'] != 'erythrocyte'].copy()
adata_hs

# %%
unique_gs=set(adata_hs.var_names[~adata_hs.var.duplicated('GeneSym', keep=False)])

# %%
# Add raw expression to X, remove lowly expr genes, and normalise
adata_hs=adata_hs[:,np.array((adata_hs.X>0).sum(axis=0)>20).ravel()]
adata_hs.layers['counts']=adata_hs.X.copy()
adata_hs=adata_hs[:,[g for g in adata_hs.var_names if g in unique_gs]]
sc.pp.normalize_total(adata_hs, target_sum=1e4)
sc.pp.log1p(adata_hs)

# %%
# Add gene symbols as var names for Saturn

adata_hs.var['hs_eid'] = adata_hs.var_names
adata_hs.var['gs_hs'] = adata_hs.var.GeneSym
adata_hs.var_names = adata_hs.var.GeneSym.astype(str)

# %%
# HVGs on the final cell subset
sc.pp.highly_variable_genes(
     adata=adata_hs, n_top_genes=3000, flavor='cell_ranger', batch_key='donor_id',
    subset=True)
adata_hs.shape

# %%
adata_hs

# %% [markdown]
# ### Combine adatas of mm and hs

# %% [markdown]
# Map cell type names

# %%
adata_mm.obs['cell_type_eval']=adata[adata_mm.obs.index].obs['cell_type_eval']
adata_hs.obs['cell_type_eval']=adata[adata_hs.obs.index].obs['cell_type_eval']

# %%
# Prepare adatas

# Human
obs_keep_hs=['donor_id', 'disease', 'sex', 'cell_type', 'cell_type_eval', 'sample_id']
adata_hs_sub=adata_hs.copy()
adata_hs_sub.obs=adata_hs_sub.obs[obs_keep_hs]
adata_hs_sub.obs.rename({'sample_id': 'batch'}, axis=1, inplace=True)
adata_hs_sub.obs.rename({c:'hs_'+c for c in adata_hs_sub.obs.columns 
                         if c not in ['cell_type_eval','batch']}, 
                         axis=1, inplace=True)
adata_hs_sub.obs['system']=1
del adata_hs_sub.obsm
del adata_hs_sub.uns
adata_hs_sub.var = adata_hs_sub.var[['hs_eid', 'gs_hs']]

# Mouse
adata_mm_sub=adata_mm.copy()
obs_keep_mm=['donor_id', 'disease', 'cell_type','cell_type_eval', 'batch']
adata_mm_sub.obs=adata_mm_sub.obs[obs_keep_mm]
adata_mm_sub.obs.rename({'batch': 'batch'}, axis=1, inplace=True)
adata_mm_sub.obs.rename({c:'mm_'+c for c in adata_mm_sub.obs.columns 
                         if c not in ['cell_type_eval','batch']}, 
                         axis=1, inplace=True)
adata_mm_sub.obs['system']=0

del adata_mm_sub.obsm
del adata_mm_sub.uns
adata_mm_sub.var = adata_mm_sub.var[['mm_eid', 'gs_mm']]

# %%

# %% [markdown]
# PCA and clusters for scglue and saturn

# %%
# PCA and clusters per system
n_pcs=15
for adata_temp in [adata_hs_sub,adata_mm_sub]:
    adata_sub=adata_temp.copy()
    sc.pp.scale(adata_sub)
    sc.pp.pca(adata_sub, n_comps=n_pcs)
    sc.pp.neighbors(adata_sub, n_pcs=n_pcs)
    sc.tl.leiden(adata_sub)
    adata_temp.obsm['X_pca_system']=adata_sub[adata_temp.obs_names,:].obsm['X_pca']
    adata_temp.obs['leiden_system']=adata_sub.obs.apply(lambda x: str(x['system'])+'_'+x['leiden'],axis=1)
del adata_sub

# %%
adata_mm_sub

# %%
adata_hs_sub

# %% [markdown]
# #### Save in format for scGLUE/Saturn

# %%
# Save with gene symbols in var names for Saturn
adata_mm_sub.write(path_save / 'skin_mm_hs_hvg-mmPart_nonortholHVG.h5ad')
adata_hs_sub.write(path_save / 'skin_mm_hs_hvg-hsPart_nonortholHVG.h5ad')

# %%
genes_mm=set(adata_mm_sub.var_names)
genes_hs=set(adata_hs_sub.var_names)
gene_mapping=orthology_info.query('gs_mm in @genes_mm and gs_hs in @genes_hs')[['gs_mm','gs_hs']]
print(gene_mapping.shape[0])
gene_mapping.to_csv(path_save / 'skin_mm_hs_hvg_nonortholHVG_geneMapping.tsv',index=False,sep='\t')


# %%
#adata_mm_sub=sc.read(path_save+'combined-mmPart_nonortholHVG.h5ad')
#adata_hs_sub=sc.read(path_save+'combined-hsPart_nonortholHVG.h5ad')

# %% [markdown]
# ## Moran's I eval of non-orthol
# To make it comparable it must anyway use the same set of genes as above for one-to-one orthologues. So no new Morans'I values/genes are selected.

# %%

# %%

# %%

# %% [markdown]
# # Remove all disease human samples as a use-case

# %%
def limit_to_human(adata):
    return adata[adata.obs['hs_disease'] != "psoriasis"].copy()


# %%
adata = sc.read(path_save / 'skin_mm_hs_hvg.h5ad')
adata = limit_to_human(adata)
adata.write(path_save / 'limited_data_skin_mm_hs_hvg.h5ad')

adata_mm_sub = sc.read(path_save / 'skin_mm_hs_mmPart_hvg.h5ad')
adata_mm_sub = limit_to_human(adata_mm_sub)
adata_mm_sub.write(path_save / 'limited_data_skin_mm_hs_mmPart_hvg.h5ad')

adata_hs_sub = sc.read(path_save / 'skin_mm_hs_hsPart_hvg.h5ad')
adata_hs_sub = limit_to_human(adata_hs_sub)
adata_hs_sub.write(path_save / 'limited_data_skin_mm_hs_hsPart_hvg.h5ad')


# %%

# %% [markdown]
# ## Non-integrated embedding

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
adata_embed.write(path_save / 'limited_data_skin_mm_hs_hvg_embed.h5ad')

# %% [markdown]
# ## Integration metrics on non-integrated data

# %%
# Reload
adata_embed=sc.read(path_save / 'limited_data_skin_mm_hs_hvg_embed.h5ad')

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
    labels=adata_embed.obs['cell_type_eval'])

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
pkl.dump({'asw_batch':asws},open(path_save / 'limited_data_skin_mm_hs_hvg_embed_integrationMetrics.pkl','wb'))


# %% [markdown]
# ## Moran's I for eval
# Find genes that would be appropriate for computing Moran's I on for evaluation in every sample-cell type group (of appropriate size) by computing Moran's I on per-sample non integrated data. This can then also be used as a reference later on to compute relative preservation of Moran's I.

# %%
adata=sc.read(path_save / 'limited_data_skin_mm_hs_hvg.h5ad')

# %%
# Potential groups to compute Moran's I on (batch-system and group)
cols=pd.get_option('display.max_rows')
pd.set_option('display.max_rows', 120)
display(pd.crosstab(adata.obs.batch,adata.obs.cell_type_eval))
pd.set_option('display.max_rows', cols)

# %%
# Filtered groups based on N cells and also remove the doublets
groups=adata.obs.groupby(['cell_type_eval','system','batch']).size()
groups=groups[(groups>=500).values&~groups.index.get_level_values(0).str.contains('\+')]
rows=pd.get_option('display.max_rows')
pd.set_option('display.max_rows', 250)
display(groups)
pd.set_option('display.max_rows', rows)
print('N cell types',groups.index.get_level_values('cell_type_eval').nunique())

# %%
# Compute Moran's I per group
data=[]
for group in groups.index:
    # Group adata
    print(group)
    adata_sub=adata[
        (adata.obs.cell_type_eval==group[0]).values&\
        (adata.obs.system==group[1]).values&\
        (adata.obs.batch==group[2]).values,:].copy()
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
# I thr
thr_mi=0.2

# %%
# N genes per group at certain thr
rows=pd.get_option('display.max_rows')
pd.set_option('display.max_rows', 250)
display(data.groupby(['group','system','batch']).apply(lambda x: (x['morans_i']>=thr_mi).sum()))
pd.set_option('display.max_rows', rows)

# %%
# N genes vs N cells in group
rcParams['figure.figsize']=(4,4)
sb.scatterplot(x='N cells',y='N genes',hue='level_0',style='system',
           data=pd.concat(
    [data.groupby(['group','system','batch']).apply(lambda x: (x['morans_i']>=thr_mi).sum()).rename('N genes'),
    groups.rename('N cells')],axis=1).reset_index(),s=20)
plt.legend(bbox_to_anchor=(1,1))
plt.xscale('log')

# %% [markdown]
# C: Thr of 0.2 has at least some genes for every group and not too many in any of the groups. Some groups would need lower/higher thr potentially.
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
pkl.dump(selected,open(path_save / 'limited_data_skin_mm_hs_hvg_moransiGenes.pkl','wb'))

# %% [markdown]
# ## Batch effects within and between systems

# %%
adata=sc.read(path_save / 'limited_data_skin_mm_hs_hvg.h5ad')

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
pca[['system','batch','group']]=adata.obs[['system','batch', 'cell_type_eval']]
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
    # Data for computing distances
    pca_s0=pca_mean[(pca_mean.index.get_level_values(0)==0) &
                    (pca_mean.index.get_level_values(2)==ct)]
    pca_s1=pca_mean[(pca_mean.index.get_level_values(0)==1) &
                    (pca_mean.index.get_level_values(2)==ct)]
    
    # Distances for s0 - within or between datasets
    d_s0=euclidean_distances(pca_s0)
    triu=np.triu_indices(pca_s0.shape[0],k=1)
    idx_map=dict(enumerate(pca_s0.index.get_level_values(1)))
    idx_within=([],[])
    idx_between=([],[])
    for i,j in zip(*triu):
        if idx_map[i]==idx_map[j]:
            idx_list=idx_within
        else:
            idx_list=idx_between
        idx_list[0].append(i)
        idx_list[1].append(j)
    d_s0_within=d_s0[idx_within]
    d_s0_between=d_s0[idx_between]
    
    # Distances for s1 and s0s1
    d_s1=euclidean_distances(pca_s1)[np.triu_indices(pca_s1.shape[0],k=1)]
    d_s0s1=euclidean_distances(pca_s0,pca_s1).ravel()
    distances[ct]={'s0_within':d_s0_within,'s0_between':d_s0_between,'s1':d_s1,'s0s1':d_s0s1}

# %%
# Save distances
pkl.dump(distances,open(path_save / 'limited_data_skin_mm_hs_hvg_PcaSysBatchDist.pkl','wb'))

# %%
# Reload distances
#distances=pkl.load(open(path_save+'combined_orthologuesHVG_PcaSysBatchDist.pkl','rb'))

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
           sharey=False, height=2.5,aspect=1.5)

# %% [markdown]
# Evaluate statisticsal significance

# %%
# Compute significance of differences within and accross systems
signif=[]
for ct,dat in distances.items():
    for ref in ['s0_within','s0_between','s1']:
        n_system=dat[ref].shape[0]
        if n_system>=3:
            u,p=mannwhitneyu( dat[ref],dat['s0s1'],alternative='less')
            signif.append(dict( cell_type=ct,system=ref, u=u,pval=p,
                               n_system=n_system,n_crossystem=dat['s0s1'].shape[0]))
signif=pd.DataFrame(signif)
signif['padj']=multipletests(signif['pval'],method='fdr_bh')[1]

# %%
signif

# %%
# Save signif
signif.to_csv(path_save / 'limited_data_skin_mm_hs_hvg_PcaSysBatchDist_Signif.tsv',sep='\t',index=False)

# %%

# %%

# %% [markdown]
# # Include non-one-to-one orthologues (Saturn, GLUE)

# %% [markdown]
# ## Prepare data for integration

# %%
adata_mm_sub = sc.read(path_save / 'skin_mm_hs_hvg-mmPart_nonortholHVG.h5ad')
adata_hs_sub = sc.read(path_save / 'skin_mm_hs_hvg-hsPart_nonortholHVG.h5ad')

# %%
adata_hs_sub = limit_to_human(adata_hs_sub)

# %% [markdown]
# PCA and clusters for scglue and saturn

# %%
# PCA and clusters per system
n_pcs=15
for adata_temp in [adata_hs_sub,adata_mm_sub]:
    adata_sub=adata_temp.copy()
    sc.pp.scale(adata_sub)
    sc.pp.pca(adata_sub, n_comps=n_pcs)
    sc.pp.neighbors(adata_sub, n_pcs=n_pcs)
    sc.tl.leiden(adata_sub)
    adata_temp.obsm['X_pca_system']=adata_sub[adata_temp.obs_names,:].obsm['X_pca']
    adata_temp.obs['leiden_system']=adata_sub.obs.apply(lambda x: str(x['system'])+'_'+x['leiden'],axis=1)
del adata_sub

# %% [markdown]
# #### Save in format for scGLUE/Saturn

# %%
# Save with gene symbols in var names for Saturn
adata_mm_sub.write(path_save / 'limited_data_skin_mm_hs_hvg-mmPart_nonortholHVG.h5ad')
adata_hs_sub.write(path_save / 'limited_data_skin_mm_hs_hvg-hsPart_nonortholHVG.h5ad')

# %%
genes_mm=set(adata_mm_sub.var_names)
genes_hs=set(adata_hs_sub.var_names)
gene_mapping=orthology_info.query('gs_mm in @genes_mm and gs_hs in @genes_hs')[['gs_mm','gs_hs']]
print(gene_mapping.shape[0])
gene_mapping.to_csv(path_save / 'limited_data_skin_mm_hs_hvg_nonortholHVG_geneMapping.tsv',index=False,sep='\t')

# %%
