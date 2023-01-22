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
# # Match cell types between mouse and human tabula
# In order to perform later eval of integration try to match cell types accross tabulas.

# %%
import scanpy as sc
import pandas as pd
import pickle as pkl
import gc
import numpy as np
from collections import defaultdict
import random

from sklearn.metrics.pairwise import euclidean_distances

import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sb

import cell_ontology_based_harmonization as co
import obonet

# %%
path_ds='/lustre/groups/ml01/workspace/karin.hrovatin/data/datasets/'
path_hs=path_ds+'tabula_sapiens/'
path_mm=path_ds+'tabula_muris_senis/'
path_genes='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/gene_lists/'
path_save='/lustre/groups/ml01/workspace/karin.hrovatin/data/cross_species_prediction/tabula/'


# %%
orthologues=pd.read_table(path_genes+'orthologues_ORGmus_musculus_ORG2homo_sapiens_V103.tsv'
                         ).rename(
    {'Gene name':'gs_mm','Human gene name':'gs_hs',
     'Gene stable ID':'eid_mm','Human gene stable ID':'eid_hs'},axis=1)

# %%
adata_mm=sc.read(path_mm+'local.h5ad', backed='r')
adata_hs=sc.read(path_hs+'local.h5ad', backed='r')

# %% [markdown]
# ## Unify tissue

# %% [markdown]
# Unify tissue annotation - manual. Use publication not ontology tissues as easier to map

# %%
# All tissues
tissues_mm=sorted(adata_mm.obs['tissue_free_annotation'].unique())
tissues_hs=sorted(adata_hs.obs['tissue_in_publication'].unique())
print(tissues_mm)
print(tissues_hs)

# %%
tissue_map={
    'Aorta':'Heart_and_Aorta',
    'Heart':'Heart_and_Aorta',
    'Limb_Muscle':'Muscle',
    'Mammary_Gland':'Mammary',
    'Marrow':'Bone_Marrow',
    'Large_Intestine':'Intestine',
    'Small_Intestine':'Intestine',
}

# %%
adata_mm.obs['tissue_unified']=adata_mm.obs['tissue_free_annotation'].replace(tissue_map).values
adata_hs.obs['tissue_unified']=adata_hs.obs['tissue_in_publication'].replace(tissue_map).values

# %%
tissues_mm=set(adata_mm.obs['tissue_unified'].unique())
tissues_hs=set(adata_hs.obs['tissue_unified'].unique())
print('Shared:',sorted(tissues_mm&tissues_hs))
print('Mm only:',sorted(tissues_mm-tissues_hs))
print('Hs only:',sorted(tissues_hs-tissues_mm))

# %% [markdown]
# ## Unify cell types

# %% [markdown]
# Which cts are shared

# %%
# Shared cts
group='cell_type'
cts_mm=set(adata_mm.obs[group].unique())
cts_hs=set(adata_hs.obs[group].unique())
print(group,
    '\nN cell groups both:',len(cts_mm&cts_hs),
      '\nN cell groups mm unique:',len(cts_mm-cts_hs),
      '\nN cell groups hs unique:',len(cts_hs-cts_mm),
     )

# %%
shared_cts=cts_mm&cts_hs

# %% [markdown]
# How do shared cts map on umap

# %%
fig,ax=plt.subplots(figsize=(8,8))
sc.pl.umap(adata_mm,ax=ax,show=False)
sc.pl.umap(adata_mm[adata_mm.obs.cell_type.isin(shared_cts),:],color='cell_type',ax=ax)

# %%
fig,ax=plt.subplots(figsize=(8,8))
sc.pl.umap(adata_hs,ax=ax,show=False)
sc.pl.umap(adata_hs[adata_hs.obs.cell_type.isin(shared_cts),:],color='cell_type',ax=ax)

# %% [markdown]
# Cell numbers

# %%
# Number of shared ct cells
print(f"N shared cells mm: {adata_mm.obs.cell_type.isin(shared_cts).sum()}'+\
 f'and hs: {adata_hs.obs.cell_type.isin(shared_cts).sum()}")

# %%
# Sizes of shared ct groups in mm
rcParams['figure.figsize']=(4,2)
plt.boxplot(adata_mm[adata_mm.obs.cell_type.isin(shared_cts),:].obs['cell_type'].value_counts())
plt.yscale('log')

# %%
# Sizes of shared ct groups in hs
rcParams['figure.figsize']=(4,2)
plt.boxplot(adata_hs[adata_hs.obs.cell_type.isin(shared_cts),:].obs['cell_type'].value_counts())
plt.yscale('log')

# %% [markdown]
# ### Combine tissue and ct

# %%
# Shared cts per tissue
group='cell_type'
for tissue in sorted(tissues_mm&tissues_hs):
    cts_mm=set(adata_mm.obs.query('tissue_unified==@tissue')[group].unique())
    cts_hs=set(adata_hs.obs.query('tissue_unified==@tissue')[group].unique())
    print('\n',tissue,
        '\nN cell groups both:',len(cts_mm&cts_hs), cts_mm&cts_hs,
          '\nN cell groups mm unique:',len(cts_mm-cts_hs), cts_mm-cts_hs,
          '\nN cell groups hs unique:',len(cts_hs-cts_mm), cts_hs-cts_mm,
         )

# %%
# Combined tissue (unified above)-ct cell group
adata_mm.obs['tissue_unified_cell_type']=adata_mm.obs.apply(
    lambda x: x.at['tissue_unified']+'_'+x.at['cell_type'],axis=1)
adata_hs.obs['tissue_unified_cell_type']=adata_hs.obs.apply(
    lambda x: x.at['tissue_unified']+'_'+x.at['cell_type'],axis=1)

# %%
# Shared cts
group='tissue_unified_cell_type'
cts_mm=set(adata_mm.obs[group].unique())
cts_hs=set(adata_hs.obs[group].unique())
print(group,
    '\nN cell groups both:',len(cts_mm&cts_hs),
      '\nN cell groups mm unique:',len(cts_mm-cts_hs),
      '\nN cell groups hs unique:',len(cts_hs-cts_mm),
     )

# %%
shared_cts=cts_mm&cts_hs

# %% [markdown]
# Location of shared tissue-ct groups on umap

# %%
fig,ax=plt.subplots(figsize=(8,8))
sc.pl.umap(adata_mm,ax=ax,show=False)
sc.pl.umap(adata_mm[adata_mm.obs.tissue_unified_cell_type.isin(shared_cts),:],
           color='tissue_unified_cell_type',ax=ax)

# %%
fig,ax=plt.subplots(figsize=(8,8))
sc.pl.umap(adata_hs,ax=ax,show=False)
sc.pl.umap(adata_hs[adata_hs.obs.tissue_unified_cell_type.isin(shared_cts),:],
           color='tissue_unified_cell_type',ax=ax)

# %% [markdown]
# Cell numbers in shared groups

# %%
print(f"N shared cells mm: {adata_mm.obs.tissue_unified_cell_type.isin(shared_cts).sum()}"+\
 f" and hs: {adata_hs.obs.tissue_unified_cell_type.isin(shared_cts).sum()}")

# %%
# Shared cell group sizes in mm
rcParams['figure.figsize']=(4,2)
plt.boxplot(adata_mm[adata_mm.obs.tissue_unified_cell_type.isin(shared_cts),:].obs['cell_type'].value_counts())
plt.yscale('log')

# %%
# Shared cell group sizes in mm
rcParams['figure.figsize']=(4,2)
plt.boxplot(adata_hs[adata_hs.obs.tissue_unified_cell_type.isin(shared_cts),:].obs['cell_type'].value_counts())
plt.yscale('log')

# %% [markdown]
# C: There are quite some shared cell types accross tissues, but they are often immune cells and not other cells of interest. 
#
# C: Especially for human many cells are lost when adding tissue.

# %% [markdown]
# #### Tissue-ct groups with min size in both species
# How many cell groups do we keep if we require at least 50 cells per group in both species?

# %%
# Groups with min size in both species
min_cells=50
counts_mm=adata_mm.obs['tissue_unified_cell_type'].value_counts()
counts_hs=adata_hs.obs['tissue_unified_cell_type'].value_counts()
shared_cts=set(counts_mm[counts_mm>=min_cells].index) & \
    set(counts_hs[counts_hs>=min_cells].index)
print('N shared not too rare:',len(shared_cts))

# %%
fig,ax=plt.subplots(figsize=(8,8))
sc.pl.umap(adata_mm,ax=ax,show=False)
sc.pl.umap(adata_mm[adata_mm.obs.tissue_unified_cell_type.isin(shared_cts),:],
           color='tissue_unified_cell_type',ax=ax)

# %%
fig,ax=plt.subplots(figsize=(8,8))
sc.pl.umap(adata_hs,ax=ax,show=False)
sc.pl.umap(adata_hs[adata_hs.obs.tissue_unified_cell_type.isin(shared_cts),:],
           color='tissue_unified_cell_type',ax=ax)

# %%
print(f"N shared cells mm: {adata_mm.obs.tissue_unified_cell_type.isin(shared_cts).sum()}"+\
 f" and hs: {adata_hs.obs.tissue_unified_cell_type.isin(shared_cts).sum()}")

# %%
# Shared tissue-cell type groups with sufficient size
sorted(shared_cts)

# %% [markdown]
# C: It is important that some cells are close by on embedding as this is important for eval of bio preservation - hard case.

# %%
# Save the mapping
cells=pd.concat([
    pd.DataFrame({
            'obs_name':adata_mm.obs_names,
            'dataset':'tabula_muris_senis',
            'cell_type':[c if c in shared_cts else np.nan
                         for c in adata_mm.obs.tissue_unified_cell_type.values]
        }),
    pd.DataFrame({
            'obs_name':adata_hs.obs_names,
            'dataset':'tabula_sapiens',
            'cell_type':[c if c in shared_cts else np.nan
                         for c in adata_hs.obs.tissue_unified_cell_type.values]
        }),    
])

# %%
print(f'''cells in mm: 
{cells.query('dataset=="tabula_muris_senis" & ~cell_type.isna()',engine='python').shape[0]} 
cells in hs: 
{cells.query('dataset=="tabula_sapiens" & ~cell_type.isna()',engine='python').shape[0]}''')


# %%
cells.to_csv(path_save+'cell_type_mapping.tsv',sep='\t',index=False)

# %%
path_save+'cell_type_mapping.tsv'

# %% [markdown]
# ### Unify cts based on ontology
# Try to get more shared cell groups based on groing up the ontology annotations
#
# Use a function from lisa (ref atlases CZI project)

# %%
# Correct cell_type names according to onthoogy
ct_map={'mesenchymal stem cell of adipose':'mesenchymal stem cell of adipose tissue',
       'Muller cell':'Mueller cell'}
adata_mm.obs['cell_type']=adata_mm.obs['cell_type'].replace(ct_map)
adata_hs.obs['cell_type']=adata_hs.obs['cell_type'].replace(ct_map)

# %%
# Ontology graph
graph = obonet.read_obo('/lustre/groups/ml01/code/karin.hrovatin/sc-ref-assembly/docs/cl.obo')

# %%
# Match cts per tissue
for tissue in sorted(tissues_mm&tissues_hs):
    print(tissue)
    ct_parents = dict()
    ct_names=set(adata_mm.obs.query('tissue_unified==@tissue').cell_type.unique())|\
              set(adata_hs.obs.query('tissue_unified==@tissue').cell_type.unique())
    for ct_name in ct_names:
        if ct_name not in ct_parents:               
            ct_parent_names = co.get_parents(
                graph=graph, cell_type_name=ct_name, max_n_parents_per_gen=5, verbose=False
            )
            ct_parents[ct_name] = ct_parent_names
    # store in dataframe:
    ct_parent_df = pd.DataFrame(ct_parents)
    
    ct_to_integration_name = co.get_finest_common_labels(
        graph, ct_names, ct_parent_df, verbose=False
        )
    ct_mm_parsed=adata_mm.obs.query('tissue_unified==@tissue').cell_type.replace(
        ct_to_integration_name)
    ct_hs_parsed=adata_hs.obs.query('tissue_unified==@tissue').cell_type.replace(
        ct_to_integration_name)
    cts_mm=set(ct_mm_parsed.unique())
    cts_hs=set(ct_hs_parsed.unique())
    print(tissue,
        '\nN cell groups both:',len(cts_mm&cts_hs),cts_mm&cts_hs,
          '\nN cell groups mm unique:',len(cts_mm-cts_hs),cts_mm-cts_hs,
          '\nN cell groups hs unique:',len(cts_hs-cts_mm),cts_hs-cts_mm,
         )

# %% [markdown]
# C: Unification does not help much as creates very coarse clusters (also cts that remain species specific), does not create many more shared types. Problematic doing by tissue as then not matching accross tissues (some overlap), but doing it accross all tissues at once is a problem as get too general cts at the end.
#
# C: Also a problem that some tissues have some cts that are not present (annotated jointly on integrated space accross tissues) - would need to first filter by size of cell group. Example: hepatocyte in heart and aorta of hs.

# %% [markdown]
# ### Expression correlation between cts
# Find similar cts based on expr correlation of cell groups accross species.

# %%
adata_mm=sc.read(path_mm+'local.h5ad')
adata_hs=sc.read(path_hs+'local.h5ad')

# %% [markdown]
# Use for correlation shared HVGs computed per spicies on one to one orthologues.

# %%
# One to one orthologues - dont have same mm/hs gene in the table 2x
oto_orthologues=orthologues[~orthologues.duplicated('eid_mm').values & 
               ~orthologues.duplicated('eid_hs').values]
oto_eids_mm=list(set(oto_orthologues.eid_mm) & set(adata_mm.var_names))
oto_eids_hs=list(set(oto_orthologues.eid_hs) & set(adata_hs.var_names))
print(oto_orthologues.shape[0],len(oto_eids_mm),len(oto_eids_hs))

# %%
# Subset to oto orthologues
adata_mm=adata_mm[:,oto_eids_mm].copy()
gc.collect()
adata_hs=adata_hs[:,oto_eids_hs].copy()
gc.collect()

# %%
print(adata_mm.shape[0],adata_hs.shape[0])

# %%
# HVGs, compute a bit more since this wont be specific to any individual tissue
sc.pp.highly_variable_genes(adata_mm, n_top_genes=5000,batch_key='mouse.id')
sc.pp.highly_variable_genes(adata_hs, n_top_genes=5000,batch_key='donor_id')

# %%
# Shared HVGs
hvg_mm=set(adata_mm.var.query('highly_variable').index)
hvg_hs=set(adata_hs.var.query('highly_variable').index)
hvg_shared_mm=list(hvg_mm & set(oto_orthologues.query('eid_hs in @hvg_hs').eid_mm))
print(len(hvg_shared_mm))

# %%
# Expression of HVGs in cell of both species
x_mm=adata_mm[:,hvg_shared_mm].to_df()
x_hs=adata_hs[:,[oto_orthologues.query('eid_mm==@eid_mm')['eid_hs'].values[0]
                 for eid_mm in hvg_shared_mm]].to_df()

# %%
# Mean per ct
x_mm['cell_type']=adata_mm.obs['cell_type']
x_mm=x_mm.groupby('cell_type').mean()
x_hs['cell_type']=adata_hs.obs['cell_type']
x_hs=x_hs.groupby('cell_type').mean()

# %%
cor=pd.DataFrame(np.corrcoef(x_mm,x_hs)[:x_mm.shape[0],x_mm.shape[0]:],
                 index=x_mm.index,columns=x_hs.index)

# %%
sb.clustermap(cor)

# %% [markdown]
# C: Expresssion correltion also isnt clear enough to match accross species.

# %% [markdown]
# #### On clusters
# Try the same as above, but using clusters instead of cell types as some cell types differ accross tissues or are not annotated finely enough.

# %% [markdown]
# ##### Define clusters

# %%
# Show clusters present in mm data
rcParams['figure.figsize']=(8,8)
sc.pl.umap(adata_mm,color='leiden',s=10)

# %%
adata_mm.obs.leiden.nunique()

# %% [markdown]
# Make data hs clusters as not present

# %%
# What was used to compute neighbors present in human data?
adata_hs.uns['neighbors']

# %% [markdown]
# C: Human data already contains pre-computed neighbors from integrated embedding

# %% [markdown]
# Human clusters at different resolutions

# %%
sc.tl.leiden(adata_hs,resolution=2, key_added='leiden_r2')

# %%
adata_hs.obs.leiden_r2.nunique()

# %%
rcParams['figure.figsize']=(8,8)
cmap={ct:f"#{random.randrange(0x1000000):06x}" for ct in adata_hs.obs['leiden_r2'].unique()}
sc.pl.umap(adata_hs,color='leiden_r2',palette=cmap, s=10)

# %%
sc.tl.leiden(adata_hs,resolution=1, key_added='leiden_r1')

# %%
adata_hs.obs.leiden_r1.nunique()

# %%
rcParams['figure.figsize']=(8,8)
cmap={ct:f"#{random.randrange(0x1000000):06x}" for ct in adata_hs.obs['leiden_r1'].unique()}
sc.pl.umap(adata_hs,color='leiden_r1',palette=cmap, s=10)

# %%
sc.tl.leiden(adata_hs,resolution=0.7, key_added='leiden_r0.7')

# %%
adata_hs.obs['leiden_r0.7'].nunique()

# %%
rcParams['figure.figsize']=(8,8)
cmap={ct:f"#{random.randrange(0x1000000):06x}" for ct in adata_hs.obs['leiden_r0.7'].unique()}
sc.pl.umap(adata_hs,color='leiden_r0.7',palette=cmap, s=10)

# %%
sc.tl.leiden(adata_hs,resolution=0.5, key_added='leiden_r0.5')

# %%
adata_hs.obs['leiden_r0.5'].nunique()

# %%
rcParams['figure.figsize']=(8,8)
cmap={ct:f"#{random.randrange(0x1000000):06x}" for ct in adata_hs.obs['leiden_r0.5'].unique()}
sc.pl.umap(adata_hs,color='leiden_r0.5',palette=cmap, s=10)

# %% [markdown]
# ##### Cluster similarity

# %%
# Expression of HVGs in cell of both species
x_mm=adata_mm[:,hvg_shared_mm].to_df()
x_hs=adata_hs[:,[oto_orthologues.query('eid_mm==@eid_mm')['eid_hs'].values[0]
                 for eid_mm in hvg_shared_mm]].to_df()

# %%
# Mean per ct
x_mm['cluster']=adata_mm.obs['leiden']
x_mm=x_mm.groupby('cluster').mean()
x_hs['cluster']=adata_hs.obs['leiden_r0.5']
x_hs=x_hs.groupby('cluster').mean()

# %%
cor=pd.DataFrame(np.corrcoef(x_mm,x_hs)[:x_mm.shape[0],x_mm.shape[0]:],
                 index=x_mm.index,columns=x_hs.index)
cor.index.name='mouse'
cor.columns.name='human'

# %%
sb.clustermap(cor)

# %% [markdown]
# C: Note that human has some unique tissues
#
# C: Probably it is better to have shared cell groups from annotation above as presevres some fine variation rather than coarsesning this to try to find some matches.

# %% [markdown]
# ### Dispersed cts per tissue
# Find cts in each tissue whose embeddings are far away (median)

# %%
# Median distance of cells per ct on embedding
dist=[]
for species,data in [('mm',adata_mm),('hs',adata_hs)]:
    for ct in data.obs.tissueU_cell_type.unique():
        # Use umap as mm does not have other integrated embedding?
        x=data[data.obs.tissueU_cell_type==ct,:].obsm['X_umap']
        dist_me=np.median(euclidean_distances(x)[np.triu_indices(x.shape[0],k=1)])
        dist.append({'species':species,'ct':ct,'dist':dist_me})
dist=pd.DataFrame(dist)

# %%
fig,ax=plt.subplots(1,2,figsize=(10,3))
ax[0].hist(dist.query('species=="mm"')['dist'],bins=50)
ax[0].set_title('mm')
ax[0].axvline(3.5,c='r')
ax[1].hist(dist.query('species=="hs"')['dist'],bins=50)
ax[0].set_title('hs')
ax[1].axvline(4.95,c='r')

# %%
dist.query('species=="mm" & dist>3.5')

# %%
dist.query('species=="hs" & dist>4.95')

# %%
