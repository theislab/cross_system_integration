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

# %%
# Data pp imports
import scanpy as sc
import pickle as pkl
import pandas as pd
from scipy.sparse import csr_matrix, find
import scipy.stats as stats
import numpy as np

from matplotlib.pyplot import rcParams
import matplotlib.pyplot as plt
import seaborn as sb

# %%
# Modelling imports
import torch

import sys
sys.path.insert(0, '/lustre/groups/ml01/code/karin.hrovatin/cross_species_prediction/')
import constraint_pancreas_example.model._xybimodel as xybm
import importlib
importlib.reload(xybm)
from constraint_pancreas_example.model._xybimodel import XYBiModel

# %%
path_data='/lustre/groups/ml01/workspace/karin.hrovatin/data/cross_species_prediction/pancreas_example/'
path_genes='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/gene_lists/'

# %%
adata_hs=sc.read(path_data+'human.h5ad')
adata_mm=sc.read(path_data+'mouse.h5ad')

# %%
# Orthologues
orthology_info=pd.read_table(path_genes+'orthologues_ORGmus_musculus_ORG2homo_sapiens_V103.tsv'
                         ).rename(
    {'Gene name':'gs_mm','Human gene name':'gs_hs',
     'Gene stable ID':'eid_mm','Human gene stable ID':'eid_hs'},axis=1)

# %%
# Build orthologues info
orthologues={'x':[],'y':[]}
eids_hs_output=set(adata_hs.var_names)
for eid_mm in adata_mm.var_names:
    eids_hs=set(orthology_info.query('eid_mm==@eid_mm')['eid_hs'])&eids_hs_output
    for eid_hs in eids_hs:
        orthologues['x'].append(eid_mm)
        orthologues['y'].append(eid_hs)
orthologues=pd.DataFrame(orthologues)
print(f'N orthologue pairs: {orthologues.shape[0]}, '+ 
      f'N genes from mouse: {orthologues.x.nunique()}, '+ 
      f'N genes from human: {orthologues.y.nunique()}')

# %% [markdown]
# ## PP non-paired

# %%
# merge
# join outer to concat unmatched indices
adata=sc.concat([adata_mm,adata_hs],axis=1,join='outer')
adata.var.drop(adata.var.columns,axis=1,inplace=True)

# %%
# Add metadata
adata.var.loc[adata_hs.var_names,'species']='y'
adata.var.loc[adata_mm.var_names,'species']='x'

adata.obs.loc[adata_hs.obs_names,'batch_y']=adata_hs.obs.study_sample
adata.obs.loc[adata_mm.obs_names,'batch_x']=adata_mm.obs.study_sample
adata.obs.loc[adata_hs.obs_names,'cell_y']=adata_hs.obs_names
adata.obs.loc[adata_mm.obs_names,'cell_x']=adata_mm.obs_names

adata.obs.loc[adata_hs.obs_names,'train_y']=1
adata.obs.loc[adata_mm.obs_names,'train_x']=1
adata.obs['train_y'].fillna(0,inplace=True)
adata.obs['train_x'].fillna(0,inplace=True)

# %%
adata

# %% [markdown]
# ## Model - without paired cells and  without correlation constraint

# %% [markdown]
# ### Train

# %%
adata_training = XYBiModel.setup_anndata(
    adata=adata,
    xy_key='species',
    train_x_key='train_x',
    train_y_key='train_y',
    categorical_covariate_keys_x=['batch_x'],
    categorical_covariate_keys_y=['batch_y'],
)
model = XYBiModel(adata=adata_training)

# %%
model.train(max_epochs=20)

# %%
losses=['reconstruction_loss_train', 'reconstruction_loss_cycle_train', 
        'kl_local_train', 'kl_local_cycle_train', 
        'z_distance_paired_train', 'z_distance_cycle_train',
        'corr_cycle_train']
fig,axs=plt.subplots(1,len(losses),figsize=(len(losses)*3,2))
for ax,l in zip(axs,losses):
    ax.plot(
        model.trainer.logger.history[l].index,
        model.trainer.logger.history[l][l])
    ax.set_title(l)

# %%
embed = model.embed(
        adata=adata_training,
        indices=None,
        batch_size=None,
        as_numpy=True)

# %%
# Make adata out of embedding
embed=sc.AnnData(
    # Sepearte embeding from x and y modalities
    np.concatenate([
        embed[0][adata_training.obs['train_x'].astype(bool),:],
        embed[1][adata_training.obs['train_y'].astype(bool),:],
    ]),
    # Add obs and add modality origin
    obs=pd.concat([
        pd.concat([
            adata_training.obs[adata_training.obs['train_x'].astype(bool)],
            pd.DataFrame(
                {'species':['x']*int(adata_training.obs['train_x'].sum())},
                 index=adata_training.obs_names[adata_training.obs['train_x'].astype(bool)])
        ], axis=1),
        pd.concat([
            adata_training.obs[adata_training.obs['train_y'].astype(bool)],
            pd.DataFrame(
                {'species':['y']*int(adata_training.obs['train_y'].sum())},
                 index=adata_training.obs_names[adata_training.obs['train_y'].astype(bool)])
        ], axis=1)
    ]))

# %%
# Add extra metadata
embed.obs.loc[adata_hs.obs_names,'cell_type']=adata_hs.obs.cell_type_final.astype('str')
embed.obs.loc[adata_mm.obs_names,'cell_type']=adata_mm.obs.cell_type_final.astype('str')
embed.obs.loc[adata_hs.obs_names,'study_sample']=adata_hs.obs.study_sample.astype('str')
embed.obs.loc[adata_mm.obs_names,'study_sample']=adata_mm.obs.study_sample.astype('str')

# %%
sc.pp.neighbors(embed, use_rep='X')
sc.tl.umap(embed)

# %%
rcParams['figure.figsize']=(8,8)
sc.pl.umap(embed,color=['species','cell_type','study_sample'],s=10,wspace=0.5)

# %%
embed.obs.groupby('cell_type')['species'].value_counts()

# %% [markdown]
# ## Model - without paired cells

# %% [markdown]
# ### Train

# %%
adata.uns['orthology']=orthologues
adata_training = XYBiModel.setup_anndata(
    adata=adata,
    xy_key='species',
    train_x_key='train_x',
    train_y_key='train_y',
    orthology_key='orthology',
    categorical_covariate_keys_x=['batch_x'],
    categorical_covariate_keys_y=['batch_y'],
)
del adata.uns['orthology']
model = XYBiModel(adata=adata_training)

# %%
model.train(max_epochs=20,
            plan_kwargs={'corr_cycle_weight':100,'z_distance_cycle_weight':10}
           )

# %%
losses=['reconstruction_loss_train', 'reconstruction_loss_cycle_train', 
        'kl_local_train', 'kl_local_cycle_train', 
        'z_distance_paired_train', 'z_distance_cycle_train',
        'corr_cycle_train']
fig,axs=plt.subplots(1,len(losses),figsize=(len(losses)*3,2))
for ax,l in zip(axs,losses):
    ax.plot(
        model.trainer.logger.history[l].index,
        model.trainer.logger.history[l][l])
    ax.set_title(l)
fig.tight_layout()

# %%
embed = model.embed(
        adata=adata_training,
        indices=None,
        batch_size=None,
        as_numpy=True)

# %%
# Make adata out of embedding
embed=sc.AnnData(
    # Sepearte embeding from x and y modalities
    np.concatenate([
        embed[0][adata_training.obs['train_x'].astype(bool),:],
        embed[1][adata_training.obs['train_y'].astype(bool),:],
    ]),
    # Add obs and add modality origin
    obs=pd.concat([
        pd.concat([
            adata_training.obs[adata_training.obs['train_x'].astype(bool)],
            pd.DataFrame(
                {'species':['x']*int(adata_training.obs['train_x'].sum())},
                 index=adata_training.obs_names[adata_training.obs['train_x'].astype(bool)])
        ], axis=1),
        pd.concat([
            adata_training.obs[adata_training.obs['train_y'].astype(bool)],
            pd.DataFrame(
                {'species':['y']*int(adata_training.obs['train_y'].sum())},
                 index=adata_training.obs_names[adata_training.obs['train_y'].astype(bool)])
        ], axis=1)
    ]))

# %%
# Add extra metadata
embed.obs.loc[adata_hs.obs_names,'cell_type']=adata_hs.obs.cell_type_final.astype('str')
embed.obs.loc[adata_mm.obs_names,'cell_type']=adata_mm.obs.cell_type_final.astype('str')
embed.obs.loc[adata_hs.obs_names,'study_sample']=adata_hs.obs.study_sample.astype('str')
embed.obs.loc[adata_mm.obs_names,'study_sample']=adata_mm.obs.study_sample.astype('str')

# %%
sc.pp.neighbors(embed, use_rep='X')
sc.tl.umap(embed)

# %%
rcParams['figure.figsize']=(8,8)
sc.pl.umap(embed,color=['species','cell_type','study_sample'],s=10,wspace=0.5)

# %% [markdown]
# C: Requiring high correlation between predicted cells within the cycle (e.g. x_y_m, x_y_x_m) reduces integration (even within species). Increasing loss for distance on cycle z's does not help.
#
# High cycle generative prediction correlation forces z-s into region of latent space that is good for individual species.
#
# C: Maybe correl loss works to osme extent as prevents prediction of different cts from the same region (less ct overlap after using it).
#
# C: The problem is that the two spaces arent aligned - when no correl loss cts overlap and when correl loss they are pushed apart accross species (and batches). --> Would need on latent space not be able to learn which species the cell came from and at the same time high correl in the cycle. - Adversarial on predicted expression may not be enough as could still learn to predict realistic looking cells from different regions of latent space.

# %% [markdown]
# ### Effect of mock covariates (all 0)
# What is the effect of covariates being all 0 in the cycle pass (e.g. when we have no paired cells)?

# %%
adata_training_nocov=adata_training.copy()
for cov in ['covariates_x','covariates_y']:
    adata_training_nocov.obsm[cov]=pd.DataFrame(
        index=adata_training.obsm[cov].index,
        columns=adata_training.obsm[cov].columns).fillna(0)
embed_nocov = model.embed(
        adata=adata_training_nocov,
        indices=None,
        batch_size=None,
        as_numpy=True)
del adata_training_nocov

# %%
# Make adata out of embedding
embed_nocov=sc.AnnData(
    # Sepearte embeding from x and y modalities
    np.concatenate([
        embed_nocov[0][adata_training.obs['train_x'].astype(bool),:],
        embed_nocov[1][adata_training.obs['train_y'].astype(bool),:],
    ]),
    # Add obs and add modality origin
    obs=pd.concat([
        pd.concat([
            adata_training.obs[adata_training.obs['train_x'].astype(bool)],
            pd.DataFrame(
                {'species':['x']*int(adata_training.obs['train_x'].sum())},
                 index=adata_training.obs_names[adata_training.obs['train_x'].astype(bool)])
        ], axis=1),
        pd.concat([
            adata_training.obs[adata_training.obs['train_y'].astype(bool)],
            pd.DataFrame(
                {'species':['y']*int(adata_training.obs['train_y'].sum())},
                 index=adata_training.obs_names[adata_training.obs['train_y'].astype(bool)])
        ], axis=1)
    ]))
embed_nocov.obs['nocov']='True'
# Add extra metadata
embed_nocov.obs.loc[adata_hs.obs_names,'cell_type']=adata_hs.obs.cell_type_final.astype('str')
embed_nocov.obs.loc[adata_mm.obs_names,'cell_type']=adata_mm.obs.cell_type_final.astype('str')

# %%
# Make joing object with embedding taht has cov info for z and subset to make umap quicker
embed.obs['nocov']='False'
embed_cov_nocov=sc.concat([embed,embed_nocov])
np.random.seed(0)
random_indices=np.random.permutation(list(range(embed_cov_nocov.shape[0])))[:10000]
embed_cov_nocov=embed_cov_nocov[random_indices,:]

# %%
sc.pp.neighbors(embed_cov_nocov, use_rep='X')
sc.tl.umap(embed_cov_nocov)

# %%
rcParams['figure.figsize']=(8,8)
sc.pl.umap(embed_cov_nocov,color=['species','cell_type','nocov'],s=10,wspace=0.5)

# %% [markdown]
# C: Having empty covariates seems to work fine for computation of z.

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
x_mm=x_mm.groupby('cell_type').mean()[orthologues['x']]

x_hs=adata_hs.to_df()
x_hs['cell_type']=adata_hs.obs['cell_type_final']
x_hs=x_hs.groupby('cell_type').mean()[orthologues['y']]

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
# C: It seems that correlation on all genes may not be the best metric for connecting cell types as some cell types are very similar to each other or have enev higher correlation with another cell type.

# %% [markdown]
# #### Likelihood on cell types
# Compute mean and var in one species and compare to it mean from other species.

# %%
# Mean expression per ct in mm and hs
# Subset to orthologue genes
x_mm=adata_mm.to_df()
x_mm['cell_type']=adata_mm.obs['cell_type_final']
x_mm=x_mm.groupby('cell_type')
x_mm_m=x_mm.mean()[orthologues['x']]
x_mm_s=x_mm.std()[orthologues['x']]

x_hs=adata_hs.to_df()
x_hs['cell_type']=adata_hs.obs['cell_type_final']
x_hs=x_hs.groupby('cell_type')
x_hs_m=x_hs.mean()[orthologues['y']]
x_hs_s=x_hs.std()[orthologues['y']]

# %% [markdown]
# LL of mouse based on human distn

# %%
# LL of mouse based on human
ll_mm_hs=pd.DataFrame(index=x_mm_m.index,columns=x_hs_m.index)
ll_hs_mm=ll_mm_hs.copy()
for ct_mm in ll_mm_hs.index:
    for ct_hs in ll_mm_hs.columns:
        x_mm_m_ct=x_mm_m.loc[ct_mm,:].values.ravel()
        x_mm_s_ct=x_mm_s.loc[ct_mm,:].values.ravel()
        x_hs_m_ct=x_hs_m.loc[ct_hs,:].values.ravel()
        x_hs_s_ct=x_hs_s.loc[ct_hs,:].values.ravel()
        # Compute both ll of mm based on hs params and vice versa
        # Gaussian PDF equals log likelihood
        # Use nanmean as similarity cant be computed for genes with var=0, 
        # alternatively could add eps to var
        ll_mm_hs.at[ct_mm,ct_hs]=np.nanmean(
            stats.norm.pdf(x=x_mm_m_ct,loc=x_hs_m_ct,scale=x_hs_s_ct)) 
        ll_hs_mm.at[ct_mm,ct_hs]=np.nanmean(
            stats.norm.pdf(x=x_hs_m_ct,loc=x_mm_m_ct,scale=x_mm_s_ct))     
ll_mm_hs.index.name='mm'
ll_mm_hs.columns.name='hs'
ll_mm_hs=ll_mm_hs.astype(float)
ll_hs_mm.index.name='mm'
ll_hs_mm.columns.name='hs'
ll_hs_mm=ll_hs_mm.astype(float)

# %%
# mouse - target, human - distn
sb.clustermap(ll_mm_hs,figsize=(5,5))

# %%
# human - target, mouse - distn
sb.clustermap(ll_hs_mm,figsize=(5,5))

# %% [markdown]
# C: Log likelihood as simialrity measure does not work so well for corss species comparison. It seems that some cell types from the param distn are more likely to produce high likelihood against all other cts from the other species.

# %% [markdown]
# #### Correlation on cell type (and batch) subsets (individual cells)

# %%
cells_mm=[cell for cells in 
    adata_mm.obs.groupby(['cell_type_final','study_sample']
                         ).apply(lambda x: x.head(5).index).values for cell in cells]
cells_hs=[cell for cells in 
    adata_hs.obs.groupby(['cell_type_final','study_sample']
                         ).apply(lambda x: x.head(5).index).values for cell in cells]
# Correlations
cors=pd.DataFrame(index=cells_mm,columns=cells_hs)
# Converting to df and looping thrugh is much faster than subsetting 
# and converting for each cell pair separateloy
x_mm=adata_mm[cells_mm,orthologues['x']].to_df()
x_hs=adata_hs[cells_hs,orthologues['y']].to_df()
for cell_mm,x_cell_mm in x_mm.iterrows():
    for cell_hs,x_cell_hs in x_hs.iterrows():
        cors.at[cell_mm,cell_hs]=np.corrcoef(x_cell_mm,x_cell_hs)[0,1]                                            
cors.index.name='mm'
cors.columns.name='hs'
cors=cors.astype(float)

# %%
# Color annotation 
anno_mm={}
anno_hs={}
# Ct
cmap_ct=dict(zip(embed.obs.cell_type.cat.categories,embed.uns['cell_type_colors']))
anno_mm['cell_type']=adata_mm[cells_mm,:].obs.cell_type_final.map(cmap_ct)
anno_hs['cell_type']=adata_hs[cells_hs,:].obs.cell_type_final.map(cmap_ct)
# Sample
cmap_sample=dict(zip(embed.obs.study_sample.cat.categories,embed.uns['study_sample_colors']))
anno_mm['study_sample']=adata_mm[cells_mm,:].obs.study_sample.map(cmap_sample)
anno_hs['study_sample']=adata_hs[cells_hs,:].obs.study_sample.map(cmap_sample)
# Df for plotting
anno_mm=pd.DataFrame(anno_mm,index=cells_mm)
anno_hs=pd.DataFrame(anno_hs,index=cells_hs)

# %%
sb.clustermap(cors,xticklabels=False,yticklabels=False,
              row_colors=anno_mm,col_colors=anno_hs)

# %% [markdown]
# C: For some cell types correlation wont work well to distinguish them from other cell types. 
#
# C: The effect of batch is not large for most cell types.

# %% [markdown]
# C: But maybe correlation loss works well enough since it pushes cts away for most cases - no longer wrong cts overlap between species (in most cases - maybe need to improve for the endocrine).
#
# C: While NLL (reconstruction loss) works for latent embedding it may not work accross species due to different expression strength of individual genes. Reconstruction: NLL is good enough to separate between cts or else they would not mape separately in the latent space. - But this is of cell against itself and not accross species - easier problem.

# %% [markdown]
# ## PP paired
# For simplicity have all cells paired for a start (also remove cts with no cells in some species).

# %%
# Random pairs of cells accross species
# This does not esure that all pairs are unique
cts=set(adata_mm.obs.cell_type_final.unique())&set(adata_hs.obs.cell_type_final.unique())
cells_mm=[]
cells_hs=[]
for ct in cts:
    cells_sub_hs=adata_hs.obs.query('cell_type_final==@ct').index.values
    cells_sub_mm=adata_mm.obs.query('cell_type_final==@ct').index.values
    n_pairs=min([5000,len(cells_sub_hs)*len(cells_sub_mm)])
    print(ct,n_pairs)
    np.random.seed(0)
    cells_hs.extend(np.random.choice(cells_sub_hs,size=n_pairs))
    cells_mm.extend(np.random.choice(cells_sub_mm,size=n_pairs))
print('Total N cell pairs:',len(cells_mm))

# %%
# merge
# join outer to concat unmatched indices

# Prepare mm data
adata_mm_concat=adata_mm[cells_mm,:].copy()
adata_mm_concat.obs['cell_x']=adata_mm_concat.obs_names
adata_mm_concat.obs['batch_x']=adata_mm.obs.study_sample
adata_mm_concat.obs_names=np.array(range(adata_mm_concat.shape[0])).astype(str)
adata_mm_concat.var['species']='x'
adata_mm_concat.obs['train_x']=1

# Prepare hs data
adata_hs_concat=adata_hs[cells_hs,:].copy()
adata_hs_concat.obs['cell_y']=adata_hs_concat.obs_names
adata_hs_concat.obs['batch_y']=adata_hs.obs.study_sample
adata_hs_concat.obs_names=np.array(range(adata_hs_concat.shape[0])).astype(str)
adata_hs_concat.var['species']='y'
adata_hs_concat.obs['train_y']=1

adata=sc.concat([adata_mm_concat,adata_hs_concat],axis=1)
#adata.var.drop(adata.var.columns,axis=1,inplace=True)
for col in ['cell_x','batch_x','train_x']:
    adata.obs[col]=adata_mm_concat.obs[col]
for col in ['cell_y','batch_y','train_y']:
    adata.obs[col]=adata_hs_concat.obs[col]
adata.obs['train_y'].fillna(0,inplace=True)
adata.obs['train_x'].fillna(0,inplace=True)
adata.var.drop([c for c in adata.var.columns if c not in ['gene_symbol','species']],
               axis=1,inplace=True)

del adata_mm_concat
del adata_hs_concat

adata

# %% [markdown]
# ## Model - with paired cells

# %% [markdown]
# ### Train

# %%
adata.uns['orthology']=orthologues
adata_training = XYBiModel.setup_anndata(
    adata=adata,
    xy_key='species',
    train_x_key='train_x',
    train_y_key='train_y',
    orthology_key='orthology',
    categorical_covariate_keys_x=['batch_x'],
    categorical_covariate_keys_y=['batch_y'],
)
del adata.uns['orthology']
model = XYBiModel(adata=adata_training)

# %%
model.train(max_epochs=20,
            #plan_kwargs={'corr_cycle_weight':100,
            #             'z_distance_cycle_weight':10,'z_distance_paired_weight':10}
           )

# %%
losses=['reconstruction_loss_train', 'reconstruction_loss_cycle_train', 
        'kl_local_train', 'kl_local_cycle_train', 
        'z_distance_paired_train', 'z_distance_cycle_train',
        'corr_cycle_train']
fig,axs=plt.subplots(1,len(losses),figsize=(len(losses)*3,2))
for ax,l in zip(axs,losses):
    ax.plot(
        model.trainer.logger.history[l].index,
        model.trainer.logger.history[l][l])
    ax.set_title(l)
fig.tight_layout()

# %%
embed = model.embed(
        adata=adata_training,
        indices=None,
        batch_size=None,
        as_numpy=True)

# %%
# Make adata out of embedding
embed=sc.AnnData(
    # Sepearte embeding from x and y modalities
    np.concatenate([
        embed[0][adata_training.obs['train_x'].astype(bool),:],
        embed[1][adata_training.obs['train_y'].astype(bool),:],
    ]),
    # Add obs and add modality origin
    obs=pd.concat([
        pd.concat([
            adata_training.obs[adata_training.obs['train_x'].astype(bool)],
            pd.DataFrame(
                {'species':['x']*int(adata_training.obs['train_x'].sum())},
                 index=adata_training.obs_names[adata_training.obs['train_x'].astype(bool)])
        ], axis=1),
        pd.concat([
            adata_training.obs[adata_training.obs['train_y'].astype(bool)],
            pd.DataFrame(
                {'species':['y']*int(adata_training.obs['train_y'].sum())},
                 index=adata_training.obs_names[adata_training.obs['train_y'].astype(bool)])
        ], axis=1)
    ]))

# %%
# Add extra metadata
embed.obs_names=embed.obs.apply(lambda x: x.cell_x if x['species']=='x' else x.cell_y , axis=1)
cells=list(set(embed.obs_names)&set(adata_hs.obs_names))
embed.obs.loc[cells,'cell_type']=adata_hs[cells,:].obs.cell_type_final.astype('str')
embed.obs.loc[cells,'study_sample']=adata_hs[cells,:].obs.study_sample.astype('str')
cells=list(set(embed.obs_names)&set(adata_mm.obs_names))
embed.obs.loc[cells,'cell_type']=adata_mm[cells,:].obs.cell_type_final.astype('str')
embed.obs.loc[cells,'study_sample']=adata_mm[cells,:].obs.study_sample.astype('str')

# %%
sc.pp.neighbors(embed, use_rep='X')
sc.tl.umap(embed)

# %%
rcParams['figure.figsize']=(8,8)
sc.pl.umap(embed,color=['species','cell_type','study_sample'],s=10,wspace=0.5)

# %% [markdown]
# C: Having paired cells for sure helps with alignment, but losses still need to be tuned.

# %%

# %%
