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
import pandas as pd
import numpy as np
import scanpy as sc
import pickle as pkl
import math
import glob
import os
import gc
from pathlib import Path

from matplotlib import rcParams
import matplotlib.pyplot as plt
import seaborn as sb
import matplotlib.colors as mcolors

# %%
path_data='/home/moinfar/io/csi/'
path_names=path_data+'names_parsed/'
path_fig=path_data+'figures/'
path_ds='/om2/user/khrovati/data/datasets/d10_1016_j_cell_2020_08_013/'
path_integration=path_data+'/eval/retina_adult_organoid/integration/'
path_save=path_data+'/eval/retina_adult_organoid/integration_summary/'
path_save_mueller=path_save+'mueller_cells/'
path_save_amacrine=path_save+'amacrine_cells/'

# %%
# Names
model_map=pkl.load(open(path_names+'models.pkl','rb'))
model_map_rev=dict(zip(model_map.values(),model_map.keys()))
param_map=pkl.load(open(path_names+'params.pkl','rb'))
metric_map=pkl.load(open(path_names+'metrics.pkl','rb'))
dataset_map=pkl.load(open(path_names+'datasets.pkl','rb'))
metric_meaning_map=pkl.load(open(path_names+'metric_meanings.pkl','rb'))
metric_map_rev=dict(zip(metric_map.values(),metric_map.keys()))
dataset_map_rev=dict(zip(dataset_map.values(),dataset_map.keys()))
system_map=pkl.load(open(path_names+'systems.pkl','rb'))
params_opt_map=pkl.load(open(path_names+'params_opt_model.pkl','rb'))
params_opt_gene_map=pkl.load(open(path_names+'params_opt_genes.pkl','rb'))
param_opt_vals=pkl.load(open(path_names+'optimized_parameter_values.pkl','rb'))

# cmap
model_cmap=pkl.load(open(path_names+'model_cmap.pkl','rb'))
obs_col_cmap=pkl.load(open(path_names+'obs_col_cmap.pkl','rb'))
metric_background_cmap=pkl.load(open(path_names+'metric_background_cmap.pkl','rb'))

# %%
# Load embeddings
embeds={}
dataset='retina_adult_organoid'
top_settings=pkl.load(open(f'{path_data}eval/{dataset}/integration_summary/top_settings.pkl','rb'))
path_integration=f'{path_data}eval/{dataset}/integration/'
for model,model_setting in top_settings.items():
    model=model_map[model]
    for run in model_setting['runs']:
        if run==model_setting['mid_run']:
            embeds[model]=sc.read(path_integration+run+'/embed.h5ad')

# %% [markdown]
# ## Distinct cell type locations
#
# RPE were reproted in the paper to differ between tissue and organoid. We also observe some (but smaller) differences between astrocytes.

# %%
# Astrocytes and retinal pigment epithelial cell on integrated UMAPs
nrow=len(embeds)
ncol=2
fig,axs=plt.subplots(nrow,ncol,figsize=(ncol*2, nrow*2))
for i,(method,embed) in enumerate(embeds.items()):
    for j, system in enumerate(sorted(embed.obs.system.unique())):
        ax=axs[i,j]
        sc.pl.umap(embed,ax=ax,show=False,frameon=False)
        sc.pl.umap(embed[embed.obs.system==system,:],color='cell_type',
                   groups=['astrocyte','retinal pigment epithelial cell'],
                   ax=ax,show=False,title=method+' '+str(system),frameon=False)
        if j==0:
            ax.get_legend().remove()

# %% [markdown]
# ## Prepare adata for analys of markers

# %%
# Load expression for plotting markers
adata={}
for name, fn in [('periphery-adult','periphery'),
                 ('fovea-adult','fovea'),
                 ('organoid','organoid')]:
    a=sc.read(f'{path_ds}{fn}.h5ad')
    a.X=a.raw.X
    adata[name]=a
del a
gc.collect()
adata=sc.concat(adata.values(),
    label='material',keys =adata.keys(),index_unique='-',join='outer',merge='same')
adata=adata[:,
            np.array((adata[adata.obs.material!="organoid",:].X>0).sum(axis=0)>20).ravel()&\
            np.array((adata[adata.obs.material=="organoid",:].X>0).sum(axis=0)>20).ravel()
           ]
adata.obs['system_region']=adata.obs.material.map({
     'periphery-adult':'adult_periphery',
     'fovea-adult':'adult_fovea',
     'organoid':'organoid'})
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
gc.collect()

# %%
del adata.raw

# %% [markdown]
# ## Integration of Mueller cells
# Check if Mueller cells are more similar to preiphery than fovea, as reported in the paper.

# %%
# Subset to mueller cells and recompute embedding
embeds_sub={model:embed[embed.obs.cell_type=="Mueller cell",:].copy() for model,embed in embeds.items()}
for embed in embeds_sub.values():
    print(embed.shape)
    sc.pp.neighbors(embed, use_rep='X', n_neighbors=90)
    sc.tl.umap(embed)

# %% [markdown]
# Periphery/fovea markers

# %%
# Get exaple adata for extracting info below
adata_sub=adata[embeds_sub['cVAE'].obs_names,:]

# %%
# Expression of periphery/fovea markers
nrow=len(embeds_sub)
genes=['ATP1A2','COL2A1','FAM237B','RHCG']
ncol=len(genes)
fig,axs=plt.subplots(nrow,ncol,figsize=(ncol*2, nrow*2))
for i,(method,embed) in enumerate(embeds_sub.items()):
    for j, gene in enumerate(genes):
        ax=axs[i,j]
        adata_sub.obsm['X_umap']=embed[adata_sub.obs_names,:].obsm['X_umap']
        sc.pl.umap(adata_sub,ax=ax,show=False,color=gene,gene_symbols='feature_name',
                   title=gene+ ' '+ method,frameon=False)

# %% [markdown]
# Embedding density of organoid/fovea/periphery

# %%
# Make categories to analyse (organoid/fovea/periphery)
system_region=embed.obs.apply(
        lambda x: x['material']+('_'+x['region'] if x['material']=='adult' else ''),axis=1)
for embed in embeds_sub.values():
    embed.obs['system_region']=system_region

# %%
# Compute density for material (organoid/fovea/periphery)
for embed in embeds_sub.values():
    sc.tl.embedding_density(embed, basis='umap', groupby='system_region')
    embed.uns['umap_density_system_region_params']['covariate']=None

# %%
# Embedding density
nrow=len(embeds_sub)
groups=sorted(system_region.unique())
ncol=len(groups)
fig,axs=plt.subplots(nrow,ncol,figsize=(ncol*2, nrow*2))
for i,(method,embed) in enumerate(embeds_sub.items()):
    for j, group in enumerate(groups):
        ax=axs[i,j]
        sc.pl.umap( embed,ax=ax,show=False, frameon=False)
        sc.pl.embedding_density(
            embed[embed.obs.system_region==group,:],
            basis='umap', key='umap_density_system_region', title=method+ ' '+group,
            ax=ax,show=False, frameon=False)


# %% [markdown]
# #### Save data for embedding&marker plots

# %%
# remove old entries
for embed in embeds_sub.values():
    try:
        for c in ['cell_type_colors', 'leiden',  'sample_id_colors', 'scaled_neighbors', 'system_colors']:
            del embed.uns[c]
        del embed.obsm['X_umap_scaled']
        for c in ['scaled_connectivities', 'scaled_distances']:
            del embed.obsp[c]
    except:
        pass
try:
    del adata.obsm['X_umap']
except:
    pass

# %%
# Add constant model names to embeds
embeds_sub={model_map_rev[k]:v for k,v in embeds_sub.items()}

# %%
# Save embeds
Path(path_save_mueller).mkdir(parents=True, exist_ok=True)
pkl.dump(embeds_sub,open(path_save_mueller+'density_topmodels.pkl','wb'))

# %%
# Save expression
genes=['ATP1A2','COL2A1','FAM237B','RHCG']
adata_sub[:,adata_sub.var.query('feature_name in @genes').index].write(
    path_save_mueller+'adata_markers.h5ad')


# %% [markdown]
# ## scGEN analysis

# %% [markdown]
# Analyse scGEN examples with kl=0.1 and seed=1. Use vamp+cycle for comparison.

# %%
# Create new embeds dict and add model results
embeds_sub1={}
embeds_sub1['vamp_cycle']=embeds[model_map['vamp_cycle']]
example_runs=pkl.load(open(f'{path_data}eval/{dataset}/integration_summary/example_runs.pkl','rb'))
for model,run in example_runs.items():
    embeds_sub1[model]=sc.read(path_integration+run+'/embed.h5ad')                    

# %% [markdown]
# ### Mueller cells

# %%
# Subset to mueller cells and recompute embedding
embeds_sub={model:embed[embed.obs.cell_type=="Mueller cell",:].copy() for model,embed in embeds_sub1.items()}
for embed in embeds_sub.values():
    print(embed.shape)
    sc.pp.neighbors(embed, use_rep='X', n_neighbors=90)
    sc.tl.umap(embed)

# %%
# Make categories to analyse (organoid/fovea/periphery)
system_region=embed.obs.apply(
        lambda x: x['material']+('_'+x['region'] if x['material']=='adult' else ''),axis=1)
for embed in embeds_sub.values():
    embed.obs['system_region']=system_region

# %% [markdown]
# Markers

# %%
# Example adata for downstream info extraction
adata_sub=adata[embed.obs_names,:]

# %%
# Expression of periphery/fovea markers
nrow=len(embeds_sub)
genes=['ATP1A2','COL2A1','FAM237B','RHCG']
ncol=len(genes)
fig,axs=plt.subplots(nrow,ncol,figsize=(ncol*2, nrow*2))
for i,(method,embed) in enumerate(embeds_sub.items()):
    for j, gene in enumerate(genes):
        ax=axs[i,j]
        adata_sub.obsm['X_umap']=embed[adata_sub.obs_names,:].obsm['X_umap']
        sc.pl.umap(adata_sub,ax=ax,show=False,color=gene,gene_symbols='feature_name',
                   title=gene+ ' '+ method,frameon=False)

# %%
# Expression of periphery/fovea marker per material group
gene='RHCG'
nrow=len(embeds_sub)
ncol=len(embed.obs.system_region.unique())
fig,axs=plt.subplots(nrow,ncol,figsize=(ncol*2, nrow*2))
for i,(method,embed) in enumerate(embeds_sub.items()):
    adata_sub.obsm['X_umap']=embed[adata_sub.obs_names,:].obsm['X_umap']
    for j, group in enumerate(sorted(adata.obs.system_region.unique())):
        ax=axs[i,j]
        sc.pl.umap(adata_sub,ax=ax,show=False,frameon=False,s=50)
        adata_sub_group=adata_sub[adata_sub.obs.system_region==group,:].copy()
        adata_sub_group.obsm['X_umap']=embed[adata_sub_group.obs_names,:].obsm['X_umap']
        print( adata_sub_group.shape)
        sc.pl.umap(adata_sub_group,ax=ax,show=False,color=gene,gene_symbols='feature_name',
                   title=group+ ' '+ method,frameon=False,s=50,sort_order=True,vmax=3.5)

# %% [markdown]
# Plot author annotated cell sub-types within Mueller cells on individual embeddings

# %%
rcParams['figure.figsize']=(2,2)
sc.pl.umap(embeds_sub['vamp_cycle'],color='author_cell_type')

# %%
sc.pl.umap(embeds_sub['scgen_sample'],color='author_cell_type')

# %%
sc.pl.umap(embeds_sub['scgen_system'],color='author_cell_type')

# %% [markdown]
# Embedding density

# %%
# Compute density for material (organoid/fovea/periphery)
for embed in embeds_sub.values():
    sc.tl.embedding_density(embed, basis='umap', groupby='system_region')
    embed.uns['umap_density_system_region_params']['covariate']=None

# %%
# Embedding density
nrow=len(embeds_sub)
groups=sorted(system_region.unique())
ncol=len(groups)
fig,axs=plt.subplots(nrow,ncol,figsize=(ncol*2, nrow*2))
for i,(method,embed) in enumerate(embeds_sub.items()):
    for j, group in enumerate(groups):
        ax=axs[i,j]
        sc.pl.umap( embed,ax=ax,show=False, frameon=False)
        sc.pl.embedding_density(
            embed[embed.obs.system_region==group,:],
            basis='umap', key='umap_density_system_region', title=method+ ' '+group,
            ax=ax,show=False, frameon=False)


# %% [markdown]
# Save

# %%
# remove old entries
for embed in embeds_sub:
    try:
        for c in ['cell_type_colors', 'leiden',  'sample_id_colors', 'scaled_neighbors', 'system_colors']:
            del embed.uns[c]
        del embed.obsm['X_umap_scaled']
        for c in ['scaled_connectivities', 'scaled_distances']:
            del embed.obsp[c]
    except:
        pass
try:
    del adata_sub.obsm['X_umap']
except:
    pass

# %%
# Save embeds
pkl.dump(
    {'scgen_'+batch:embeds_sub['scgen_'+batch] for batch in ['sample','system']},
     open(path_save_mueller+'density_scgen_example.pkl','wb'))

# %% [markdown]
# ### Amarcine
# Analyse amarcine cell type subtypes on different embeddings

# %%
# List subtype counts
pd.crosstab(
    adata.obs.query('cell_type=="amacrine cell"')['system_region'],
    adata.obs.query('cell_type=="amacrine cell"')['author_cell_type']).T

# %%
# Subset to amarcine cells and recompute embedding
embeds_sub={model:embed[embed.obs.cell_type=="amacrine cell",:].copy() for model,embed in embeds_sub1.items()}
for embed in embeds_sub.values():
    print(embed.shape)
    sc.pp.neighbors(embed, use_rep='X', n_neighbors=90)
    sc.tl.umap(embed)

# %%
# Make categories to analyse (organoid/fovea/periphery)
system_region=embed.obs.apply(
        lambda x: x['material']+('_'+x['region'] if x['material']=='adult' else ''),axis=1)
for embed in embeds_sub.values():
    embed.obs['system_region']=system_region

# %%
# Get example adata for downstream metadata extraction
adata_sub=adata[embed.obs_names,:]

# %%
# Expression of starburst amacrine subtype marker
gene='SLC18A3'
nrow=len(embeds_sub)
ncol=len(embed.obs.system_region.unique())
fig,axs=plt.subplots(nrow,ncol,figsize=(ncol*2, nrow*2))
for i,(method,embed) in enumerate(embeds_sub.items()):
    adata_sub.obsm['X_umap']=embed[adata_sub.obs_names,:].obsm['X_umap']
    for j, group in enumerate(sorted(adata.obs.system_region.unique())):
        ax=axs[i,j]
        sc.pl.umap(adata_sub,ax=ax,show=False,frameon=False,s=50)
        adata_sub_group=adata_sub[adata_sub.obs.system_region==group,:].copy()
        adata_sub_group.obsm['X_umap']=embed[adata_sub_group.obs_names,:].obsm['X_umap']
        print( adata_sub_group.shape)
        sc.pl.umap(adata_sub_group,ax=ax,show=False,color=gene,gene_symbols='feature_name',
                   title=group+ ' '+ method,frameon=False,s=100)

# %%
# remove old entries
for embed in embeds_sub:
    try:
        for c in ['cell_type_colors', 'leiden',  'sample_id_colors', 'scaled_neighbors', 'system_colors']:
            del embed.uns[c]
        del embed.obsm['X_umap_scaled']
        for c in ['scaled_connectivities', 'scaled_distances']:
            del embed.obsp[c]
    except:
        pass
try:
    del adata_sub.obsm['X_umap']
except:
    pass

# %%
Path(path_save_amacrine).mkdir(parents=True, exist_ok=True)

# %%
# Save scGEN embeds
pkl.dump(
    {'scgen_'+batch:embeds_sub['scgen_'+batch] for batch in ['sample','system']},
     open(path_save_amacrine+'density_scgen_example.pkl','wb'))

# %%
# Save other embeds
pkl.dump({'vamp_cycle':embeds_sub['vamp_cycle']},
         open(path_save_amacrine+'density_topmodels.pkl','wb'))

# %%
# Save expression
# Only SLC18A3 used downstream
genes=['SLC18A3','PRDM13']
adata_sub[:,adata_sub.var.query('feature_name in @genes').index].write(
    path_save_amacrine+'adata_markers.h5ad')


# %%

# %%

# %%

# %%
