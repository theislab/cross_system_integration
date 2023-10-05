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
import pandas as pd
import numpy as np
import scanpy as sc
import pickle as pkl
import math
import glob
import os
import gc

from matplotlib import rcParams
import matplotlib.pyplot as plt
import seaborn as sb
import matplotlib.colors as mcolors

# %%
path_data='/om2/user/khrovati/data/cross_system_integration/'
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
# ## Distinct ct locations
#
# Astrocites sem to differ while other cell types are aligned. Same for RPE (also reported in the paper).

# %%
# Astrocytes and retinal pigment epithelial cell
nrow=len(embeds)
ncol=2
fig,axs=plt.subplots(nrow,ncol,figsize=(ncol*2, nrow*2))
for i,(method,embed) in enumerate(embeds.items()):
    for j, system in enumerate(sorted(embed.obs.system.unique())):
        ax=axs[i,j]
        sc.pl.umap(embed,ax=ax,show=False,frameon=False)
        sc.pl.umap(embed[embed.obs.system==system,:],color='cell_type',
                   groups=['astrocyte','retinal pigment epithelial cell'],
                   ax=ax,show=False,title=method+' '+system,frameon=False)
        if j==0:
            ax.get_legend().remove()

# %% [markdown]
# ## Adata for analys of markers

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
# Sample metdata UMPAs

# %%
# Add organoid age info
org_age=embeds_sub['cVAE'].obs.apply(
    lambda x: x.condition.split('W')[1].split('_')[0] if  x.system=='0' else np.nan,
    axis=1)
for embed in embeds_sub.values():
    embed.obs['org_age']=org_age

# %%
# Sample info on Mueller cells
nrow=len(embeds_sub)
ncol=2
fig,axs=plt.subplots(nrow,ncol,figsize=(ncol*2, nrow*2))
for i,(method,embed) in enumerate(embeds_sub.items()):
    for j, system in enumerate(sorted(embed.obs.system.unique())[::-1]):
        ax=axs[i,j]
        sc.pl.umap(embed,ax=ax,show=False,frameon=False)
        if system=='0':
            sc.pl.umap(embed[embed.obs.system==system,:],color='org_age',
                       ax=ax,show=False,title=method+' '+system,frameon=False,s=1)
        else:
            sc.pl.umap(embed[embed.obs.system==system,:],color='region',
                       ax=ax,show=False,title=method+' '+system,frameon=False,s=1)
plt.subplots_adjust(wspace=1)

# %% [markdown]
# Periphery/fovea markers

# %%
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
# MEbedding density
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
# Save embeds with constant model names
embeds_sub={model_map_rev[k]:v for k,v in embeds_sub.items()}

# %%
# Save embeds
pkl.dump(embeds_sub,open(path_save_mueller+'density_topmodels.pkl','wb'))

# %%
# Reload
# embeds_sub={model_map[k]:v for k,v in 
#             pkl.load(open(path_save_mueller+'density_topmodels.pkl','rb')).items()}

# %%
# Save expression
genes=['ATP1A2','COL2A1','FAM237B','RHCG']
adata_sub[:,adata_sub.var.query('feature_name in @genes').index].write(
    path_save_mueller+'adata_markers.h5ad')


# %% [markdown]
# ## scGEN analysis

# %% [markdown]
# Analyse scGEN example with sample-level integration, kl=0.1 and seed=1. Use vamp+cycle for comparison.

# %%
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
adata_sub=adata[embed.obs_names,:]

# %%
# Expression of periphery/fovea markers, limiting value range
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
# Expression of periphery/fovea markers
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
# MEbedding density
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
# ### Cones

# %%
pd.crosstab(
    adata.obs.query('cell_type=="retinal cone cell"')['system_region'],
    adata.obs.query('cell_type=="retinal cone cell"')['author_cell_type'])

# %%
# Subset to mueller cells and recompute embedding
embeds_sub={model:embed[embed.obs.cell_type=="retinal cone cell",:].copy() for model,embed in embeds_sub1.items()}
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
adata_sub=adata[embed.obs_names,:]

# %%
# Expression of periphery/fovea markers
nrow=len(embeds_sub)
genes=['ARR3','OPN1SW']
ncol=len(genes)
fig,axs=plt.subplots(nrow,ncol,figsize=(ncol*2, nrow*2))
for i,(method,embed) in enumerate(embeds_sub.items()):
    for j, gene in enumerate(genes):
        ax=axs[i,j]
        adata_sub.obsm['X_umap']=embed[adata_sub.obs_names,:].obsm['X_umap']
        sc.pl.umap(adata_sub,ax=ax,show=False,color=gene,gene_symbols='feature_name',
                   title=gene+ ' '+ method,frameon=False)

# %%
# Expression of periphery/fovea markers
gene='OPN1SW'
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
rcParams['figure.figsize']=(2,2)
for i,(method,embed) in enumerate(embeds_sub.items()):
    sc.pl.umap(embed,color=['system_region','author_cell_type'],wspace=0.8)

# %%
# Astrocytes and retinal pigment epithelial cell
nrow=len(embeds_sub)
ncol=3
fig,axs=plt.subplots(nrow,ncol,figsize=(ncol*2, nrow*2))
for i,(method,embed) in enumerate(embeds_sub.items()):
    for j, group in enumerate(sorted(embed.obs.system_region.unique())):
        ax=axs[i,j]
        sc.pl.umap(embed,ax=ax,show=False,frameon=False)
        sc.pl.umap(embed[embed.obs.system_region==group,:],color='author_cell_type',
                   groups=['L/M cone','S cone'],
                   ax=ax,show=False,title=method+'\n'+group,frameon=False)
        if j<2:
            ax.get_legend().remove()

# %% [markdown]
# C: Cone cells are not the best example as they dont have enough cells per ct of interest

# %% [markdown]
# ### Amarcine
# AC_Y_03, AC_B_16

# %%
pd.crosstab(
    adata.obs.query('cell_type=="amacrine cell"')['system_region'],
    adata.obs.query('cell_type=="amacrine cell"')['author_cell_type']).T

# %%
# Subset to mueller cells and recompute embedding
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
# Astrocytes and retinal pigment epithelial cell
nrow=len(embeds_sub)
ncol=3
fig,axs=plt.subplots(nrow,ncol,figsize=(ncol*2, nrow*2))
for i,(method,embed) in enumerate(embeds_sub.items()):
    for j, group in enumerate(sorted(embed.obs.system_region.unique())):
        ax=axs[i,j]
        sc.pl.umap(embed,ax=ax,show=False,frameon=False)
        sc.pl.umap(embed[embed.obs.system_region==group,:],color='author_cell_type',
                   groups=['AC_Y_03','AC_B_16'],
                   palette={**{ct:'lightgray' for ct in embed.obs.author_cell_type.unique()},
                            **{'AC_Y_03':'r','AC_B_16':'b',}},
                   ax=ax,show=False,title=method+'\n'+group,frameon=False,s=30)
        if j<2:
            ax.get_legend().remove()

# %%
adata_sub=adata[embed.obs_names,:]

# %%
# Expression of periphery/fovea markers
nrow=len(embeds_sub)
genes=['GJD2','SLC18A3']
ncol=len(genes)
fig,axs=plt.subplots(nrow,ncol,figsize=(ncol*2, nrow*2))
for i,(method,embed) in enumerate(embeds_sub.items()):
    for j, gene in enumerate(genes):
        ax=axs[i,j]
        adata_sub.obsm['X_umap']=embed[adata_sub.obs_names,:].obsm['X_umap']
        sc.pl.umap(adata_sub,ax=ax,show=False,color=gene,gene_symbols='feature_name',
                   title=gene+ ' '+ method,frameon=False,s=50)

# %%
# Expression of periphery/fovea markers
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
gene='GJD2'
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
sc.pl.umap(embeds_sub['vamp_cycle'],color='author_cell_type')

# %%
sc.pl.umap(embeds_sub['scgen_sample'],color='author_cell_type')

# %%
sc.pl.umap(embeds_sub['scgen_system'],color='author_cell_type')

# %%
# Expression of periphery/fovea markers
gene='PRDM13'
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

# %% [markdown]
# C: It may be better to rely on expression of markers than author annotation as it may be biased & noisy

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
     open(path_save_amacrine+'density_scgen_example.pkl','wb'))

# %%
# Save embeds
pkl.dump({'vamp_cycle':embeds_sub['vamp_cycle']},
         open(path_save_amacrine+'density_topmodels.pkl','wb'))

# %%
# Save expression
genes=['SLC18A3','PRDM13']
adata_sub[:,adata_sub.var.query('feature_name in @genes').index].write(
    path_save_amacrine+'adata_markers.h5ad')


# %%
