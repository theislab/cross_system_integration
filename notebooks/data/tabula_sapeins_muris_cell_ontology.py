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
import scanpy as sc
import pandas as pd
import pickle as pkl
import gc
import numpy as np
from collections import defaultdict

from sklearn.metrics.pairwise import euclidean_distances

import matplotlib.pyplot as plt
from matplotlib import rcParams

import cell_ontology_based_harmonization as co
import obonet

# %%
path_ds='/lustre/groups/ml01/workspace/karin.hrovatin/data/datasets/'
path_hs=path_ds+'tabula_sapiens/'
path_mm=path_ds+'tabula_muris_senis/'
path_genes='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/gene_lists/'


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

# %% [markdown]
# ### Unify cts based on ontology

# %%
# Correct cell_type names according to onthoogy
ct_map={'mesenchymal stem cell of adipose':'mesenchymal stem cell of adipose tissue',
       'Muller cell':'Mueller cell'}
adata_mm.obs['cell_type']=adata_mm.obs['cell_type'].replace(ct_map)
adata_hs.obs['cell_type']=adata_hs.obs['cell_type'].replace(ct_map)

# %%
graph = obonet.read_obo('/lustre/groups/ml01/code/karin.hrovatin/sc-ref-assembly/docs/cl.obo')

# %%
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
# C: Unification doe snot help much as creates very coarse clusters (also cts that remain species specific), does not create many more shared types, problematic doing by tissue as then not matching accross tissues (some overlap) but doin accross tissues problem as get too general cts at the end.

# %% [markdown]
# ### Expression correlation between cts
# Find similar cts based on expr correlation.

# %%
adata_mm=sc.read(path_mm+'local.h5ad')
adata_hs=sc.read(path_hs+'local.h5ad')

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
# HVGs, compute a bit more since this wont be specific to any individual tissue
sc.pp.highly_variable_genes(adata_mm, n_top_genes=5000,batch_key='mouse.id')
sc.pp.highly_variable_genes(adata_hs, n_top_genes=5000,batch_key='donor_id')

# %%
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
cor=pd.DataFrame(np.corrcoef(x_mm,x_hs)[x_mm.shape[0]:,:x_mm.shape[0]],
                 index=x_mm.index,columns=x_hs.index)

# %% [markdown]
# ### Dispersed cts per tissue
# Find cts in each tissue whose embeddings are far away (median)

# %%
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
