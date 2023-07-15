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

import pickle as pkl

from matplotlib import rcParams
import matplotlib.pyplot as plt
import seaborn as sb

# %%
path='/net/bmc-lab6/data/lab/kellis/users/khrovati/data/'
path_data=path+'datasets/10_1016_j_cell_2020_08_013/'
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
adata

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
del adata_sub.uns
del adata_sub.obsm
adata_sub.obs=adata_sub.obs[[
    'cell_type','cell_type_group','author_cell_type','condition', 
    'dataset','sample_id','ega_sample_alias', 'hca_data_portal_donor_uuid', 'hca_data_portal_cellsuspension_uuid', 
    'region',  'material', 'system']]

# %%
adata_sub.obs['system']=adata_sub.obs['material'].map({"organoid":0,'adult':1})

# %%
adata_sub.layers['counts']=adata[adata_sub.obs_names,adata_sub.var_names].X.copy()

# %%
adata_sub

# %%
adata_sub.write(path_train+'combined_HVG.h5ad')

# %% [markdown]
# # Moran's I for eval

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
    group['genes']=data_sub.query('morans_i>=@thr_mi')['morans_i']
    selected.append(group)

# %%
# Save
pkl.dump(selected,open(path_train+'combined_HVG_moransiGenes.pkl','wb'))
