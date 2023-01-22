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
# Look at human markers in the data used for model training.
#
# Prepare gene marker embedding from the human pancreas data.

# %%
import scanpy as sc
import pandas as pd
import pickle as pkl
import gc
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import minmax_scale, MinMaxScaler

import seaborn as sb

# %%
path_mm='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/combined/'
fn_hs='/lustre/groups/ml01/workspace/eva.lavrencic/data/pancreas/combined/adata_integrated_log_normalised_reann.h5ad'
path_genes='/lustre/groups/ml01/workspace/karin.hrovatin/data/cross_species_prediction/pancreas_ct_example_constraint/'
path_hs_symbols='/lustre/groups/ml01/projects/2020_pancreas_karin.hrovatin/data/pancreas/scRNA/P21002/rev7/cellranger/MUC26033/count_matrices/filtered_feature_bc_matrix/features.tsv'

# %%
markers=pkl.load(open(path_genes+'endo_markers_set_hs.pkl','rb'))

# %% [markdown]
# ### Markers only

# %%
hs_genes=list(set(
    [g for gs in markers.values() 
          for g in gs]))

# %%
adata_hs=sc.read(fn_hs)
size_factors=adata_hs.obs['size_factors'].values[:,None]
adata_hs=adata_hs.raw.to_adata()
adata_hs.X=csr_matrix(adata_hs.X)
adata_hs.var['gene_symbol']=pd.read_table(path_hs_symbols,index_col=0,header=None
                          ).rename({1:'gene_symbol'},axis=1).loc[adata_hs.var_names,'gene_symbol']

del adata_hs.obsm
adata_hs=adata_hs[:,adata_hs.var.query('gene_symbol in @hs_genes').index].copy()
gc.collect()
adata_hs.X /=size_factors
sc.pp.log1p(adata_hs)
adata_hs.X=csr_matrix(adata_hs.X)

# %%
adata_hs.obs['age_years']=adata_hs.obs['age'].str.replace(' y','').astype(float)
adata_hs=adata_hs[(adata_hs.obs['age_years']>=19) & (adata_hs.obs['age_years']<=64)  &\
    (adata_hs.obs['disease']=='healthy') & (adata_hs.obs['study']!='human_spikein_drug'),
                  :].copy()

# %%
cts=['alpha','beta','gamma','delta']

# %%
adata_hs=adata_hs[adata_hs.obs.query('cell_type_integrated_subcluster in @cts').index,:].copy()

# %%
gc.collect()

# %%
adata_hs.obs['cell_type_final']=adata_hs.obs['cell_type_integrated_subcluster']

# %%
adata_hs

# %%
x_pb=adata_hs.to_df()
group_cols=['study_sample','cell_type_final']
x_pb[group_cols]=adata_hs.obs[group_cols]
x_pb=x_pb.groupby(group_cols,observed=True).apply(
    lambda x:x.drop(group_cols,axis=1).mean(axis=0))

# %%
# Remove all 0 expressed genes
print('N all-zero genes:',(x_pb==0).all().sum())
x_pb=x_pb.loc[:,(x_pb>0).any()]
x_pb.shape

# %%
# Gene marker anno
marker_anno=pd.Series(index=x_pb.columns)
colors={'beta':'r','alpha':'y','gamma':'g','delta':'b'}
for ct,genes in markers.items():
    eids=adata_hs.var.query('gene_symbol in @genes').index
    marker_anno.loc[eids]=colors[ct]

# %%
# Ct anno
ct_anno=pd.Series(
    [colors[ct] for ct in x_pb.index.to_frame()['cell_type_final'].values], index=x_pb.index)


# %%
sb.clustermap(pd.DataFrame(minmax_scale(x_pb),columns=x_pb.columns, index=x_pb.index),
              xticklabels=False,yticklabels=False,
             col_colors=marker_anno,row_colors=ct_anno)

# %% [markdown]
# Save

# %%
scaler=MinMaxScaler().fit(x_pb)
x_pb_scl=pd.DataFrame(scaler.transform(x_pb),index=x_pb.index,columns=x_pb.columns)

# %%
x_pb_scl.to_csv(path_genes+'pseudobulkHS_pancreasEndo.tsv',sep='\t')
pd.DataFrame({'min':scaler.data_min_,'max':scaler.data_max_},index=x_pb.columns
            ).to_csv(path_genes+'pseudobulkHS_pancreasEndo_MinMaxScale.tsv',sep='\t')

# %% [markdown]
# ### Markers + HVGs

# %%
adata_hs=sc.read(fn_hs)

# %%
# Get only healthy adults
adata_hs.obs['age_years']=adata_hs.obs['age'].str.replace(' y','').astype(float)
adata_hs=adata_hs[(adata_hs.obs['age_years']>=19) & (adata_hs.obs['age_years']<=64)  &\
    (adata_hs.obs['disease']=='healthy') & (adata_hs.obs['study']!='human_spikein_drug'),
                  :].copy()

# %%
# Cts to use
cts=['alpha','beta','gamma','delta']

# %%
# Subset to cts
adata_hs=adata_hs[adata_hs.obs.query('cell_type_integrated_subcluster in @cts').index,:].copy()

# %%
gc.collect()

# %%
adata_hs.obs['cell_type_final']=adata_hs.obs['cell_type_integrated_subcluster']

 # %%
 # HVGs to supplement markers
    sc.pp.highly_variable_genes(
     adata=adata_hs, n_top_genes=2000, flavor='cell_ranger', batch_key='study_sample')

# %%
# Add gene symbols to adata
adata_hs.var['gene_symbol']=pd.read_table(path_hs_symbols,index_col=0,header=None
                          ).rename(
    {1:'gene_symbol'},axis=1).loc[adata_hs.var_names,'gene_symbol']

# %%
# markers + HVGs
hs_genes=list(set(
    [g for gs in markers.values() 
          for g in gs]) | set(adata_hs.var['gene_symbol'][adata_hs.var['highly_variable']]))

# %%
size_factors=adata_hs.obs['size_factors'].values[:,None]
# get raw expr to also have markers that were otherwise filtered out
adata_hs=adata_hs.raw.to_adata()
# Add gene symbols to adata (now raw)
adata_hs.var['gene_symbol']=pd.read_table(path_hs_symbols,index_col=0,header=None
                          ).rename(
    {1:'gene_symbol'},axis=1).loc[adata_hs.var_names,'gene_symbol']
adata_hs.X=csr_matrix(adata_hs.X)
del adata_hs.obsm
adata_hs=adata_hs[:,adata_hs.var.query('gene_symbol in @hs_genes').index].copy()
gc.collect()

# Normalise raw
adata_hs.X /=size_factors
sc.pp.log1p(adata_hs)
adata_hs.X=csr_matrix(adata_hs.X)

# %%
# Pseudobulk by sample and ct
x_pb=adata_hs.to_df()
group_cols=['study_sample','cell_type_final']
x_pb[group_cols]=adata_hs.obs[group_cols]
x_pb=x_pb.groupby(group_cols,observed=True).apply(
    lambda x:x.drop(group_cols,axis=1).mean(axis=0))

# %%
# Remove all 0 expressed genes
print('N all-zero genes:',(x_pb==0).all().sum())
x_pb=x_pb.loc[:,(x_pb>0).any()]
x_pb.shape

# %% [markdown]
# Save

# %%
scaler=MinMaxScaler().fit(x_pb)
x_pb_scl=pd.DataFrame(scaler.transform(x_pb),index=x_pb.index,columns=x_pb.columns)

# %%
x_pb_scl.to_csv(path_genes+'pseudobulkHS_pancreasEndo_markerHVG.tsv',sep='\t')
pd.DataFrame({'min':scaler.data_min_,'max':scaler.data_max_},index=x_pb.columns
            ).to_csv(path_genes+'pseudobulkHS_pancreasEndo_markerHVG_MinMaxScale.tsv',sep='\t')

# %%
