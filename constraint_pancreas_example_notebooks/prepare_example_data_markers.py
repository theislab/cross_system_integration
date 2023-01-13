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
# Prepare mouse and human endcorine pancreas data with markers as features for training of the model.

# %%
import scanpy as sc
import pandas as pd
import pickle as pkl
import gc
import numpy as np
from scipy.sparse import csr_matrix

# %%
path_mm='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/combined/'
fn_hs='/lustre/groups/ml01/workspace/eva.lavrencic/data/pancreas/combined/adata_integrated_log_normalised_reann.h5ad'
path_genes='/lustre/groups/ml01/workspace/karin.hrovatin/data/cross_species_prediction/pancreas_ct_example_constraint/'
path_hs_symbols='/lustre/groups/ml01/projects/2020_pancreas_karin.hrovatin/data/pancreas/scRNA/P21002/rev7/cellranger/MUC26033/count_matrices/filtered_feature_bc_matrix/features.tsv'

# %% [markdown]
# ## Markers + HVGs

# %% [markdown]
# ### Load and subset mouse and human data

# %%
adata_mm=sc.read(path_mm+'data_rawnorm_integrated_annotated.h5ad')
adata_mm.X=adata_mm.layers['X_sf_integrated']

# %%
# Subset to healthy adult
samples=['STZ_G1_control','VSG_MUC13633_chow_WT','VSG_MUC13634_chow_WT',
         'Fltp_adult_mouse1_head_Fltp-','Fltp_adult_mouse2_head_Fltp+',
         'Fltp_adult_mouse4_tail_Fltp+','Fltp_adult_mouse3_tail_Fltp-']
adata_mm=adata_mm[adata_mm.obs.query('study_sample_design in @samples').index,:].copy()

# %%
adata_mm.obs['cell_type_integrated_v1']=sc.read(
    path_mm+'data_integrated_analysed.h5ad',backed='r'
).obs.loc[adata_mm.obs_names,'cell_type_integrated_v1'].copy()

# %%
cts=['alpha','beta','gamma','delta']

# %%
# Get endo cts
adata_mm=adata_mm[adata_mm.obs.query('cell_type_integrated_v1 in @cts').index,:].copy()

# %%
adata_mm.obs['cell_type_final']=adata_mm.obs['cell_type_integrated_v1']

# %%
del adata_mm.layers
del adata_mm.obsm

# %%
# HVGs on the final cell subset
sc.pp.highly_variable_genes(
     adata=adata_mm, n_top_genes=2000, flavor='cell_ranger', batch_key='study_sample')

# %%
# Markers+HVGs
mm_genes=list(set(
    [g for gs in pkl.load(open(path_genes+'endo_markers_set_mm.pkl','rb')).values() 
          for g in gs])| set(adata_mm.var_names[adata_mm.var['highly_variable']]))
print(len(mm_genes))

# %%
# Subset genes
adata_mm=adata_mm[:,mm_genes].copy()

# %%
gc.collect()

# %%
adata_mm

# %%
adata_hs=sc.read(fn_hs)

# %%
# Healthy adult
adata_hs.obs['age_years']=adata_hs.obs['age'].str.replace(' y','').astype(float)
adata_hs=adata_hs[(adata_hs.obs['age_years']>=19) & (adata_hs.obs['age_years']<=64)  &\
    (adata_hs.obs['disease']=='healthy') & (adata_hs.obs['study']!='human_spikein_drug'),
                  :].copy()

# %%
# Endo cells
adata_hs=adata_hs[adata_hs.obs.query('cell_type_integrated_subcluster in @cts').index,:].copy()

# %%
gc.collect()

# %%
adata_hs.obs['cell_type_final']=adata_hs.obs['cell_type_integrated_subcluster']

# %%
sc.pp.highly_variable_genes(
     adata=adata_hs, n_top_genes=2000, flavor='cell_ranger', batch_key='study_sample')

# %%
# Add gene symbols to adata
adata_hs.var['gene_symbol']=pd.read_table(path_hs_symbols,index_col=0,header=None
                          ).rename(
    {1:'gene_symbol'},axis=1).loc[adata_hs.var_names,'gene_symbol']

# %%
# Markers + HVGs
hs_genes=list(set(
    [g for gs in pkl.load(open(path_genes+'endo_markers_set_hs.pkl','rb')).values() 
          for g in gs])| set(adata_hs.var['gene_symbol'][adata_hs.var['highly_variable']]))

# %%
size_factors=adata_hs.obs['size_factors'].values[:,None]
# Raw data as contains more genes
adata_hs=adata_hs.raw.to_adata()
adata_hs.X=csr_matrix(adata_hs.X)
# Add gene symbols to raw
adata_hs.var['gene_symbol']=pd.read_table(path_hs_symbols,index_col=0,header=None
                          ).rename({1:'gene_symbol'},axis=1).loc[adata_hs.var_names,'gene_symbol']

del adata_hs.obsm
# Normalise raw
adata_hs=adata_hs[:,adata_hs.var.query('gene_symbol in @hs_genes').index].copy()
gc.collect()
adata_hs.X /=size_factors
sc.pp.log1p(adata_hs)
adata_hs.X=csr_matrix(adata_hs.X)

# %%
adata_hs

# %% [markdown]
# ### Make random matched samples

# %%
adata_hs.obs.cell_type_final.value_counts()

# %%
adata_mm.obs.cell_type_final.value_counts()

# %%
pairs=set()
for ct in cts:
    pairs_sub=set()
    cells_mm=adata_mm.obs.query('cell_type_final==@ct').index.values
    cells_hs=adata_hs.obs.query('cell_type_final==@ct').index.values
    np.random.seed(0)
    while len(pairs_sub)<5000:
        pairs_sub.add((np.random.choice(cells_mm,1)[0],np.random.choice(cells_hs,1)[0]))
    pairs.update(pairs_sub)

# %%
cells_mm=np.array(list(pairs))[:,0]
obs=adata_mm[cells_mm,:].obs[['cell_type_final','study', 'study_sample', 'file']].copy()
obs['cell']=adata_mm[cells_mm,:].obs_names
obs.columns=[c+'_mm' if c!='cell_type_final' else c for c in obs.columns]
obs.index=range(obs.shape[0])
adata_mm_sub=sc.AnnData(adata_mm[cells_mm,:].X,obs=obs,var=adata_mm.var[['gene_symbol']])
adata_mm_sub.var['xy']='x'

cells_hs=np.array(list(pairs))[:,1]
obs=adata_hs[cells_hs,:].obs[['study', 'study_sample', 'file']].copy()
obs['cell']=adata_hs[cells_hs,:].obs_names
obs.columns=[c+'_hs' if c!='cell_type_final' else c for c in obs.columns]
obs.index=range(obs.shape[0])
adata_hs_sub=sc.AnnData(adata_hs[cells_hs,:].X,obs=obs,
                        var=adata_hs.var[['gene_symbol']])
adata_hs_sub.var['xy']='y'

# %%
adata_merged=sc.concat([adata_mm_sub,adata_hs_sub],axis=1,merge="unique")
adata_merged

# %%
adata_merged.write(path_genes+'data_mm_hs_markerHVG.h5ad')

# %% [markdown]
# ## Markers

# %% [markdown]
# ### Load and subset mouse and human data

# %%
adata_mm=sc.read(path_mm+'data_rawnorm_integrated_annotated.h5ad')
adata_mm.X=adata_mm.layers['X_sf_integrated']

# %%
mm_genes=list(set(
    [g for gs in pkl.load(open(path_genes+'endo_markers_set_mm.pkl','rb')).values() 
          for g in gs]))

# %%
del adata_mm.layers
del adata_mm.obsm
adata_mm=adata_mm[:,mm_genes].copy()

# %%
# Subset to healthy adult
samples=['STZ_G1_control','VSG_MUC13633_chow_WT','VSG_MUC13634_chow_WT',
         'Fltp_adult_mouse1_head_Fltp-','Fltp_adult_mouse2_head_Fltp+',
         'Fltp_adult_mouse4_tail_Fltp+','Fltp_adult_mouse3_tail_Fltp-']
adata_mm=adata_mm[adata_mm.obs.query('study_sample_design in @samples').index,:].copy()

# %%
adata_mm.obs['cell_type_integrated_v1']=sc.read(
    path_mm+'data_integrated_analysed.h5ad',backed='r'
).obs.loc[adata_mm.obs_names,'cell_type_integrated_v1'].copy()

# %%
cts=['alpha','beta','gamma','delta']

# %%
adata_mm=adata_mm[adata_mm.obs.query('cell_type_integrated_v1 in @cts').index,:].copy()

# %%
adata_mm.obs['cell_type_final']=adata_mm.obs['cell_type_integrated_v1']

# %%
gc.collect()

# %%
adata_mm

# %%
hs_genes=list(set(
    [g for gs in pkl.load(open(path_genes+'endo_markers_set_hs.pkl','rb')).values() 
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
adata_hs=adata_hs[adata_hs.obs.query('cell_type_integrated_subcluster in @cts').index,:].copy()

# %%
gc.collect()

# %%
adata_hs.obs['cell_type_final']=adata_hs.obs['cell_type_integrated_subcluster']

# %%
adata_hs

# %% [markdown]
# ### Make random matched samples

# %%
adata_hs.obs.cell_type_final.value_counts()

# %%
adata_mm.obs.cell_type_final.value_counts()

# %%
pairs=set()
for ct in cts:
    pairs_sub=set()
    cells_mm=adata_mm.obs.query('cell_type_final==@ct').index.values
    cells_hs=adata_hs.obs.query('cell_type_final==@ct').index.values
    np.random.seed(0)
    while len(pairs_sub)<5000:
        pairs_sub.add((np.random.choice(cells_mm,1)[0],np.random.choice(cells_hs,1)[0]))
    pairs.update(pairs_sub)

# %%
adata_mm[np.array(list(pairs))[:,0],:].X

# %%
cells_mm=np.array(list(pairs))[:,0]
obs=adata_mm[cells_mm,:].obs[['cell_type_final','study', 'study_sample', 'file']].copy()
obs['cell']=adata_mm[cells_mm,:].obs_names
obs.columns=[c+'_mm' if c!='cell_type_final' else c for c in obs.columns]
obs.index=range(obs.shape[0])
adata_mm_sub=sc.AnnData(adata_mm[cells_mm,:].X,obs=obs,var=adata_mm.var[['gene_symbol']])
adata_mm_sub.var['xy']='x'

cells_hs=np.array(list(pairs))[:,1]
obs=adata_hs[cells_hs,:].obs[['study', 'study_sample', 'file']].copy()
obs['cell']=adata_hs[cells_hs,:].obs_names
obs.columns=[c+'_hs' if c!='cell_type_final' else c for c in obs.columns]
obs.index=range(obs.shape[0])
adata_hs_sub=sc.AnnData(adata_hs[cells_hs,:].X,obs=obs,
                        var=adata_hs.var[['gene_symbol']])
adata_hs_sub.var['xy']='y'

# %%
adata_merged=sc.concat([adata_mm_sub,adata_hs_sub],axis=1,merge="unique")
adata_merged

# %%
adata_merged.write(path_genes+'data_mm_hs.h5ad')
