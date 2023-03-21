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
import numpy as np
from collections import defaultdict
import glob
from scipy.io import mmread
from scipy.sparse import csr_matrix

import gc

from matplotlib import rcParams

# %%
path_data='/lustre/groups/ml01/workspace/karin.hrovatin/data/'
path_e=path_data+'datasets/d10_1038_s41422-021-00486-w/'
path_e_carina='/lustre/groups/ml01/workspace/karin.hrovatin/data/pigE_cross_species/datasets/hs/'
path_o=path_data+'datasets/pancreas_organoid/'

path_save=path_data+'cross_species_prediction/pancreas_organoid_embryo/'

# %% [markdown]
# ## Combined adatas

# %%
adata_o=sc.concat([adata_o5[adata_o5.obs['sample']=='S5_WT',:],
                   adata_o6[adata_o6.obs['sample']=='S6_WT',:]],
                  label='dataset',keys=['S5','S6'],index_unique='-')

# %%
# Organoid
adata_o6=sc.read(path_o+'scRNA-seq_iPSC_IIR-KO_S6_adata_rmDoublets_normalized_integrated_annotated.h5ad')
adata_o5=sc.read(path_o+'scRNA-seq_Integrated_S5_adata_rmDoublets_normalized_integrated_annotated.h5ad')
# Subset to samples that should be used
adata_o=sc.concat([adata_o5[adata_o5.obs['sample']=='S5_WT',:],
                   adata_o6[adata_o6.obs['sample']=='S6_WT',:]],
                  label='dataset',keys=['S5','S6'],index_unique='-')

# Metadata
adata_o.obs['system']=0
adata_o.obs['sample']=adata_o.obs['sample'].astype('category')
adata_o.obs['cell_type']=adata_o.obs['cell_type'].astype('category')
adata_o.obs=adata_o.obs[['system','sample','cell_type','dataset']]

# Subset to annotated cells
adata_o=adata_o[~adata_o.obs.cell_type.isna(),:]

# Subset to expr genes and normalise
adata_o.X=adata_o.layers['raw_counts']
adata_o=sc.AnnData(adata_o.X,obs=adata_o.obs,var=pd.DataFrame(index=adata_o.var_names))
# Retain only relevant info in adata
adata_o=adata_o[:,np.array((adata_o.X>0).sum(axis=0)>20).ravel()].copy()
adata_o.layers['counts']=adata_o.X.copy()
sc.pp.normalize_total(adata_o)
sc.pp.log1p(adata_o)
print('norm')
# HVG
sc.pp.highly_variable_genes(
     adata=adata_o, n_top_genes=3000, flavor='cell_ranger', batch_key='sample', subset=True)
display(adata_o)

# %%
# PP embryo
adata_e=sc.read(path_e+'adata.h5ad')
adata_e.obs['cell_type']=adata_e.obs['annotation_nodoub_2'].astype('category')
adata_e.obs['system']=1
adata_e.obs=adata_e.obs[['system','sample','cell_type']]

# Subset to annotated cells
adata_e=adata_e[~adata_e.obs.cell_type.isna(),:]

# Subset to expr genes and normalise
adata_e=adata_e[:,np.array((adata_e.X>0).sum(axis=0)>20).ravel()].copy()
adata_e.layers['counts']=adata_e.X.copy()
sc.pp.normalize_total(adata_e)
sc.pp.log1p(adata_e)
# HVG
sc.pp.highly_variable_genes(
     adata=adata_e, n_top_genes=3000, flavor='cell_ranger', batch_key='sample', subset=True)

display(adata_e)

# %%
# Shared HVGs
shared_hvgs=list(set(adata_e.var_names) & set(adata_o.var_names))
len(shared_hvgs)

# %%
for ds,data in adata_o.obs.groupby('dataset'):
    print(ds)
    print(sorted(data.cell_type.unique()))

# %%
sorted(adata_o.obs.cell_type.unique())

# %%
sorted(adata_e.obs.cell_type.unique())

# %%
adata_e.obs['cell_type_original']=adata_e.obs['cell_type']
adata_e.obs['cell_type']=adata_e.obs['cell_type'].replace({
 'Acinar':'acinar',
 'Alpha':'alpha',
 'Beta':'beta',
 'Delta':'delta',
 'Ductal':'epi_lin',
 'EP_0':'EP_pre',
 'EP_1':'EP_early',
 'Epsilon':'epsilon',
 'FEV':'FEV+',
 'MPC':'MPC',
 'PP':'gamma',
 'Pre_Alpha':'alpha_prog',
 'Pre_Beta':'beta_prog',
 'Pre_Delta':'delta_prog',
 'Pre_Epsilon':'epsilon_prog'
})

# %%
adata_o.obs['cell_type_original']=adata_o.obs['cell_type']
adata_o.obs['cell_type']=adata_o.obs['cell_type'].replace({
 'ARX+':'alpha_prog',
 'Alpha':'alpha',
 'Alpha (Cycling)':'alpha_cyc',
 'Alpha Prog. (ARX+)':'alpha_prog',
 'Alpha-Beta':'polyhoromonal',
 'Alpha-EC':'alpha_prog',
 'Beta':'beta',
 'Beta (ASCL1+)':'beta',
 'Beta (Cycling)':'beta_cyc',
 'Beta (GAP43+)':'beta',
 'Beta Prog.':'beta_prog',
 'Delta':'delta',
 'EC':'EC',
 'EC (Cycling)':'EC',
 'EC (GAP43+)':'EC',
 'EC (LDHA++)':'EC',
 'EC Prog.':'EC',
 'EC Prog. (ASCL1+)':'EC',
 'Early Prog.':'EP_early',
 'Early Prog. (Cycling)':'EP_early',
 'Epithelial':'epi_lin',
 'FEV+':'FEV+',
 'GAP43+':'beta_prog',
 'Mesenchymal':'mesenchymal',
 'Polyhormonal':'polyhoromonal',
 'Pre-Endocrine':'EP_pre',
 'Pre-Endocrine (Cycling)':'EP_pre',
 'Prog. (FEV+)':'FEV+'
})

# %%
sorted(adata_o.obs.cell_type.unique())

# %%
sorted(adata_e.obs.cell_type.unique())

# %%
# Subset to shraed HVGs and concat
adata=sc.concat([adata_o[:,shared_hvgs], adata_e[:,shared_hvgs]],
                join='outer',
                index_unique='_', keys=['organoid','embryo'])
adata

# %%
adata.write(path_save+'organoid_embryo_public-soon.h5ad')

# %%
