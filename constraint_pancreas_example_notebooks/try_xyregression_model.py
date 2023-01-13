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
# Try x-> y model and the regression constraint loss on endcorine data.

# %%
import scanpy as sc
import pickle as pkl
import pandas as pd

# %%
import sys
sys.path.insert(0, '/lustre/groups/ml01/code/karin.hrovatin/cross_species_prediction/')
import constraint_pancreas_example.model._xymodel as xym
import importlib
importlib.reload(xym)
from constraint_pancreas_example.model._xymodel import XYModel

# %%
path_data='/lustre/groups/ml01/workspace/karin.hrovatin/data/cross_species_prediction/pancreas_ct_example_constraint/'

# %%
markers=pkl.load(open(path_data+'endo_markers_set_hs.pkl','rb'))

# %%
constraints=pd.read_table(path_data+'coefs_hs.tsv')

# %%
adata=sc.read(path_data+'data_mm_hs.h5ad')

# %%
adata_training = XYModel.setup_anndata(
    adata,
    xy_key='xy')
model = XYModel(adata=adata_training)

# %%
model.train(max_epochs=100)

# %%
y_training = model.translate(
        adata=adata_training,
        indices=None,
        give_mean=True,
        batch_size=None,
        as_numpy=True)

# %%
y_training=sc.AnnData(y_training,obs=adata_training.obs,var=adata_training.var.query('xy=="y"'))

# %%
markers_plot={'beta':['INS','G6PC2','SLC30A8'],'alpha':['GCG','IRX2'],
              'gamma':['PPY','ID2'],'delta':['SST','LEPR']}

# %%
sc.pl.dotplot(adata_training,var_names=markers_plot, gene_symbols='gene_symbol',
              groupby='cell_type_final', standard_scale='var')

# %%
sc.pl.dotplot(y_training,var_names=markers_plot, gene_symbols='gene_symbol',
              groupby='cell_type_final', standard_scale='var' )

# %%
sc.pl.dotplot(y_training,var_names=markers_plot, gene_symbols='gene_symbol',
              groupby='cell_type_final')

# %%
top_beta_genes=['INS', 'IAPP', 'G6PC2', 'ADCYAP1', 'ERO1B', 'DLK1', 'NPTX2', 
                 'GSN', 'INS-IGF2', 'HADH', 'IGF2', 'SURF4', 'TSPAN1', 'PEBP1',
                 'PFKFB2', 'SLC30A8', 'TSPAN13', 'ABCC8', 'SCGN', 'FAM105A', 'SLC6A6', 
                 'CDKN1C', 'ENO1', 'PCSK1', 'TIMP2', 'SORL1', 'RPL3', 'EIF4A2', 'FAM159B', 
                 'PLCXD3', 'CYYR1', 'SLC39A14', 'SCG3', 'RBP4', 'SYT13', 'PDX1', 'VEGFA', 
                 'SAMD11', 'ENTPD3', 'SCD', 'C1orf127', 'ELMO1', 'DNAJB9', 'RGS16', 'RPS3', 
                 'RPS23', 'RPS6', 'LDHB']
ga=set(adata.var.query('xy=="y"').gene_symbol)
top_beta_genes=[g for g in top_beta_genes if g in ga]

# %%
# Top beta genes (copied from human table) that are in the current adata (pre-filtered)
sc.pl.dotplot(adata_training, var_names=top_beta_genes, gene_symbols='gene_symbol',
              groupby='cell_type_final',  standard_scale='var')

# %%
# Top beta genes (copied from human table) that are in the current adata (pre-filtered)
sc.pl.dotplot(y_training, var_names=top_beta_genes,  gene_symbols='gene_symbol',
              groupby='cell_type_final',  standard_scale='var' )

# %% [markdown]
# C: Marker prediction makes sense.
#
# C: Hormone genes are missing from this adata. 

# %% [markdown]
# Try the same but without beta cells.

# %%
adata_training_nb = XYModel.setup_anndata(
    adata[adata.obs.cell_type_final!='beta',:],
    xy_key='xy')
model_nb = XYModel(adata=adata_training_nb)

# %%
model_nb.train(max_epochs=100)

# %%
y_training_nb = model_nb.translate(
        adata=adata_training,
        indices=None,
        give_mean=True,
        batch_size=None,
        as_numpy=True)

# %%
y_training_nb=sc.AnnData(y_training_nb,
                         obs=adata_training.obs,var=adata_training.var.query('xy=="y"'))

# %%
sc.pl.dotplot(y_training_nb,var_names=markers_plot, gene_symbols='gene_symbol',
              groupby='cell_type_final',  standard_scale='var' )

# %%
sc.pl.dotplot(y_training_nb,var_names=markers_plot, gene_symbols='gene_symbol',
              groupby='cell_type_final' )

# %%
sc.pl.dotplot(y_training_nb, var_names=top_beta_genes,  gene_symbols='gene_symbol',
              groupby='cell_type_final',  standard_scale='var' )

# %% [markdown]
# C: If beta cells are not included in training it can not predict them

# %%
adata.uns['y_corr']=constraints

# %%
adata_training_nb_c = XYModel.setup_anndata(
    adata[adata.obs.cell_type_final!='beta',:],
    xy_key='xy',
y_corr_key='y_corr')
model_nb_c = XYModel(adata=adata_training_nb_c)

# %%
model_nb_c.train(max_epochs=100)

# %%
y_training_nb_c = model_nb_c.translate(
        adata=adata_training,
        indices=None,
        give_mean=True,
        batch_size=None,
        as_numpy=True)

# %%
y_training_nb_c=sc.AnnData(y_training_nb_c,
                         obs=adata_training.obs,var=adata_training.var.query('xy=="y"'))

# %%
sc.pl.dotplot(y_training_nb_c,var_names=markers_plot, gene_symbols='gene_symbol',
              groupby='cell_type_final',  standard_scale='var' )

# %%
sc.pl.dotplot(y_training_nb_c, var_names=top_beta_genes,  gene_symbols='gene_symbol',
              groupby='cell_type_final',  standard_scale='var' )


# %%
def gs_to_eid(gs):
    return adata.var.query('gene_symbol==@gs').index[0]


# %%
adata.uns['y_corr_fake']=pd.DataFrame([
    {'gx':gs_to_eid('INS'),'gy':gs_to_eid('GCG'),'coef':-1,'intercept':6},
    {'gx':gs_to_eid('INS'),'gy':gs_to_eid('IRX2'),'coef':-1,'intercept':1},
    {'gx':gs_to_eid('INS'),'gy':gs_to_eid('SST'),'coef':-1,'intercept':6},
    {'gx':gs_to_eid('INS'),'gy':gs_to_eid('LEPR'),'coef':-1,'intercept':1},
    {'gx':gs_to_eid('INS'),'gy':gs_to_eid('PPY'),'coef':-1,'intercept':6},
    {'gx':gs_to_eid('INS'),'gy':gs_to_eid('ID2'),'coef':-1,'intercept':3},
    {'gx':gs_to_eid('INS'),'gy':gs_to_eid('PEBP1'),'coef':1,'intercept':2},
    {'gx':gs_to_eid('INS'),'gy':gs_to_eid('ERO1B'),'coef':1,'intercept':2},
    {'gx':gs_to_eid('INS'),'gy':gs_to_eid('SLC30A8'),'coef':1,'intercept':2},
    {'gx':gs_to_eid('INS'),'gy':gs_to_eid('HADH'),'coef':1,'intercept':2},
    {'gx':gs_to_eid('INS'),'gy':gs_to_eid('PDX1'),'coef':1,'intercept':2},
])

# %%
adata_training_nb_c_f = XYModel.setup_anndata(
    adata[adata.obs.cell_type_final!='beta',:],
    xy_key='xy',
    y_corr_key='y_corr_fake')
model_nb_c_f = XYModel(adata=adata_training_nb_c_f,constraint_weight=10)

# %%
model_nb_c_f.train(max_epochs=100)

# %%
y_training_nb_c_f = model_nb_c_f.translate(
        adata=adata_training,
        indices=None,
        give_mean=True,
        batch_size=None,
        as_numpy=True)

# %%
y_training_nb_c_f=sc.AnnData(y_training_nb_c_f,
                         obs=adata_training.obs,var=adata_training.var.query('xy=="y"'))

# %%
sc.pl.dotplot(y_training_nb_c_f,var_names=markers_plot, gene_symbols='gene_symbol',
              groupby='cell_type_final',  standard_scale='var' )

# %%
