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
# Try linear decoder (gene embedding) model on endcrine data.

# %%
import scanpy as sc
import pickle as pkl
import pandas as pd
from scipy.sparse import csr_matrix
import numpy as np
import torch

from matplotlib.pyplot import rcParams
import matplotlib.pyplot as plt
import seaborn as sb

# %%
import sys
sys.path.insert(0, '/lustre/groups/ml01/code/karin.hrovatin/cross_species_prediction/')
import constraint_pancreas_example.model._xymodel as xym
import importlib
importlib.reload(xym)
from constraint_pancreas_example.model._xylinmodel import XYLinModel

# %%
path_data='/lustre/groups/ml01/workspace/karin.hrovatin/data/cross_species_prediction/pancreas_ct_example_constraint/'
path_genes='/lustre/groups/ml01/workspace/karin.hrovatin/data/cross_species_prediction/pancreas_ct_example_constraint/'


# %% [markdown]
# ## PCA embedding from Tabula Sapiens

# %%
adata=sc.read(path_data+'data_mm_hs.h5ad')
adata.shape

# %%
embed_y=pd.read_table(path_data+'pca_hs.tsv',index_col=0)
embed_y=embed_y.reindex(adata.var_names)
y_adata=adata.var.query('xy=="y"').index
no_embed=embed_y.loc[y_adata,:].isna().any(axis=1)
y_no_embed=set(y_adata[no_embed])
adata=adata[:,[v for v in adata.var_names if v not in y_no_embed]].copy()
adata.varm['embed']=embed_y.loc[adata.var_names,:].values
adata.shape

# %%
adata_training = XYLinModel.setup_anndata(
    adata,
    xy_key='xy',
    gene_embed_key='embed')
model = XYLinModel(adata=adata_training)

# %%
model.train(max_epochs=50)

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
# ## Pseudobulk from human pancreas data

# %%
adata=sc.read(path_data+'data_mm_hs.h5ad')
adata.shape

# %%
# Add embed
embed_y=pd.read_table(path_data+'pseudobulkHS_pancreasEndo.tsv',index_col=0)
embed_y=embed_y.T.reindex(adata.var_names)
y_adata=adata.var.query('xy=="y"').index
no_embed=embed_y.loc[y_adata,:].isna().any(axis=1)
y_no_embed=set(y_adata[no_embed])
adata=adata[:,[v for v in adata.var_names if v not in y_no_embed]].copy()
adata.varm['embed']=embed_y.loc[adata.var_names,:].values.astype('float')
adata.shape

# %%
# Scale human expression
minmax=pd.read_table(path_data+'pseudobulkHS_pancreasEndo_MinMaxScale.tsv', index_col=0)
minmax['dif']=minmax['max']-minmax['min']
x=adata[:,adata.var.xy=="y"].to_df()
minmax=minmax.loc[x.columns,:]
x=(x-minmax['min'].T)/minmax['dif']
adata.X=adata.X.todense()
adata[:,x.columns]=x.values
adata.X=csr_matrix(adata.X)

# %%
adata_training = XYLinModel.setup_anndata(
    adata,
    xy_key='xy',
    gene_embed_key='embed')
model = XYLinModel(adata=adata_training)

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
# C: It seems taht if the gene embedding is good it can learn the relationship on training data.

# %% [markdown]
# Plot embedding on predicted genes

# %%
# Embedding on predicted expr
y_training_copy=y_training.copy()
sc.pp.scale(y_training_copy)
sc.pp.pca(y_training_copy,15)
sc.pp.neighbors(y_training_copy, n_pcs=15)
sc.tl.umap(y_training_copy)

# %%
sc.pl.umap(y_training_copy,color=['cell_type_final','study_hs'])

# %% [markdown]
# C: Could the poor gamma/delta embedding (scattered points all over) be due to less markers? 
# ALPHA 187
# BETA 371
# DELTA 54
# GAMMA 86
#
# Is the effect of unbalanced markers on PCA or on prediction?

# %% [markdown]
# Expression of genes across cts, genes are ordered based on ct they mark.

# %%
markers_hs={ct:list(adata.var.query('gene_symbol in @gs').index)
            for ct,gs in pkl.load(open(path_genes+'endo_markers_set_hs.pkl','rb')).items()}

# %%
gene_order=[g for ct in ['alpha','beta','delta','gamma'] for g in markers_hs[ct] ]

# %%
rcParams['figure.figsize']=(20,3)
ax=y_training[y_training.obs.cell_type_final=='beta',gene_order].to_df().boxplot(whis=100)
ax.grid(False)

# %%
rcParams['figure.figsize']=(20,3)
ax=y_training[y_training.obs.cell_type_final=='alpha',gene_order].to_df().boxplot(whis=100)
ax.grid(False)

# %%
rcParams['figure.figsize']=(20,3)
ax=y_training[y_training.obs.cell_type_final=='delta',gene_order].to_df().boxplot(whis=100)
ax.grid(False)

# %%
rcParams['figure.figsize']=(20,3)
y_training[y_training.obs.cell_type_final=='gamma',gene_order].to_df().boxplot(whis=100)

# %% [markdown]
# C: It seems that gamma also do not have such striking marker up patterns? Maybe there is too little data for gamma and delta to be predicted - delta actually shows some upregulation of markers.

# %% [markdown]
# Try also UMAP based on embedding not decoded expression

# %%
y_embed = model.embed(
        adata=adata_training,
        indices=None,
        batch_size=None,
        as_numpy=True)

# %%
y_embed=sc.AnnData(y_embed,obs=adata_training.obs)

# %%
sc.pp.neighbors(y_embed, use_rep='X')
sc.tl.umap(y_embed)

# %%
sc.pl.umap(y_embed,color=['cell_type_final','study_hs'])

# %% [markdown]
# C: Also on embedding level it does not look good - scattered g/d. Maybe the problem could be little N genes for some cts.

# %% [markdown]
# ### Training without beta cells

# %%
adata_training_nb = XYLinModel.setup_anndata(
    adata[adata.obs.cell_type_final!='beta',:],
    xy_key='xy',
    gene_embed_key='embed')
model_nb = XYLinModel(adata=adata_training_nb)

# %%
model_nb.train(max_epochs=100)

# %%
y_training_nb = model_nb.translate(
        adata=adata_training, # Use now whole data for prediction, from above
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

# %% [markdown]
# C: It seems that beta markers are predicted equally accross all, but beta do not obtain other ct markers.

# %%
sc.pl.dotplot(y_training_nb, var_names=top_beta_genes,  gene_symbols='gene_symbol',
              groupby='cell_type_final',  standard_scale='var' )

# %% [markdown]
# C: Again cant predict beta, maybe can learn good position in latent for this ct? try with gamma as more similar to alpha.

# %% [markdown]
# Plot embedding on predicted expression

# %%
# Compute embedding on predicted expression
y_training_nb_copy=y_training_nb.copy()
sc.pp.scale(y_training_nb_copy)
sc.pp.pca(y_training_nb_copy,15)
sc.pp.neighbors(y_training_nb_copy, n_pcs=15)
sc.tl.umap(y_training_nb_copy)

# %%
sc.pl.umap(y_training_nb_copy,color=['cell_type_final','study_hs'])

# %% [markdown]
# ### Training without gamma
# May be more similar to alpha so may be easier to predict

# %%
adata_training_ng = XYLinModel.setup_anndata(
    adata[adata.obs.cell_type_final!='gamma',:],
    xy_key='xy',
    gene_embed_key='embed')
model_ng = XYLinModel(adata=adata_training_ng)

# %%
model_ng.train(max_epochs=100)

# %%
y_training_ng = model_ng.translate(
        adata=adata_training, # Use now whole data for prediction, from above
        indices=None,
        give_mean=True,
        batch_size=None,
        as_numpy=True)

# %%
y_training_ng=sc.AnnData(y_training_ng,
                         obs=adata_training.obs,var=adata_training.var.query('xy=="y"'))

# %%
sc.pl.dotplot(y_training_ng,var_names=markers_plot, gene_symbols='gene_symbol',
              groupby='cell_type_final',  standard_scale='var' )

# %%
sc.pl.dotplot(y_training_ng,var_names=markers_plot, gene_symbols='gene_symbol',
              groupby='cell_type_final' )

# %% [markdown]
# C: Also cant predict gamma - maybe cant learn cell embedding with that little data. Could have 2 decoders - one linear for human and one for mouse to learn cell embeding even when human cell is absent.

# %% [markdown]
# ## Pseudobulk from human pancreas data - HVG+markers
# Was computed before adding x reconstruction loss

# %% [markdown]
# # If this is rerun again it will be overwritten by model that also reconstructs x!

# %%
adata=sc.read(path_data+'data_mm_hs_markerHVG.h5ad')
adata.shape

# %%
# Add embed
embed_y=pd.read_table(path_data+'pseudobulkHS_pancreasEndo_markerHVG.tsv',index_col=0)
embed_y=embed_y.T.reindex(adata.var_names)
y_adata=adata.var.query('xy=="y"').index
no_embed=embed_y.loc[y_adata,:].isna().any(axis=1)
y_no_embed=set(y_adata[no_embed])
adata=adata[:,[v for v in adata.var_names if v not in y_no_embed]].copy()
adata.varm['embed']=embed_y.loc[adata.var_names,:].values.astype('float')
adata.shape

# %%
# Scale human expression
minmax=pd.read_table(path_data+'pseudobulkHS_pancreasEndo_markerHVG_MinMaxScale.tsv', index_col=0)
minmax['dif']=minmax['max']-minmax['min']
x=adata[:,adata.var.xy=="y"].to_df()
minmax=minmax.loc[x.columns,:]
x=(x-minmax['min'].T)/minmax['dif']
adata.X=adata.X.todense()
adata[:,x.columns]=x.values
adata.X=csr_matrix(adata.X)

# %%
adata_training = XYLinModel.setup_anndata(
    adata,
    xy_key='xy',
    gene_embed_key='embed')
model = XYLinModel(adata=adata_training)

# %%
model.train(max_epochs=100)

# %%
# Plot reconstruction loss
plt.plot(
    model.trainer.logger.history['reconstruction_loss_train'].index,
    model.trainer.logger.history['reconstruction_loss_train']['reconstruction_loss_train'])

# %% [markdown]
# C: Loss stops improving quickly

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
sc.pl.dotplot(y_training,var_names=markers_plot, gene_symbols='gene_symbol',
              groupby='cell_type_final', standard_scale='var' )

# %%
sc.pl.dotplot(y_training,var_names=markers_plot, gene_symbols='gene_symbol',
              groupby='cell_type_final')

# %% [markdown]
#  UMAP based on latent embedding from model

# %%
y_embed = model.embed(
        adata=adata_training,
        indices=None,
        batch_size=None,
        as_numpy=True)

# %%
y_embed=sc.AnnData(y_embed,obs=adata_training.obs)

# %%
sc.pp.neighbors(y_embed, use_rep='X')
sc.tl.umap(y_embed)

# %%
sc.pl.umap(y_embed,color=['cell_type_final','study_hs'])

# %% [markdown]
# C: Still can not predict delta and gamma embed well - scattered! Seems like whole region predicts them? TODO: look more in detail how their markers and genes are predicted
#
# C: increasing N epochs did not help - loss does not drop anyways

# %% [markdown]
# Look at MSE per ct

# %%
sse=pd.DataFrame({
    'sse':np.array(
        np.square(adata_training[:,adata_training.var.xy=='y'].X-y_training.X).sum(axis=1)).ravel(),
    'cell_type':adata_training.obs['cell_type_final']})

# %%
sb.boxplot(y='sse',x='cell_type',data=sse)
plt.yscale('log')

# %% [markdown]
# C: Seems that SSE is not worse in delta and gamma (but this does not accounts for std, marker/not etc)

# %% [markdown]
# Plot sse on umap - maybe certain regions arent predicted as well

# %%
y_embed.obs['sse']=sse['sse'].values
y_embed.obs['log_sse']=np.log(y_embed.obs['sse'])

# %%
sc.pl.umap(y_embed,color=['cell_type_final','log_sse'])

# %% [markdown]
# ## Pseudobulk from human pancreas data - HVG+markers
# Was run after adding x recon loss

# %% [markdown]
# # If this is rerun again it will be overwritten by model that also has kl loss!

# %%
adata=sc.read(path_data+'data_mm_hs_markerHVG.h5ad')
adata.shape

# %%
# Add embed
embed_y=pd.read_table(path_data+'pseudobulkHS_pancreasEndo_markerHVG.tsv',index_col=0)
embed_y=embed_y.T.reindex(adata.var_names)
y_adata=adata.var.query('xy=="y"').index
no_embed=embed_y.loc[y_adata,:].isna().any(axis=1)
y_no_embed=set(y_adata[no_embed])
adata=adata[:,[v for v in adata.var_names if v not in y_no_embed]].copy()
adata.varm['embed']=embed_y.loc[adata.var_names,:].values.astype('float')
adata.shape

# %%
# Scale human expression
minmax=pd.read_table(path_data+'pseudobulkHS_pancreasEndo_markerHVG_MinMaxScale.tsv', index_col=0)
minmax['dif']=minmax['max']-minmax['min']
x=adata[:,adata.var.xy=="y"].to_df()
minmax=minmax.loc[x.columns,:]
x=(x-minmax['min'].T)/minmax['dif']
adata.X=adata.X.todense()
adata[:,x.columns]=x.values
adata.X=csr_matrix(adata.X)

# %%
adata_training = XYLinModel.setup_anndata(
    adata,
    xy_key='xy',
    gene_embed_key='embed')
model = XYLinModel(adata=adata_training)

# %%
model.train(max_epochs=100)

# %%
# Plot reconstruction loss for x
plt.plot(
    model.trainer.logger.history['reconstruction_loss_x_train'].index,
    model.trainer.logger.history['reconstruction_loss_x_train']['reconstruction_loss_x_train'])

# %%
# Plot reconstruction loss for y
plt.plot(
    model.trainer.logger.history['reconstruction_loss_train'].index,
    model.trainer.logger.history['reconstruction_loss_train']['reconstruction_loss_train'])

# %% [markdown]
# C: It seems that x loss needs longer to improve than y loss. Maybe the constraint on y is too high so it cant optimise further anyways?

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
sc.pl.dotplot(y_training,var_names=markers_plot, gene_symbols='gene_symbol',
              groupby='cell_type_final', standard_scale='var' )

# %%
sc.pl.dotplot(y_training,var_names=markers_plot, gene_symbols='gene_symbol',
              groupby='cell_type_final')

# %% [markdown]
#  UMAP based on latent embedding from model

# %%
y_embed = model.embed(
        adata=adata_training,
        indices=None,
        batch_size=None,
        as_numpy=True)

# %%
y_embed=sc.AnnData(y_embed,obs=adata_training.obs)

# %%
sc.pp.neighbors(y_embed, use_rep='X')
sc.tl.umap(y_embed)

# %%
sc.pl.umap(y_embed,color=['cell_type_final','study_hs'])

# %% [markdown]
# C: Adding x reconstruction does not solve the problem of d/g scatter. The reason could be that that this is not an VAE but rather regression modell so there is no KL loss on the "latent" space.
#

# %% [markdown]
# ## Pseudobulk from human pancreas data - HVG+markers; KL loss
# Was run after adding x recon loss and kl loss

# %%
adata=sc.read(path_data+'data_mm_hs_markerHVG.h5ad')
adata.shape

# %%
# Add embed
embed_y=pd.read_table(path_data+'pseudobulkHS_pancreasEndo_markerHVG.tsv',index_col=0)
embed_y=embed_y.T.reindex(adata.var_names)
y_adata=adata.var.query('xy=="y"').index
no_embed=embed_y.loc[y_adata,:].isna().any(axis=1)
y_no_embed=set(y_adata[no_embed])
adata=adata[:,[v for v in adata.var_names if v not in y_no_embed]].copy()
adata.varm['embed']=embed_y.loc[adata.var_names,:].values.astype('float')
adata.shape

# %%
# Scale human expression
minmax=pd.read_table(path_data+'pseudobulkHS_pancreasEndo_markerHVG_MinMaxScale.tsv', index_col=0)
minmax['dif']=minmax['max']-minmax['min']
x=adata[:,adata.var.xy=="y"].to_df()
minmax=minmax.loc[x.columns,:]
x=(x-minmax['min'].T)/minmax['dif']
adata.X=adata.X.todense()
adata[:,x.columns]=x.values
adata.X=csr_matrix(adata.X)

# %%
adata_training = XYLinModel.setup_anndata(
    adata,
    xy_key='xy',
    gene_embed_key='embed')
model = XYLinModel(adata=adata_training)

# %%
model.train(max_epochs=100)

# %%
# Plot reconstruction loss for x
plt.plot(
    model.trainer.logger.history['reconstruction_loss_x_train'].index,
    model.trainer.logger.history['reconstruction_loss_x_train']['reconstruction_loss_x_train'])

# %%
# Plot reconstruction loss for y
plt.plot(
    model.trainer.logger.history['reconstruction_loss_train'].index,
    model.trainer.logger.history['reconstruction_loss_train']['reconstruction_loss_train'])

# %%
# Plot reconstruction loss for y
plt.plot(
    model.trainer.logger.history['kl_local_train'].index,
    model.trainer.logger.history['kl_local_train']['kl_local_train'])

# %% [markdown]
# C: It seems that kl loss is small - maybe think on adjusting it if needed. Also y decoding has longer training when adding kl loss.

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
sc.pl.dotplot(y_training,var_names=markers_plot, gene_symbols='gene_symbol',
              groupby='cell_type_final', standard_scale='var' )

# %%
sc.pl.dotplot(y_training,var_names=markers_plot, gene_symbols='gene_symbol',
              groupby='cell_type_final')

# %% [markdown]
#  UMAP based on latent embedding from model

# %%
y_embed = model.embed(
        adata=adata_training,
        indices=None,
        batch_size=None,
        as_numpy=True)

# %%
y_embed=sc.AnnData(y_embed,obs=adata_training.obs)

# %%
sc.pp.neighbors(y_embed, use_rep='X')
sc.tl.umap(y_embed)

# %%
sc.pl.umap(y_embed,color=['cell_type_final','study_hs','study_mm'])

# %% [markdown]
# C: Adding kl loss helps in making more defined cell type clusters, but also need to add batch covariates - in latent space it is mouse study and then in decoding mouse for x decoding. For y decoding (human) adding covariate should be also done not to affect embedding (it seems to affect it a bit, but less than mouse).
#

# %% [markdown]
# ## Pseudobulk from human pancreas data - HVG+markers; KL loss; add x covariates
# Was run after adding x recon and Kl loss

# %%
adata=sc.read(path_data+'data_mm_hs_markerHVG.h5ad')
adata.shape

# %%
# Add embed
embed_y=pd.read_table(path_data+'pseudobulkHS_pancreasEndo_markerHVG.tsv',index_col=0)
embed_y=embed_y.T.reindex(adata.var_names)
y_adata=adata.var.query('xy=="y"').index
no_embed=embed_y.loc[y_adata,:].isna().any(axis=1)
y_no_embed=set(y_adata[no_embed])
adata=adata[:,[v for v in adata.var_names if v not in y_no_embed]].copy()
adata.varm['embed']=embed_y.loc[adata.var_names,:].values.astype('float')
adata.shape

# %%
# Scale human expression
minmax=pd.read_table(path_data+'pseudobulkHS_pancreasEndo_markerHVG_MinMaxScale.tsv', index_col=0)
minmax['dif']=minmax['max']-minmax['min']
x=adata[:,adata.var.xy=="y"].to_df()
minmax=minmax.loc[x.columns,:]
x=(x-minmax['min'].T)/minmax['dif']
adata.X=adata.X.todense()
adata[:,x.columns]=x.values
adata.X=csr_matrix(adata.X)

# %%
# Use all cells for training y
adata.obs['train_y']=1

# %%
adata_training = XYLinModel.setup_anndata(
    adata,
    xy_key='xy',
    train_y_key='train_y',
    categorical_covariate_keys_x=['study_mm'],
    gene_embed_key='embed')
model = XYLinModel(adata=adata_training)

# %%
model.train(max_epochs=100)

# %%
# Plot reconstruction loss for x
plt.plot(
    model.trainer.logger.history['reconstruction_loss_x_train'].index,
    model.trainer.logger.history['reconstruction_loss_x_train']['reconstruction_loss_x_train'])

# %%
# Plot reconstruction loss for y
plt.plot(
    model.trainer.logger.history['reconstruction_loss_train'].index,
    model.trainer.logger.history['reconstruction_loss_train']['reconstruction_loss_train'])

# %%
# Plot reconstruction loss for y
plt.plot(
    model.trainer.logger.history['kl_local_train'].index,
    model.trainer.logger.history['kl_local_train']['kl_local_train'])

# %% [markdown]
# C: It seems that kl loss is small - maybe think on adjusting it if needed. Also y decoding has longer training when adding kl loss.

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
sc.pl.dotplot(y_training,var_names=markers_plot, gene_symbols='gene_symbol',
              groupby='cell_type_final', standard_scale='var' )

# %%
sc.pl.dotplot(y_training,var_names=markers_plot, gene_symbols='gene_symbol',
              groupby='cell_type_final')

# %% [markdown]
#  UMAP based on latent embedding from model

# %%
y_embed = model.embed(
        adata=adata_training,
        indices=None,
        batch_size=None,
        as_numpy=True)

# %%
y_embed=sc.AnnData(y_embed,obs=adata_training.obs)

# %%
sc.pp.neighbors(y_embed, use_rep='X')
sc.tl.umap(y_embed)

# %%
sc.pl.umap(y_embed,color=['cell_type_final','study_hs','study_mm'])

# %% [markdown]
# ### Do not train on beta

# %%
# Use cells excluing beta for training y
adata.obs['train_y_nb']=(adata.obs.cell_type_final!='beta').astype('int')

# %%
adata_training_nb = XYLinModel.setup_anndata(
    adata,
    xy_key='xy',
    train_y_key='train_y_nb',
    categorical_covariate_keys_x=['study_mm'],
    gene_embed_key='embed')
model_nb = XYLinModel(adata=adata_training_nb)

# %%
model_nb.train(max_epochs=100)

# %%
# Plot reconstruction loss for x
plt.plot(
    model_nb.trainer.logger.history['reconstruction_loss_x_train'].index,
    model_nb.trainer.logger.history['reconstruction_loss_x_train']['reconstruction_loss_x_train'])

# %%
# Plot reconstruction loss for y
plt.plot(
    model_nb.trainer.logger.history['reconstruction_loss_train'].index,
    model_nb.trainer.logger.history['reconstruction_loss_train']['reconstruction_loss_train'])

# %%
# Plot reconstruction loss for y
plt.plot(
    model_nb.trainer.logger.history['kl_local_train'].index,
    model_nb.trainer.logger.history['kl_local_train']['kl_local_train'])

# %% [markdown]
# C: It seems that kl loss is small - maybe think on adjusting it if needed. Also y decoding has longer training when adding kl loss.

# %%
y_training_nb = model_nb.translate(
        adata=adata_training_nb,
        indices=None,
        give_mean=True,
        batch_size=None,
        as_numpy=True)

# %%
y_training_nb=sc.AnnData(y_training_nb,obs=adata_training.obs,var=adata_training.var.query('xy=="y"'))

# %%
markers_plot={'beta':['INS','G6PC2','SLC30A8'],'alpha':['GCG','IRX2'],
              'gamma':['PPY','ID2'],'delta':['SST','LEPR']}

# %%
sc.pl.dotplot(y_training_nb,var_names=markers_plot, gene_symbols='gene_symbol',
              groupby='cell_type_final', standard_scale='var' )

# %%
sc.pl.dotplot(y_training_nb,var_names=markers_plot, gene_symbols='gene_symbol',
              groupby='cell_type_final')

# %% [markdown]
# C: Removing beta cells from training of y (keeping only for x) again does not work

# %% [markdown]
#  UMAP based on latent embedding from model

# %%
y_embed_nb = model_nb.embed(
        adata=adata_training_nb,
        indices=None,
        batch_size=None,
        as_numpy=True)

# %%
y_embed_nb=sc.AnnData(y_embed_nb,obs=adata_training.obs)

# %%
sc.pp.neighbors(y_embed_nb, use_rep='X')
sc.tl.umap(y_embed_nb)

# %%
sc.pl.umap(y_embed_nb,color=['cell_type_final','study_hs','study_mm'])

# %% [markdown]
# C: the embedding of beta cells is worse after removing them from training of y - potentially this also helps integration as batches in y are independent of batches in x and the z needs to decode both x and y.

# %% [markdown]
# Potential things to do:
# - Fill better latent space with known cells to better learn how z must be encoded (mixup, learning perturbations instead of cts as have more cells cloase by, more cts to learn on,  ...)

# %% [markdown]
# ### Would mixup help in predicting ct not used in training?
# Mixup alpha, delta, gamma

# %%
cts=['alpha','delta','gamma']
alpha=2
np.random.seed(0)
x=[]
obsm=[]
for i in range(len(cts)-1):
    for j in range(i+1,len(cts)):
        ct_i=cts[i]
        ct_j=cts[j]
        cells_i=list(adata_training.obs.query('cell_type_final==@ct_i').index)
        cells_j=list(adata_training.obs.query('cell_type_final==@ct_j').index)
        for c in range (1000):
            mixup_ratio_i = np.random.beta(alpha, alpha)
            mixup_ratio_j = 1 - mixup_ratio_i
            cell_i=np.random.choice(cells_i)
            cell_j=np.random.choice(cells_j)
            x.append((adata_training[cell_i,:].X*mixup_ratio_i +\
                     adata_training[cell_j,:].X*mixup_ratio_j).todense())
            obsm.append(adata_training[cell_i,:].obsm['covariates_x'].values*mixup_ratio_i +\
                        adata_training[cell_j,:].obsm['covariates_x'].values*mixup_ratio_j)
x=pd.DataFrame(np.concatenate(x),columns=adata_training.var_names)
obsm=np.concatenate(obsm).astype('float')

# %%
adata_training_nb_mixup=sc.concat([
    sc.AnnData(X=x,obs=pd.DataFrame({'train_y':[1]*x.shape[0]})),
    sc.AnnData(X=adata_training.to_df(),
               obs=adata_training_nb.obs['train_y_nb'].rename('train_y').to_frame())])
adata_training_nb_mixup.obsm['covariates_x']=np.concatenate([obsm,adata_training.obsm['covariates_x']])

# %%
adata_training_nb_mixup.uns['_scvi_uuid']=adata_training_nb.uns['_scvi_uuid']
adata_training_nb_mixup.uns['_scvi_manager_uuid']=adata_training_nb.uns['_scvi_manager_uuid']
adata_training_nb_mixup.uns['xsplit']=adata_training_nb.uns['xsplit']
adata_training_nb_mixup.var[['xy', 'mean', 'std']]=adata_training_nb.var[['xy', 'mean', 'std']]
adata_training_nb_mixup.varm['embed']=adata_training_nb.varm['embed']

# %%
# Circumvent data registration to work with already processed fields
from scvi.data.fields import ObsmField, NumericalObsField, LayerField
from scvi.data import AnnDataManager
setup_method_args = XYLinModel._get_setup_method_args(cls=XYLinModel)
anndata_fields = [
    LayerField('X', None, is_count_data=False),
    ObsmField('covariates_x', 'covariates_x'),
    NumericalObsField('train_y', 'train_y')
]
adata_manager = AnnDataManager(
    fields=anndata_fields, setup_method_args=setup_method_args
)
adata_manager.register_fields(adata_training_nb_mixup)
XYLinModel.register_manager(adata_manager)

# %%
model_nb_mixup = XYLinModel(adata=adata_training_nb_mixup)

# %%
model_nb_mixup.train(max_epochs=100)

# %%
# Plot reconstruction loss for x
plt.plot(
    model_nb_mixup.trainer.logger.history['reconstruction_loss_x_train'].index,
    model_nb_mixup.trainer.logger.history['reconstruction_loss_x_train']['reconstruction_loss_x_train'])

# %%
# Plot reconstruction loss for y
plt.plot(
    model_nb_mixup.trainer.logger.history['reconstruction_loss_train'].index,
    model_nb_mixup.trainer.logger.history['reconstruction_loss_train']['reconstruction_loss_train'])

# %%
# Plot reconstruction loss for y
plt.plot(
    model_nb_mixup.trainer.logger.history['kl_local_train'].index,
    model_nb_mixup.trainer.logger.history['kl_local_train']['kl_local_train'])

# %% [markdown]
# C: It seems that kl loss is small - maybe think on adjusting it if needed. Also y decoding has longer training when adding kl loss.

# %%
y_training_nb_mixup = model_nb_mixup.translate(
        adata=adata_training_nb_mixup,
        indices=None,
        give_mean=True,
        batch_size=None,
        as_numpy=True)

# %%
obs_nb_mixup=pd.concat([pd.DataFrame(np.zeros((3000,1))).replace(0,np.nan),adata_training.obs])

# %%
y_training_nb_mixup=sc.AnnData(y_training_nb_mixup,
                         obs=obs_nb_mixup, # TODO
                         var=adata_training.var.query('xy=="y"'))

# %%
markers_plot={'beta':['INS','G6PC2','SLC30A8'],'alpha':['GCG','IRX2'],
              'gamma':['PPY','ID2'],'delta':['SST','LEPR']}

# %%
sc.pl.dotplot(y_training_nb_mixup,var_names=markers_plot, gene_symbols='gene_symbol',
              groupby='cell_type_final', standard_scale='var' )

# %%
sc.pl.dotplot(y_training_nb_mixup,var_names=markers_plot, gene_symbols='gene_symbol',
              groupby='cell_type_final')

# %% [markdown]
# C: the above mixup wasnt enough

# %% [markdown]
#  UMAP based on latent embedding from model

# %%
y_embed_nb_mixup = model_nb_mixup.embed(
        adata=adata_training_nb_mixup,
        indices=None,
        batch_size=None,
        as_numpy=True)

# %%
y_embed_nb_mixup=sc.AnnData(y_embed_nb_mixup,obs=obs_nb_mixup) 

# %%
sc.pp.neighbors(y_embed_nb_mixup, use_rep='X')
sc.tl.umap(y_embed_nb_mixup)

# %%
y_embed_nb_mixup.obs_names=pd.Series(list(range(y_embed_nb_mixup.shape[0]))).astype(str)
y_embed_nb_mixup.obs['cell_type_final']=obs_nb_mixup['cell_type_final'].\
                                        astype(str).replace({'nan':'mixup'}).values

# %%
sc.pl.umap(y_embed_nb_mixup[
    [c for c,ct in zip(y_embed_nb_mixup.obs_names,y_embed_nb_mixup.obs.cell_type_final) 
     if ct!='mixup']+
    [c for c,ct in zip(y_embed_nb_mixup.obs_names,y_embed_nb_mixup.obs.cell_type_final) 
     if ct=='mixup'],:
],
           color=['cell_type_final','study_hs','study_mm'])

# %%
