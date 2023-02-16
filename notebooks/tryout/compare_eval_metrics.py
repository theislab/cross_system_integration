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
# # Possible eval metrics
# Try different eval metrics for comparing prediction and real data:
# - correlation between mean of real and predicted data (and mean over genes)
# - gaussian LL between predicted (=x) and real( =distn params) as mean over genes and then cells
# - only for ref cells: gaussian LL between real (=x) and predicted ( =distn params) as mean over genes and then cells (as in query dont have matching between cells in prediction - comming from ref - and real cells); for this also the covariates are taken into account
# - MAE (mean abs err) as mean over genes and then cells; expected not to be ok due to gene outliers/expression strength

# %%
# Data pp imports
import scanpy as sc
import pickle as pkl
import pandas as pd
from scipy.sparse import csr_matrix, find
import numpy as np
from sklearn.preprocessing import minmax_scale
from scipy.stats import entropy
from sklearn.cluster import MiniBatchKMeans

from scipy.stats import norm

import gc

from matplotlib.pyplot import rcParams
import matplotlib.pyplot as plt
import seaborn as sb

# %%
# Modelling imports
import torch

import sys
sys.path.insert(0, '/lustre/groups/ml01/code/karin.hrovatin/cross_species_prediction/')
import constraint_pancreas_example.model._xxjointmodel as xxjm
import importlib
importlib.reload(xxjm)
from constraint_pancreas_example.model._xxjointmodel import XXJointModel

# %%
path_data='/lustre/groups/ml01/workspace/karin.hrovatin/data/cross_species_prediction/pancreas_example/'

# %%
adata=sc.read(path_data+'combined_orthologues.h5ad')

# %% [markdown]
# ### Metrics computation helper

# %%
# Cell used for translation in ref system and ground truth in query system
eval_cells_query=''
cells_ref=list(adata.obs.query(
    f'cell_type_final=="beta" &'+ 
    f'system!=1'+eval_cells_query).index)
cells_query=list(adata.obs.query(
    f'cell_type_final=="beta" &'+ 
    f'system==1'+eval_cells_query).index)
print(f'N cells ref: {len(cells_ref)} and query: {len(cells_query)}')

# %%
# Prediction metrics

# True expression summary
mean_ref=adata[cells_ref,:].to_df().mean()
mean_ref.name='ref'
std_ref=adata[cells_ref,:].to_df().std()
std_ref.name='ref'
mean_query=adata[cells_query,:].to_df().mean()
mean_query.name='query'
std_query=adata[cells_query,:].to_df().std()
std_query.name='query'

def metrics():
    pred=[]
    for translation_type, switch in {'ref':False,
                                     'query':True}.items():
        pred.append(
            sc.AnnData(
                model.translate(
                    adata= adata_training[cells_ref,:],
                    switch_system = switch,
                    covariates = None),
                obs=pd.DataFrame({
                    #'translation':[translation_type]*len(cells_ref),
                    'cells_ref':cells_ref,
                    'meta':['pred_'+str(translation_type)]*len(cells_ref)
                }),
                var=adata_training.var
            )
        )
    # Concat to single adata
    pred=sc.concat(pred)

    # Mean expression per prediction group
    mean_pred=pred.to_df()
    mean_pred['meta']=pred.obs['meta']
    mean_pred=mean_pred.groupby('meta').mean()

    # Correlation of predicted mean and real mean
    corr={}
    x=pd.concat([mean_pred.T,mean_ref,mean_query],axis=1).T
    x=pd.DataFrame(np.corrcoef(x),index=x.index,columns=x.index)
    for pred_group in ['pred_ref','pred_query']:
        for group in ['ref','query']:
            corr[group+'-'+pred_group]=x.loc[group,pred_group]
    #display(corr)

    # Gaussian ll of predicted mean vs distn on real cells
    ll={}
    for pred_group in ['pred_ref','pred_query']:
        for group,mean,std in [('ref',mean_ref,std_ref),('query',mean_query,std_query)]:
            ll[group+'-'+pred_group]=np.nanmean(norm.logpdf(
                pred[pred.obs.meta==pred_group,:].to_df(),
                loc=pd.DataFrame(mean).T, scale=pd.DataFrame(std).T),axis=1).mean()
    #display(ll)

    # MAE of predicted and real mean
    mae={}
    for pred_group in ['pred_ref','pred_query']:
        for group,mean,std in [('ref',mean_ref,std_ref),('query',mean_query,std_query)]:
            mae[group+'-'+pred_group]=abs(
                pred[pred.obs.meta==pred_group,:].to_df()-mean).mean(axis=1).mean()
    #display(mae)

    # Translate ref cells with matching covariates
    mean_ref_pred_cells,var_ref_pred_cells=model.translate(
                adata= adata_training[cells_ref,:],
                switch_system = False,
                covariates = adata[cells_ref,:].obs,
                give_mean=True,
                give_var=True,
            )
    mean_ref_pred_cells=pd.DataFrame(
        mean_ref_pred_cells,columns=adata_training.var_names,index=cells_ref)
    var_ref_pred_cells=pd.DataFrame(
        var_ref_pred_cells,columns=adata_training.var_names,index=cells_ref)

    # Correlation between translated and real cells, per cell 
    def corr_samples(x,y):
        corr=[]
        for i in range(x.shape[0]):
            corr.append(np.corrcoef(x[i,:],y[i,:])[0,1])
        return np.array(corr)
    corr_ref_cells={'ref-pred_ref':
            corr_samples(mean_ref_pred_cells.values,adata_training[cells_ref,:].to_df().values
                        ).mean()}
    #display(corr_ref_cells)

    # Gaussian LL of ref cells (x) vs their prediction (distn)
    ll_ref_cells={
        'ref-pred_ref':np.nanmean(norm.logpdf(
                adata_training[cells_ref,:].to_df(),
                loc=mean_ref_pred_cells, scale=np.sqrt(var_ref_pred_cells)),axis=1).mean()
    }
    #display(ll_ref_cells)

    mae_ref_cells={'ref-pred_ref':
            abs(mean_ref_pred_cells-adata_training[cells_ref,:].to_df()).mean(axis=1).mean()}
    #display(mae_ref_cells)
    
    return dict(corr=corr,
                ll=ll,
                mae=mae,
                corr_ref_cells=corr_ref_cells,
                ll_ref_cell=ll_ref_cells, 
                mae_ref_cells=mae_ref_cells)


# %% [markdown]
# Consideration on all 0 genes:
# - Problem for LL as 0 var, options: namean (logpdf=nan so dont use in mean), eps clipping of var - problem as can lead to large value affecting the overall mean, dont use for eval - problem as have different set of genes in ref and query then&how to decide which to use

# %% [markdown]
# ### Train

# %%
train_filter=(adata.obs['cell_type_final']!='beta').values |\
             (adata.obs['system']!=1).values
adata_training = XXJointModel.setup_anndata(
    adata=adata[train_filter,:],
    system_key='system',
    #class_key='cell_type_final',
    categorical_covariate_keys=['study_sample'],
)

# %%
model = XXJointModel(adata=adata_training)
metrics_data=[]
for i in range (20):
    model.train(max_epochs=1,
               plan_kwargs={'loss_weights':{
                   'reconstruction_mixup_weight':0,
                   'reconstruction_cycle_weight':0,
                   'kl_cycle_weight':0,
                   'z_distance_cycle_weight':0,
                   'translation_corr_weight':0,
                   'z_contrastive_weight':0,

               }})
    metrics_data.append(metrics())

# %%
# Make seaborn plotting data
metrics_plot=[]
for epoch, data in enumerate(metrics_data):
    for metric,data_metric in data.items():
        for comparison,val in data_metric.items():
            metrics_plot.append(dict(
                epoch=epoch,metric=metric,comparison=comparison,value=val))
metrics_plot=pd.DataFrame(metrics_plot)

# %%
sb.relplot(
    data=metrics_plot,
    x="epoch", y="value",
    row="comparison",col="metric",
    kind="line", 
    height=2.5, aspect=1,
    facet_kws=dict(margin_titles=True,sharey=False)
)

# %% [markdown]
# C: As expected, the metric between query-ref pairs are worse than between q-q and r-r. 
#
# C: Correl does not improve accross epochs. - maybe not sensitive enough? - e.g. already good after 1st epoch
#
# C: LL may be able to detect finer changes OR it corresponds more to training as this is also used for optimisation.
#
# C: ref-ref prediction is beter than query for corr and mae, but not ll.
#
# C: On sample level corr works better than on group lvl. LL is anyway similar to loss

# %%
