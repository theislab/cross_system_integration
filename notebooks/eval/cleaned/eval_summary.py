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
import glob
import pickle as pkl
import os
import itertools

from sklearn.preprocessing import minmax_scale

from matplotlib import rcParams
import matplotlib.pyplot as plt
import seaborn as sb
import matplotlib.image as mpimg
import matplotlib.colors as mcolors

# %%
path_eval='/om2/user/khrovati/data/cross_system_integration/eval/'


# %%
def metrics_data_heatmap(metrics_data,res,metric='jaccard_label'):
    metric='jaccard_label'
    dat={}
    for metrics_res in metrics_data:
        dat[metrics_res['name']]=metrics_res[metric]
    dat=pd.DataFrame(dat).T
    dat['params_opt']=dat.index.map(lambda x: res.at[x,'params_opt'])
    dat['param_opt_val']=dat.index.map(lambda x: res.at[x,'param_opt_val'])
    dat=dat.groupby(['params_opt','param_opt_val']).mean()
    dat.index=dat.index.map(lambda x: '-'.join([str(i) for i in x]))
    dat=dat/np.clip(dat.max(axis=0),a_min=1e-3, a_max=None)
    sb.clustermap(dat,row_cluster=False, xticklabels=True,yticklabels=True)


# %% [markdown]
# ## Pancreas conditions MIA HPAP2

# %%
path_integration=path_eval+'pancreas_conditions_MIA_HPAP2/integration/'

# %%
# Load integration results - params and metrics
res=[]
metrics_data=[]
for run in glob.glob(path_integration+'*/'):
    if os.path.exists(run+'args.pkl') and \
        os.path.exists(run+'scib_metrics.pkl') and \
        os.path.exists(run+'scib_metrics_scaled.pkl') and\
        os.path.exists(run+'scib_metrics_data.pkl'):
        args=pd.Series(vars(pkl.load(open(run+'args.pkl','rb'))))
        metrics=pd.Series(pkl.load(open(run+'scib_metrics.pkl','rb')))
        metrics_scl=pd.Series(pkl.load(open(run+'scib_metrics_scaled.pkl','rb')))
        metrics_scl.index=metrics_scl.index.map(lambda x: x+'_scaled')
        data=pd.concat([args,metrics,metrics_scl])
        name=run.split('/')[-2]
        data.name=name
        res.append(data)
        metrics_data_sub=pkl.load(open(run+'scib_metrics_data.pkl','rb'))
        metrics_data_sub['name']=name
        metrics_data.append(metrics_data_sub)
res=pd.concat(res,axis=1).T

# %%
#  Param that was optimised
res['params_opt']=res.params_opt.replace(
    {
     'scglue_no_lam_graph':'scglue_lam_graph_no',
     'scglue_no_rel_gene_weight':'scglue_rel_gene_weight_no', 
     'scglue_no_lam_align':'scglue_lam_align_no',
     'saturn_no_pe_sim_penalty':'saturn_pe_sim_penalty_no'})
res['param_opt_col']=res.params_opt.replace(
    {'vamp':'n_prior_components',
     'z_distance_cycle_weight_std':'z_distance_cycle_weight',
     'scglue_lam_graph':'lam_graph',
     'scglue_rel_gene_weight':'rel_gene_weight', 
     'scglue_lam_align':'lam_align',
     'scglue_lam_graph_no':'lam_graph',
     'scglue_rel_gene_weight_no':'rel_gene_weight', 
     'scglue_lam_align_no':'lam_align',
     'saturn_pe_sim_penalty':'pe_sim_penalty',
     'saturn_pe_sim_penalty_no':'pe_sim_penalty',
     
     'scvi':None})
res['param_opt_val']=res.apply(
    lambda x: x[x['param_opt_col']] if x['param_opt_col'] is not None else 0,axis=1)

# %%
res['params_opt']=pd.Categorical(res['params_opt'],sorted(res['params_opt'].unique()), True)

# %%
g=sb.catplot( x='param_opt_val', y="asw_group",  col='params_opt',
           kind="swarm", data=res,sharex=False,
          height=2.5,aspect=1.3,color='k')
for ax in g.axes.ravel():
    ax.axhline(0.56,lw=0.5,c='gray')

# %%
g=sb.catplot( x='param_opt_val', y="nmi",  col='params_opt',
           kind="swarm", data=res,sharex=False,
          height=2.5,aspect=1.3,color='k')
for ax in g.axes.ravel():
    ax.axhline(0.90,lw=0.5,c='gray')
    ax.axhline(0.85,lw=0.5,c='gray')
    ax.axhline(0.80,lw=0.5,c='gray')

# %%
g=sb.catplot( x='param_opt_val', y="nmi_scaled",  col='params_opt',
           kind="swarm", data=res,sharex=False,
          height=2.5,aspect=1.3,color='k')
for ax in g.axes.ravel():
    ax.axhline(0.90,lw=0.5,c='gray')
    ax.axhline(0.85,lw=0.5,c='gray')
    ax.axhline(0.80,lw=0.5,c='gray')

# %%
g=sb.catplot( x='param_opt_val', y="ari",  col='params_opt',
           kind="swarm", data=res,sharex=False,
          height=2.5,aspect=1.3,color='k')

# %%
g=sb.catplot( x='param_opt_val', y="jaccard",  col='params_opt',
           kind="swarm", data=res,sharex=False,
          height=2.5,aspect=1.3,color='k')

# %%
g=sb.catplot( x='param_opt_val', y="jaccard_macro",  col='params_opt',
           kind="swarm", data=res,sharex=False,
          height=2.5,aspect=1.3,color='k')

# %%
metrics_data_heatmap(metrics_data=metrics_data,res=res,metric='jaccard_label')

# %% [markdown]
# C: For example scglue_cXxtBJm8 has mixing of acinar and immune.

# %%
g=sb.catplot( x='param_opt_val', y="clisi",  col='params_opt',
           kind="swarm", data=res,sharex=False,
          height=2.5,aspect=1.3,color='k')

# %%
g=sb.catplot( x='param_opt_val', y="moransi",  col='params_opt',
           kind="swarm", data=res,sharex=False,
          height=2.5,aspect=1.3,color='k')
for ax in g.axes.ravel():
    ax.axhline(0.90,lw=0.5,c='gray')
    ax.axhline(0.93,lw=0.5,c='gray')
    ax.axhline(0.96,lw=0.5,c='gray')

# %%
g=sb.catplot( x='param_opt_val', y="moransi_scaled",  col='params_opt',
           kind="swarm", data=res,sharex=False,
          height=2.5,aspect=1.3,color='k')
for ax in g.axes.ravel():
    ax.axhline(0.90,lw=0.5,c='gray')
    ax.axhline(0.93,lw=0.5,c='gray')
    ax.axhline(0.96,lw=0.5,c='gray')

# %%
g=sb.catplot( x='param_opt_val', y="ilisi_system",  col='params_opt',
           kind="swarm", data=res,sharex=False,
          height=2.5,aspect=1.3,color='k')
for ax in g.axes.ravel():
    ax.axhline(0.03,lw=0.5,c='gray')

# %%
g=sb.catplot( x='param_opt_val', y="ilisi_system_scaled",  col='params_opt',
           kind="swarm", data=res,sharex=False,
          height=2.5,aspect=1.3,color='k')
for ax in g.axes.ravel():
    ax.axhline(0.03,lw=0.5,c='gray')

# %% [markdown]
# C: Scalling may affect VAMP differently as the prior is not N(0,1) in the first place so maybe different features really have different importance. - Not just pushed to low values to satisfy prior.
#
# C: z_dist_cycle_w_std combats the effect of pushing towards N(0,1) as does by scaled z_dist_cyc and not KL to N(0,1).

# %%
rcParams['figure.figsize']=(2,2)
g=sb.scatterplot(x='ilisi_system',y='nmi',
               hue='params_opt',
               data=res.groupby(['params_opt','param_opt_val'],observed=True
                               ).mean().reset_index(),
                palette='tab20')
sb.move_legend(g, loc='upper right',bbox_to_anchor=(3, 1))

# %%
rcParams['figure.figsize']=(2,2)
g=sb.scatterplot(x='ilisi_system',y='moransi',
               hue='params_opt',
               data=res.groupby(['params_opt','param_opt_val'],observed=True
                               ).mean().reset_index(),
                palette='tab20')
sb.move_legend(g, loc='upper right',bbox_to_anchor=(3, 1))

# %% [markdown]
# ## Retina adult organoid

# %%
path_integration=path_eval+'retina_adult_organoid/integration/'

# %%
# Load integration results - params and metrics
res=[]
metrics_data=[]
for run in glob.glob(path_integration+'*/'):
    if os.path.exists(run+'args.pkl') and \
        os.path.exists(run+'scib_metrics.pkl') and \
        os.path.exists(run+'scib_metrics_scaled.pkl') and\
        os.path.exists(run+'scib_metrics_data.pkl'):
        args=pd.Series(vars(pkl.load(open(run+'args.pkl','rb'))))
        metrics=pd.Series(pkl.load(open(run+'scib_metrics.pkl','rb')))
        metrics_scl=pd.Series(pkl.load(open(run+'scib_metrics_scaled.pkl','rb')))
        metrics_scl.index=metrics_scl.index.map(lambda x: x+'_scaled')
        data=pd.concat([args,metrics,metrics_scl])
        name=run.split('/')[-2]
        data.name=name
        res.append(data)
        metrics_data_sub=pkl.load(open(run+'scib_metrics_data.pkl','rb'))
        metrics_data_sub['name']=name
        metrics_data.append(metrics_data_sub)
res=pd.concat(res,axis=1).T

# %%
#  Param that was optimised
res['param_opt_col']=res.params_opt.replace(
    {'vamp':'n_prior_components',
     'z_distance_cycle_weight_std':'z_distance_cycle_weight',
     'scglue_lam_graph':'lam_graph',
     'scglue_rel_gene_weight':'rel_gene_weight', 
     'scglue_lam_align':'lam_align',
     'scvi':None})
res['param_opt_val']=res.apply(
    lambda x: x[x['param_opt_col']] if x['param_opt_col'] is not None else 0,axis=1)

# %%
res['params_opt']=pd.Categorical(res['params_opt'],sorted(res['params_opt'].unique()), True)

# %%
g=sb.catplot( x='param_opt_val', y="asw_group",  col='params_opt',
           kind="swarm", data=res,sharex=False,
          height=2.5,aspect=1.3,color='k')
for ax in g.axes.ravel():
    ax.axhline(0.55,lw=0.5,c='gray')
    ax.axhline(0.65,lw=0.5,c='gray')

# %%
g=sb.catplot( x='param_opt_val', y="asw_group_scaled",  col='params_opt',
           kind="swarm", data=res,sharex=False,
          height=2.5,aspect=1.3,color='k')
for ax in g.axes.ravel():
    ax.axhline(0.55,lw=0.5,c='gray')
    ax.axhline(0.65,lw=0.5,c='gray')

# %%
g=sb.catplot( x='param_opt_val', y="asw_group_macro",  col='params_opt',
           kind="swarm", data=res,sharex=False,
          height=2.5,aspect=1.3,color='k')

# %%
g=sb.catplot( x='param_opt_val', y="nmi",  col='params_opt',
           kind="swarm", data=res,sharex=False,
          height=2.5,aspect=1.3,color='k')
for ax in g.axes.ravel():
    ax.axhline(0.88,lw=0.5,c='gray')
    ax.axhline(0.92,lw=0.5,c='gray')
    ax.axhline(0.96,lw=0.5,c='gray')

# %%
g=sb.catplot( x='param_opt_val', y="nmi_scaled",  col='params_opt',
           kind="swarm", data=res,sharex=False,
          height=2.5,aspect=1.3,color='k')
for ax in g.axes.ravel():
    ax.axhline(0.88,lw=0.5,c='gray')
    ax.axhline(0.92,lw=0.5,c='gray')
    ax.axhline(0.96,lw=0.5,c='gray')

# %%
g=sb.catplot( x='param_opt_val', y="ari",  col='params_opt',
           kind="swarm", data=res,sharex=False,
          height=2.5,aspect=1.3,color='k')

# %%
g=sb.catplot( x='param_opt_val', y="jaccard",  col='params_opt',
           kind="swarm", data=res,sharex=False,
          height=2.5,aspect=1.3,color='k')

# %%
g=sb.catplot( x='param_opt_val', y="jaccard_macro",  col='params_opt',
           kind="swarm", data=res,sharex=False,
          height=2.5,aspect=1.3,color='k')

# %%
metrics_data_heatmap(metrics_data=metrics_data,res=res,metric='jaccard_label')

# %%
g=sb.catplot( x='param_opt_val', y="clisi",  col='params_opt',
           kind="swarm", data=res,sharex=False,
          height=2.5,aspect=1.3,color='k')

# %%
g=sb.catplot( x='param_opt_val', y="clisi_macro",  col='params_opt',
           kind="swarm", data=res,sharex=False,
          height=2.5,aspect=1.3,color='k')

# %%
g=sb.catplot( x='param_opt_val', y="moransi",  col='params_opt',
           kind="swarm", data=res,sharex=False,
          height=2.5,aspect=1.3,color='k')
for ax in g.axes.ravel():
    ax.axhline(0.92,lw=0.5,c='gray')
    ax.axhline(0.94,lw=0.5,c='gray')
    ax.axhline(0.96,lw=0.5,c='gray')

# %%
g=sb.catplot( x='param_opt_val', y="moransi_scaled",  col='params_opt',
           kind="swarm", data=res,sharex=False,
          height=2.5,aspect=1.3,color='k')
for ax in g.axes.ravel():
    ax.axhline(0.92,lw=0.5,c='gray')
    ax.axhline(0.94,lw=0.5,c='gray')
    ax.axhline(0.96,lw=0.5,c='gray')

# %%
g=sb.catplot( x='param_opt_val', y="ilisi_system",  col='params_opt',
           kind="swarm", data=res,sharex=False,
          height=2.5,aspect=1.3,color='k')
for ax in g.axes.ravel():
    ax.axhline(0.02,lw=0.5,c='gray')
    ax.axhline(0.12,lw=0.5,c='gray')
    ax.axhline(0.22,lw=0.5,c='gray')

# %%
g=sb.catplot( x='param_opt_val', y="ilisi_system_scaled",  col='params_opt',
           kind="swarm", data=res,sharex=False,
          height=2.5,aspect=1.3,color='k')
for ax in g.axes.ravel():
    ax.axhline(0.02,lw=0.5,c='gray')
    ax.axhline(0.12,lw=0.5,c='gray')
    ax.axhline(0.22,lw=0.5,c='gray')

# %%
g=sb.catplot( x='param_opt_val', y="ilisi_system_macro",  col='params_opt',
           kind="swarm", data=res,sharex=False,
          height=2.5,aspect=1.3,color='k')

# %%
g=sb.catplot( x='param_opt_val', y="ilisi_system_macro_scaled",  col='params_opt',
           kind="swarm", data=res,sharex=False,
          height=2.5,aspect=1.3,color='k')

# %%
rcParams['figure.figsize']=(2,2)
g=sb.scatterplot(x='ilisi_system',y='nmi',
               hue='params_opt',
               data=res.groupby(['params_opt','param_opt_val'],observed=True
                               ).mean().reset_index(),
                palette='colorblind')
sb.move_legend(g, loc='upper right',bbox_to_anchor=(3, 1))

# %%
rcParams['figure.figsize']=(2,2)
g=sb.scatterplot(x='ilisi_system',y='moransi',
               hue='params_opt',
               data=res.groupby(['params_opt','param_opt_val'],observed=True
                               ).mean().reset_index(),
                palette='colorblind')
sb.move_legend(g, loc='upper right',bbox_to_anchor=(3, 1))

# %%
# Examples
# for params,run_dir in res.groupby(['params_opt','param_opt_val']).apply(
#     lambda x: x.sort_values('ilisi_system',ascending=True).index[0]).iteritems():
    
#     print(params,run_dir)
#     display(res.loc[run_dir,['ilisi_system','asw_group']])
#     with plt.rc_context({'figure.figsize':(40,10)}):
#         path_run=path_integration+run_dir+'/'
#         for img_fn in ['umap.png','losses.png']:
#             img = mpimg.imread(path_run+img_fn)
#             imgplot = plt.imshow(img)
#             plt.show()

# %% [markdown]
# ## Adipose sc sn updated

# %%
path_integration=path_eval+'adipose_sc_sn_updated/integration/'

# %%
# Load integration results - params and metrics
res=[]
metrics_data=[]
for run in glob.glob(path_integration+'*/'):
    if os.path.exists(run+'args.pkl') and \
        os.path.exists(run+'scib_metrics.pkl') and \
        os.path.exists(run+'scib_metrics_scaled.pkl') and\
        os.path.exists(run+'scib_metrics_data.pkl'):
        args=pd.Series(vars(pkl.load(open(run+'args.pkl','rb'))))
        metrics=pd.Series(pkl.load(open(run+'scib_metrics.pkl','rb')))
        metrics_scl=pd.Series(pkl.load(open(run+'scib_metrics_scaled.pkl','rb')))
        metrics_scl.index=metrics_scl.index.map(lambda x: x+'_scaled')
        data=pd.concat([args,metrics,metrics_scl])
        name=run.split('/')[-2]
        data.name=name
        res.append(data)
        metrics_data_sub=pkl.load(open(run+'scib_metrics_data.pkl','rb'))
        metrics_data_sub['name']=name
        metrics_data.append(metrics_data_sub)
res=pd.concat(res,axis=1).T

# %%
res.params_opt.unique()

# %%
#  Param that was optimised
res['param_opt_col']=res.params_opt.replace(
    {'vamp':'n_prior_components',
     'z_distance_cycle_weight_std':'z_distance_cycle_weight',
     'scglue_lam_graph':'lam_graph',
     'scglue_rel_gene_weight':'rel_gene_weight', 
     'scglue_lam_align':'lam_align',
     'scvi':None})
res['param_opt_val']=res.apply(
    lambda x: x[x['param_opt_col']] if x['param_opt_col'] is not None else 0,axis=1)

# %%
res['params_opt']=pd.Categorical(res['params_opt'],sorted(res['params_opt'].unique()), True)

# %%
g=sb.catplot( x='param_opt_val', y="asw_group",  col='params_opt',
           kind="swarm", data=res,sharex=False,
          height=2.5,aspect=1.3,color='k')
for ax in g.axes.ravel():
    ax.axhline(0.56,lw=0.5,c='gray')

# %%
g=sb.catplot( x='param_opt_val', y="nmi",  col='params_opt',
           kind="swarm", data=res,sharex=False,
          height=2.5,aspect=1.3,color='k')
for ax in g.axes.ravel():
    ax.axhline(0.90,lw=0.5,c='gray')
    ax.axhline(0.85,lw=0.5,c='gray')
    ax.axhline(0.80,lw=0.5,c='gray')

# %%
g=sb.catplot( x='param_opt_val', y="nmi_scaled",  col='params_opt',
           kind="swarm", data=res,sharex=False,
          height=2.5,aspect=1.3,color='k')
for ax in g.axes.ravel():
    ax.axhline(0.90,lw=0.5,c='gray')
    ax.axhline(0.85,lw=0.5,c='gray')
    ax.axhline(0.80,lw=0.5,c='gray')

# %%
g=sb.catplot( x='param_opt_val', y="ari",  col='params_opt',
           kind="swarm", data=res,sharex=False,
          height=2.5,aspect=1.3,color='k')

# %%
g=sb.catplot( x='param_opt_val', y="jaccard",  col='params_opt',
           kind="swarm", data=res,sharex=False,
          height=2.5,aspect=1.3,color='k')

# %%
g=sb.catplot( x='param_opt_val', y="jaccard_macro",  col='params_opt',
           kind="swarm", data=res,sharex=False,
          height=2.5,aspect=1.3,color='k')

# %%
metrics_data_heatmap(metrics_data=metrics_data,res=res,metric='jaccard_label')

# %%
g=sb.catplot( x='param_opt_val', y="clisi",  col='params_opt',
           kind="swarm", data=res,sharex=False,
          height=2.5,aspect=1.3,color='k')

# %%
g=sb.catplot( x='param_opt_val', y="moransi",  col='params_opt',
           kind="swarm", data=res,sharex=False,
          height=2.5,aspect=1.3,color='k')
for ax in g.axes.ravel():
    ax.axhline(0.90,lw=0.5,c='gray')
    ax.axhline(0.93,lw=0.5,c='gray')
    ax.axhline(0.96,lw=0.5,c='gray')

# %%
g=sb.catplot( x='param_opt_val', y="moransi_scaled",  col='params_opt',
           kind="swarm", data=res,sharex=False,
          height=2.5,aspect=1.3,color='k')
for ax in g.axes.ravel():
    ax.axhline(0.90,lw=0.5,c='gray')
    ax.axhline(0.93,lw=0.5,c='gray')
    ax.axhline(0.96,lw=0.5,c='gray')

# %%
g=sb.catplot( x='param_opt_val', y="ilisi_system",  col='params_opt',
           kind="swarm", data=res,sharex=False,
          height=2.5,aspect=1.3,color='k')
for ax in g.axes.ravel():
    ax.axhline(0.03,lw=0.5,c='gray')

# %%
g=sb.catplot( x='param_opt_val', y="ilisi_system_scaled",  col='params_opt',
           kind="swarm", data=res,sharex=False,
          height=2.5,aspect=1.3,color='k')
for ax in g.axes.ravel():
    ax.axhline(0.03,lw=0.5,c='gray')

# %%
rcParams['figure.figsize']=(2,2)
g=sb.scatterplot(x='ilisi_system',y='nmi',
               hue='params_opt',
               data=res.groupby(['params_opt','param_opt_val'],observed=True
                               ).mean().reset_index(),
                palette='colorblind')
sb.move_legend(g, loc='upper right',bbox_to_anchor=(3, 1))

# %%
rcParams['figure.figsize']=(2,2)
g=sb.scatterplot(x='ilisi_system',y='moransi',
               hue='params_opt',
               data=res.groupby(['params_opt','param_opt_val'],observed=True
                               ).mean().reset_index(),
                palette='colorblind')
sb.move_legend(g, loc='upper right',bbox_to_anchor=(3, 1))
