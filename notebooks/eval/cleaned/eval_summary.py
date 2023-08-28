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
path_data='/om2/user/khrovati/data/cross_system_integration/'
path_eval=path_data+'eval/'
path_names=path_data+'names_parsed/'


# %%
def metrics_data_heatmap(metrics_data,res,metric='jaccard_label'):
    dat={}
    for metrics_res in metrics_data:
        dat[metrics_res['name']]=metrics_res[metric]
    dat=pd.DataFrame(dat).T
    dat=dat.loc[res.sort_values(['params_opt','param_opt_val']).index,:]
    #dat['params_opt']=dat.index.map(lambda x: res.at[x,'params_opt'])
    #dat['param_opt_val']=dat.index.map(lambda x: res.at[x,'param_opt_val'])
    #dat=dat.groupby(['params_opt','param_opt_val']).mean()
    dat.index=dat.index.map(lambda x: '-'.join(
    [str(res.at[x,'params_opt']),
     str(res.at[x,'param_opt_val'])]))
    dat=dat/np.clip(dat.max(axis=0),a_min=1e-3, a_max=None)
    sb.clustermap(dat,row_cluster=False, xticklabels=True,yticklabels=True)


# %%
# Map between params and model
params_opt_map=pkl.load(open(path_names+'params_opt_model.pkl','rb'))# Query to remove irrelevant runs accross all models for final model selection
# Parameter values in specific models to be optimized
param_opt_vals=pkl.load(open(path_names+'optimized_parameter_values.pkl','rb'))
def get_top_runs(res,param_opt_vals=param_opt_vals,params_opt_map=params_opt_map):
    """
    Find best runs for each method accross tunned params. 
    For brevity find just one saturn and scGLUE methods for orthologues and non-orthologues. 
    Compute overall score (for each method) by first minmax-normalizing scores within method 
    and then having ilisi for batch and mean of moransi and nmi_opt for bio; 
    overall score is mean of bio and batch.
    Return top runs and top setting (mean over runs) longside with median run in the setting
    """
    # Keep relevant params and name model
    params_opt_vals=set(params_opt_map.keys())
    res_sub=res.query('params_opt in @params_opt_vals').copy()
    res_sub['model']=res_sub.params_opt.replace(params_opt_map).astype(str)   
    nonopt_models=list(
        (set(params_opt_map.values()) & set(res_sub['model'].unique()))-set(
        [model for models,params_vals in param_opt_vals for model in models]))
    res_query=[f'model in {nonopt_models}']
    for models,params_vals in param_opt_vals:
        res_query_sub=[]
        for param,vals in params_vals:
            if param in res_sub.columns:
                res_query_sub.append(f'({param} in {vals} & "{param}"==param_opt_col)')
        if len(res_query_sub)>0:
            res_query_sub='(('+' | '.join(res_query_sub)+f') & model in {models})'
            res_query.append(res_query_sub)
    res_query=' | '.join(res_query)
    #print(res_query)
    res_sub=res_sub.query(res_query).copy()
    display(res_sub.groupby(['model','params_opt'],observed=True).size())
    
    # Normalize relevant metrics per model
    metrics=['nmi_opt','moransi','ilisi_system']
    for metric in metrics:
        res_sub[metric+'_norm']=res_sub.groupby('model')[metric].transform(minmax_scale)
    # Compute batch and bio metrics
    res_sub['batch_score']=res_sub['ilisi_system_norm']
    res_sub['bio_score']=res_sub[['nmi_opt_norm','moransi_norm']].mean(axis=1)
    #res_sub['overall_score']=res_sub[['bio_score','batch_score']].mean(axis=1)
    res_sub['overall_score']=res_sub['bio_score']*0.6+res_sub['batch_score']*0.4
    
    # Top run per method
    top_runs=res_sub.groupby('model').apply(lambda x: x.index[x['overall_score'].argmax()]
                                           ).to_dict()
    # Top parameters setting per method and the middle performing run from that setting
    top_settings={}
    for model,res_model in res_sub.groupby('model'):
        setting_cols=['params_opt','param_opt_val']
        setting_means=res_model.groupby(setting_cols,observed=True)['overall_score'].mean()
        top_setting=dict(zip(setting_cols,setting_means.index[setting_means.argmax()]))
        runs_data=res_model.query(
            f'params_opt=="{top_setting["params_opt"]}" & param_opt_val== {top_setting["param_opt_val"]}')   
        mid_run=runs_data.index[runs_data.overall_score==runs_data.overall_score.median()][0]
        top_settings[model]=dict(
            params=top_setting, runs=list(runs_data.index),mid_run=mid_run)
    
    return top_runs, top_settings


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
     'saturn_no_pe_sim_penalty':'saturn_pe_sim_penalty_no',
     'saturn_no_pe_sim_penalty_super':'saturn_pe_sim_penalty_super_no'})
res['param_opt_col']=res.params_opt.replace(
    {'vamp':'n_prior_components',
     'vamp_eval':'n_prior_components',
     'vamp_eval_fixed':'n_prior_components',
     'z_distance_cycle_weight_std':'z_distance_cycle_weight',
     'vamp_z_distance_cycle_weight_std':'z_distance_cycle_weight',
     'vamp_z_distance_cycle_weight_std_eval':'z_distance_cycle_weight',
     'vamp_kl_weight':'kl_weight',
     'vamp_kl_weight_eval':'kl_weight',
     'scglue_lam_graph':'lam_graph',
     'scglue_rel_gene_weight':'rel_gene_weight', 
     'scglue_lam_align':'lam_align',
     'scglue_lam_graph_no':'lam_graph',
     'scglue_rel_gene_weight_no':'rel_gene_weight', 
     'scglue_lam_align_no':'lam_align',
     'saturn_pe_sim_penalty':'pe_sim_penalty',
     'saturn_pe_sim_penalty_no':'pe_sim_penalty',
     'saturn_pe_sim_penalty_super':'pe_sim_penalty',
     'saturn_pe_sim_penalty_super_no':'pe_sim_penalty',
     
     'scvi':None,
     'scvi_kl_anneal':'kl_weight'})
res['param_opt_val']=res.apply(
    lambda x: x[x['param_opt_col']] if x['param_opt_col'] is not None else 0,axis=1)

# %%
res['params_opt']=pd.Categorical(res['params_opt'],sorted(res['params_opt'].unique()), True)

# %%
res.query('params_opt=="scglue_lam_align"')['param_opt_val']

# %%
# List all param values by param opt
#res.groupby(['param_opt_col']).apply(lambda x: sorted(x['param_opt_val'].unique())).to_dict()

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
g=sb.catplot( x='param_opt_val', y="nmi_opt",  col='params_opt',
           kind="swarm", data=res,sharex=False,
          height=2.5,aspect=1.3,color='k')

# %%
# Check vamp and cVAE in more detail only
# res_temp=res.query( '(params_opt.str.startswith("vamp") & not params_opt.str.contains("cycle")) | params_opt=="kl_weight"', 
#                engine='python').copy()
# res_temp['params_opt']=res_temp['params_opt'].cat.remove_unused_categories()
# g=sb.catplot( x='param_opt_val', y="nmi_opt",  col='params_opt',
#            kind="swarm", data=res_temp,
#              sharex=False, height=2.5,aspect=1.3,color='k')
# for ax in g.axes.ravel():
#     ax.axhline(0.90,lw=0.5,c='gray')
#     ax.axhline(0.85,lw=0.5,c='gray')
#     ax.axhline(0.80,lw=0.5,c='gray')

# %%
# Check saturn only in more detail
res_temp=res.query( 'params_opt.str.startswith("saturn") & not params_opt.str.contains("super")', 
               engine='python').copy()
res_temp['params_opt']=res_temp['params_opt'].cat.remove_unused_categories()
g=sb.catplot( x='param_opt_val', y="nmi_opt",  col='params_opt',
           kind="swarm", data=res_temp,
             sharex=False, height=2.5,aspect=1.3,color='k')

# %%
g=sb.catplot( x='param_opt_val', y="nmi_opt_scaled",  col='params_opt',
           kind="swarm", data=res,sharex=False,
          height=2.5,aspect=1.3,color='k')

# %%
g=sb.catplot( x='param_opt_val', y="ari_opt",  col='params_opt',
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
res_sub=res.query('seed==1')
metrics_data_heatmap(
    metrics_data=[run for run in metrics_data if run['name'] in res_sub.index],
    res=res_sub,metric='jaccard_label')

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
# Check vamp and cVAE in more detail only
# res_temp=res.query( '(params_opt.str.startswith("vamp") & not params_opt.str.contains("cycle")) | params_opt=="kl_weight"', 
#                engine='python').copy()
# res_temp['params_opt']=res_temp['params_opt'].cat.remove_unused_categories()
# g=sb.catplot( x='param_opt_val', y="moransi",  col='params_opt',
#            kind="swarm", data=res_temp,
#              sharex=False, height=2.5,aspect=1.3,color='k')
# for ax in g.axes.ravel():
#     ax.axhline(0.94,lw=0.5,c='gray')
#     ax.axhline(0.93,lw=0.5,c='gray')
#     ax.axhline(0.90,lw=0.5,c='gray')

# %%
# Check saturn only in more detail
res_temp=res.query( 'params_opt.str.startswith("saturn") & not params_opt.str.contains("super")', 
               engine='python').copy()
res_temp['params_opt']=res_temp['params_opt'].cat.remove_unused_categories()
g=sb.catplot( x='param_opt_val', y="moransi",  col='params_opt',
           kind="swarm", data=res_temp,
             sharex=False, height=2.5,aspect=1.3,color='k')

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
# Check vamp and cVAE in more detail only
# res_temp=res.query( '(params_opt.str.startswith("vamp") & not params_opt.str.contains("cycle")) | params_opt=="kl_weight"', 
#                engine='python').copy()
# res_temp['params_opt']=res_temp['params_opt'].cat.remove_unused_categories()
# g=sb.catplot( x='param_opt_val', y="ilisi_system",  col='params_opt',
#            kind="swarm", data=res_temp,
#              sharex=False, height=2.5,aspect=1.3,color='k')
# for ax in g.axes.ravel():
#     ax.axhline(0.2,lw=0.5,c='gray')
#     ax.axhline(0.15,lw=0.5,c='gray')
#     ax.axhline(0.1,lw=0.5,c='gray')

# %%
# Check saturn only in more detail
res_temp=res.query( 'params_opt.str.startswith("saturn") & not params_opt.str.contains("super")', 
               engine='python').copy()
res_temp['params_opt']=res_temp['params_opt'].cat.remove_unused_categories()
g=sb.catplot( x='param_opt_val', y="ilisi_system",  col='params_opt',
           kind="swarm", data=res_temp,
             sharex=False, height=2.5,aspect=1.3,color='k')

# %%
g=sb.catplot( x='param_opt_val', y="ilisi_system_scaled",  col='params_opt',
           kind="swarm", data=res,sharex=False,
          height=2.5,aspect=1.3,color='k')
for ax in g.axes.ravel():
    ax.axhline(0.03,lw=0.5,c='gray')

# %%
g=sb.catplot( x='param_opt_val', y="ilisi_system_macro",  col='params_opt',
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
g=sb.scatterplot(x='ilisi_system',y='nmi_opt',
               hue='params_opt',
               data=res.groupby(['params_opt','param_opt_val'],observed=True
                               ).mean().reset_index(),
                palette='tab20')
sb.move_legend(g, loc='upper right',bbox_to_anchor=(3.2, 1))

# %%
rcParams['figure.figsize']=(2,2)
g=sb.scatterplot(x='ilisi_system',y='moransi',
               hue='params_opt',
               data=res.groupby(['params_opt','param_opt_val'],observed=True
                               ).mean().reset_index(),
                palette='tab20')
sb.move_legend(g, loc='upper right',bbox_to_anchor=(3.2, 1))

# %% [markdown]
# rcParams['figure.figsize']=(2,2)
# g=sb.scatterplot(x='ilisi_system',y='moransi',
#                hue='params_opt',
#                data=res.groupby(['params_opt','param_opt_val'],observed=True
#                                ).mean().reset_index(),
#                 palette='tab20')
# sb.move_legend(g, loc='upper right',bbox_to_anchor=(3, 1))

# %% [markdown]
# ### Best runs

# %%
top_runs,top_settings=get_top_runs(res)
print('Top runs')
display(top_runs)
print('Top settings')
for model,setting in top_settings.items():
    print(model)
    print(tuple(setting['params'].values()))
    print(setting['mid_run'])

# %%
res.loc[top_runs.values(),['params_opt','param_opt_val','seed']]

# %%
pkl.dump(top_runs,open(path_integration.rstrip('/')+'_summary/top_runs.pkl','wb'))
pkl.dump(top_settings,open(path_integration.rstrip('/')+'_summary/top_settings.pkl','wb'))

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
     'vamp_eval':'n_prior_components',
     'vamp_eval_fixed':'n_prior_components',
     'z_distance_cycle_weight_std':'z_distance_cycle_weight',
     'vamp_z_distance_cycle_weight_std':'z_distance_cycle_weight',
     'vamp_z_distance_cycle_weight_std_eval':'z_distance_cycle_weight',
     'vamp_kl_weight':'kl_weight',
     'vamp_kl_weight_eval':'kl_weight',
     'scglue_lam_graph':'lam_graph',
     'scglue_rel_gene_weight':'rel_gene_weight', 
     'scglue_lam_align':'lam_align',
     'scvi':None,
     'scvi_kl_anneal':'kl_weight'})
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
g=sb.catplot( x='param_opt_val', y="nmi_opt",  col='params_opt',
           kind="swarm", data=res,sharex=False,
          height=2.5,aspect=1.3,color='k')

# %%
# Check vamp and cVAE in more detail only
# res_temp=res.query( '(params_opt.str.startswith("vamp") & not params_opt.str.contains("cycle")) | params_opt=="kl_weight"', 
#                engine='python').copy()
# res_temp['params_opt']=res_temp['params_opt'].cat.remove_unused_categories()
# g=sb.catplot( x='param_opt_val', y="nmi_opt",  col='params_opt',
#            kind="swarm", data=res_temp,
#              sharex=False, height=2.5,aspect=1.3,color='k')
# for ax in g.axes.ravel():
#     ax.axhline(0.95,lw=0.5,c='gray')
#     ax.axhline(0.90,lw=0.5,c='gray')
#     ax.axhline(0.85,lw=0.5,c='gray')

# %%
g=sb.catplot( x='param_opt_val', y="nmi_opt_scaled",  col='params_opt',
           kind="swarm", data=res,sharex=False,
          height=2.5,aspect=1.3,color='k')

# %%
g=sb.catplot( x='param_opt_val', y="ari_opt",  col='params_opt',
           kind="swarm", data=res,sharex=False,
          height=2.5,aspect=1.3,color='k')

# %%
g=sb.catplot( x='param_opt_val', y="jaccard",  col='params_opt',
           kind="swarm", data=res,sharex=False,
          height=2.5,aspect=1.3,color='k')

# %%
g=sb.catplot( x='param_opt_val', y="jaccard_scaled",  col='params_opt',
           kind="swarm", data=res,sharex=False,
          height=2.5,aspect=1.3,color='k')

# %%
g=sb.catplot( x='param_opt_val', y="jaccard_macro",  col='params_opt',
           kind="swarm", data=res,sharex=False,
          height=2.5,aspect=1.3,color='k')

# %%
res_sub=res.query('seed==1')
metrics_data_heatmap(
    metrics_data=[run for run in metrics_data if run['name'] in res_sub.index],
    res=res_sub,metric='jaccard_label')

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
# Check vamp and cVAE in more detail only
# res_temp=res.query( '(params_opt.str.startswith("vamp") & not params_opt.str.contains("cycle")) | params_opt=="kl_weight"', 
#                engine='python').copy()
# res_temp['params_opt']=res_temp['params_opt'].cat.remove_unused_categories()
# g=sb.catplot( x='param_opt_val', y="moransi",  col='params_opt',
#            kind="swarm", data=res_temp,
#              sharex=False, height=2.5,aspect=1.3,color='k')
# for ax in g.axes.ravel():
#     ax.axhline(0.96,lw=0.5,c='gray')
#     ax.axhline(0.95,lw=0.5,c='gray')
#     ax.axhline(0.94,lw=0.5,c='gray')

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
# # Check vamp and cVAE in more detail only
# res_temp=res.query( '(params_opt.str.startswith("vamp") & not params_opt.str.contains("cycle")) | params_opt=="kl_weight"', 
#                engine='python').copy()
# res_temp['params_opt']=res_temp['params_opt'].cat.remove_unused_categories()
# g=sb.catplot( x='param_opt_val', y="ilisi_system",  col='params_opt',
#            kind="swarm", data=res_temp,
#              sharex=False, height=2.5,aspect=1.3,color='k')
# for ax in g.axes.ravel():
#     ax.axhline(0.2,lw=0.5,c='gray')
#     ax.axhline(0.15,lw=0.5,c='gray')
#     ax.axhline(0.1,lw=0.5,c='gray')

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
g=sb.scatterplot(x='ilisi_system',y='nmi_opt',
               hue='params_opt',
               data=res.groupby(['params_opt','param_opt_val'],observed=True
                               ).mean().reset_index(),
                palette='colorblind')
sb.move_legend(g, loc='upper right',bbox_to_anchor=(3.2, 1))

# %%
rcParams['figure.figsize']=(2,2)
g=sb.scatterplot(x='ilisi_system',y='moransi',
               hue='params_opt',
               data=res.groupby(['params_opt','param_opt_val'],observed=True
                               ).mean().reset_index(),
                palette='colorblind')
sb.move_legend(g, loc='upper right',bbox_to_anchor=(3.2, 1))

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
# ### Best runs

# %%
top_runs,top_settings=get_top_runs(res)
print('Top runs')
display(top_runs)
print('Top settings')
for model,setting in top_settings.items():
    print(model)
    print(tuple(setting['params'].values()))
    print(setting['mid_run'])

# %%
res.loc[top_runs.values(),['params_opt','param_opt_val','seed']]

# %%
pkl.dump(top_runs,open(path_integration.rstrip('/')+'_summary/top_runs.pkl','wb'))
pkl.dump(top_settings,open(path_integration.rstrip('/')+'_summary/top_settings.pkl','wb'))

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
#  Param that was optimised
res['param_opt_col']=res.params_opt.replace(
    {'kl_weight_anneal':'kl_weight',
     'vamp':'n_prior_components',
     'vamp_eval':'n_prior_components',
     'vamp_eval_fixed':'n_prior_components',
     'vamp_kl_anneal':'n_prior_components',
     'z_distance_cycle_weight_std':'z_distance_cycle_weight',
     'z_distance_cycle_weight_std_kl_anneal':'z_distance_cycle_weight',
     'vamp_z_distance_cycle_weight_std':'z_distance_cycle_weight',
     'vamp_z_distance_cycle_weight_std_eval':'z_distance_cycle_weight',
     'vamp_kl_weight':'kl_weight',
     'vamp_kl_weight_eval':'kl_weight',
     'scglue_lam_graph':'lam_graph',
     'scglue_rel_gene_weight':'rel_gene_weight', 
     'scglue_lam_align':'lam_align',
     'scvi':None,
     'scvi_kl_anneal':'kl_weight'})
res['param_opt_val']=res.apply(
    lambda x: (x[x['param_opt_col']] if not isinstance(x[x['param_opt_col']],dict)
              else x[x['param_opt_col']]['weight_end']) 
    if x['param_opt_col'] is not None else 0,axis=1)

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
g=sb.catplot( x='param_opt_val', y="nmi_opt",  col='params_opt',
           kind="swarm", data=res,sharex=False,
          height=2.5,aspect=1.3,color='k')

# %%
# Check vamp and cVAE in more detail only
# res_temp=res.query( '(params_opt.str.startswith("vamp") & not params_opt.str.contains("cycle")) | params_opt=="kl_weight"', 
#                engine='python').copy()
# res_temp['params_opt']=res_temp['params_opt'].cat.remove_unused_categories()
# g=sb.catplot( x='param_opt_val', y="nmi_opt",  col='params_opt',
#            kind="swarm", data=res_temp,
#              sharex=False, height=2.5,aspect=1.3,color='k')
# for ax in g.axes.ravel():
#     ax.axhline(0.85,lw=0.5,c='gray')
#     ax.axhline(0.80,lw=0.5,c='gray')
#     ax.axhline(0.75,lw=0.5,c='gray')

# %%
g=sb.catplot( x='param_opt_val', y="nmi_opt_scaled",  col='params_opt',
           kind="swarm", data=res,sharex=False,
          height=2.5,aspect=1.3,color='k')

# %%
g=sb.catplot( x='param_opt_val', y="ari_opt",  col='params_opt',
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
res_sub=res.query('seed==1')
metrics_data_heatmap(
    metrics_data=[run for run in metrics_data if run['name'] in res_sub.index],
    res=res_sub,metric='jaccard_label')

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
# Check vamp and cVAE in more detail only
# res_temp=res.query( '(params_opt.str.startswith("vamp") & not params_opt.str.contains("cycle")) | params_opt=="kl_weight"', 
#                engine='python').copy()
# res_temp['params_opt']=res_temp['params_opt'].cat.remove_unused_categories()
# g=sb.catplot( x='param_opt_val', y="moransi",  col='params_opt',
#            kind="swarm", data=res_temp,
#              sharex=False, height=2.5,aspect=1.3,color='k')
# for ax in g.axes.ravel():
#     ax.axhline(0.95,lw=0.5,c='gray')
#     ax.axhline(0.94,lw=0.5,c='gray')
#     ax.axhline(0.93,lw=0.5,c='gray')

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
    ax.axhline(0.15,lw=0.5,c='gray')

# %%
# Check vamp and cVAE in more detail only
# res_temp=res.query( '(params_opt.str.startswith("vamp") & not params_opt.str.contains("cycle")) | params_opt=="kl_weight"', 
#                engine='python').copy()
# res_temp['params_opt']=res_temp['params_opt'].cat.remove_unused_categories()
# g=sb.catplot( x='param_opt_val', y="ilisi_system",  col='params_opt',
#            kind="swarm", data=res_temp,
#              sharex=False, height=2.5,aspect=1.3,color='k')
# for ax in g.axes.ravel():
#     ax.axhline(0.2,lw=0.5,c='gray')
#     ax.axhline(0.15,lw=0.5,c='gray')
#     ax.axhline(0.1,lw=0.5,c='gray')

# %%
g=sb.catplot( x='param_opt_val', y="ilisi_system_scaled",  col='params_opt',
           kind="swarm", data=res,sharex=False,
          height=2.5,aspect=1.3,color='k')
for ax in g.axes.ravel():
    ax.axhline(0.03,lw=0.5,c='gray')

# %%
g=sb.catplot( x='param_opt_val', y="ilisi_system_macro",  col='params_opt',
           kind="swarm", data=res,sharex=False,
          height=2.5,aspect=1.3,color='k')
for ax in g.axes.ravel():
    ax.axhline(0.03,lw=0.5,c='gray')

# %%
rcParams['figure.figsize']=(2,2)
g=sb.scatterplot(x='ilisi_system',y='nmi_opt',
               hue='params_opt',
               data=res.groupby(['params_opt','param_opt_val'],observed=True
                               ).mean().reset_index(),
                palette='colorblind')
sb.move_legend(g, loc='upper right',bbox_to_anchor=(3.2, 1))

# %%
rcParams['figure.figsize']=(2,2)
g=sb.scatterplot(x='ilisi_system',y='moransi',
               hue='params_opt',
               data=res.groupby(['params_opt','param_opt_val'],observed=True
                               ).mean().reset_index(),
                palette='colorblind')
sb.move_legend(g, loc='upper right',bbox_to_anchor=(3.2, 1))

# %% [markdown]
# ### Best runs

# %%
top_runs,top_settings=get_top_runs(res)
print('Top runs')
display(top_runs)
print('Top settings')
for model,setting in top_settings.items():
    print(model)
    print(tuple(setting['params'].values()))
    print(setting['mid_run'])

# %%
res.loc[top_runs.values(),['params_opt','param_opt_val','seed']]

# %%
pkl.dump(top_runs,open(path_integration.rstrip('/')+'_summary/top_runs.pkl','wb'))
pkl.dump(top_settings,open(path_integration.rstrip('/')+'_summary/top_settings.pkl','wb'))

# %%
