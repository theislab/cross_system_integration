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
import colorcet as cc

from params_opt_maps import *

# %%
path_data='/om2/user/khrovati/data/cross_system_integration/'
path_eval=path_data+'eval/'
path_names=path_data+'names_parsed/'

# %%
# Map between params and model
params_opt_map=pkl.load(open(path_names+'params_opt_model.pkl','rb'))# Query to remove irrelevant runs accross all models for final model selection
# Parameter values in specific models to be optimized
param_opt_vals=pkl.load(open(path_names+'optimized_parameter_values.pkl','rb'))
def get_top_runs(res,param_opt_vals=param_opt_vals,params_opt_map=params_opt_map):
    """
    Find best runs for each method-and-parameter accross tunned params values. 
    For brevity find just one saturn and scGLUE methods for orthologues and non-orthologues. 
    Compute overall score (for each method) by first minmax-normalizing scores within method 
    and then having ilisi for batch and mean of moransi and nmi_opt for bio; 
    overall score is mean of bio and batch.
    Return top runs and top setting (mean over runs with same param values) 
    alongside with median run in the setting.
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


# %%
params_opt_colors=sb.color_palette(cc.glasbey, n_colors=len(param_opt_col_map))

# %% [markdown]
# ## Pancreas conditions MIA HPAP2

# %% [markdown]
# Load data

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
#  Parse param that was optimised
res['params_opt']=res.params_opt.replace(params_opt_correct_map)
res['param_opt_col']=res.params_opt.replace(param_opt_col_map)
res['param_opt_val']=res.apply(
    lambda x: x[x['param_opt_col']] if x['param_opt_col'] is not None else 0,axis=1)
res['params_opt']=pd.Categorical(res['params_opt'],sorted(res['params_opt'].unique()), True)

# %% [markdown]
# ### Best runs

# %%
# Top runs/settings
top_runs,top_settings=get_top_runs(res)
print('Top runs')
display(top_runs)
print('Top settings')
for model,setting in top_settings.items():
    print(model)
    print(tuple(setting['params'].values()))
    print(setting['mid_run'])

# %%
# Save
pkl.dump(top_runs,open(path_integration.rstrip('/')+'_summary/top_runs.pkl','wb'))
pkl.dump(top_settings,open(path_integration.rstrip('/')+'_summary/top_settings.pkl','wb'))

# %% [markdown]
# ## Retina adult organoid

# %% [markdown]
# Load data

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
#  Parse param that was optimised
res['param_opt_col']=res.params_opt.replace(param_opt_col_map)
res['param_opt_val']=res.apply(
    lambda x: x[x['param_opt_col']] if x['param_opt_col'] is not None else 0,axis=1)
res['params_opt']=pd.Categorical(res['params_opt'],sorted(res['params_opt'].unique()), True)

# %% [markdown]
# ### Best runs

# %%
# Top runs/settings
top_runs,top_settings=get_top_runs(res)
print('Top runs')
display(top_runs)
print('Top settings')
for model,setting in top_settings.items():
    print(model)
    print(tuple(setting['params'].values()))
    print(setting['mid_run'])

# %%
# Save
pkl.dump(top_runs,open(path_integration.rstrip('/')+'_summary/top_runs.pkl','wb'))
pkl.dump(top_settings,open(path_integration.rstrip('/')+'_summary/top_settings.pkl','wb'))

# %% [markdown]
# ### Select example scGEN runs (non-benchmarked)

# %%
example_runs={
    'scgen_sample':res.query('params_opt=="scgen_sample_kl" & seed==1 & kl_weight==0.1').index[0],
    'scgen_system':res.query('params_opt=="scgen_kl" & seed==1 & kl_weight==0.1').index[0],
}
pkl.dump(example_runs,open(path_integration.rstrip('/')+'_summary/example_runs.pkl','wb'))

# %% [markdown]
# ## Adipose sc sn updated

# %% [markdown]
# Load data

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
#  Parse param that was optimised
res['param_opt_col']=res.params_opt.replace(param_opt_col_map)
res['param_opt_val']=res.apply(
    lambda x: (x[x['param_opt_col']] if not isinstance(x[x['param_opt_col']],dict)
              else x[x['param_opt_col']]['weight_end']) 
    if x['param_opt_col'] is not None else 0,axis=1)
res['params_opt']=pd.Categorical(res['params_opt'],sorted(res['params_opt'].unique()), True)

# %% [markdown]
# ### Best runs

# %%
# Top runs/settings
top_runs,top_settings=get_top_runs(res)
print('Top runs')
display(top_runs)
print('Top settings')
for model,setting in top_settings.items():
    print(model)
    print(tuple(setting['params'].values()))
    print(setting['mid_run'])

# %%
# Save
pkl.dump(top_runs,open(path_integration.rstrip('/')+'_summary/top_runs.pkl','wb'))
pkl.dump(top_settings,open(path_integration.rstrip('/')+'_summary/top_settings.pkl','wb'))
