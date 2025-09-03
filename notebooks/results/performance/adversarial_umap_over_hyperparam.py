# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.3
#   kernelspec:
#     display_name: csi
#     language: python
#     name: csi
# ---

# %%
import pandas as pd
import numpy as np
import scanpy as sc
import pickle as pkl
import yaml
import math
import glob
import os
import itertools
from datetime import datetime
from collections import defaultdict

from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

from matplotlib import rcParams
import matplotlib.pyplot as plt
import seaborn as sb
import matplotlib.colors as mcolors

from pathlib import Path
import sys
sys.path.append('/'.join(os.getcwd().split('/')[:-2]+['eval','cleaned','']))
from params_opt_maps import *

# %%
sc.set_figure_params(vector_friendly=True, dpi_save=300)

# %%
import warnings
# warnings.filterwarnings(action='once')
warnings.filterwarnings('ignore')

# %%
path_data='/home/moinfar/io/csi/'
path_names=path_data+'names_parsed/'
path_fig=path_data+'figures/'
path_tab=path_data+'tables/'

# %%
Path(path_fig).mkdir(parents=True, exist_ok=True)
Path(path_tab).mkdir(parents=True, exist_ok=True)

# %%
# Names
model_map={**pkl.load(open(path_names+'models.pkl','rb')),
           **pkl.load(open(path_names+'models_additional.pkl','rb'))}
param_map=pkl.load(open(path_names+'params.pkl','rb'))
metric_map=pkl.load(open(path_names+'metrics.pkl','rb'))
dataset_map=pkl.load(open(path_names+'datasets.pkl','rb'))
metric_meaning_map=pkl.load(open(path_names+'metric_meanings.pkl','rb'))
metric_map_rev=dict(zip(metric_map.values(),metric_map.keys()))
dataset_map_rev=dict(zip(dataset_map.values(),dataset_map.keys()))
system_map=pkl.load(open(path_names+'systems.pkl','rb'))
params_opt_map=pkl.load(open(path_names+'params_opt_model.pkl','rb'))
params_opt_gene_map=pkl.load(open(path_names+'params_opt_genes.pkl','rb'))
param_opt_vals=pkl.load(open(path_names+'optimized_parameter_values.pkl','rb'))
cell_type_map=pkl.load(open(path_names+'cell_types.pkl','rb'))

# cmap
model_cmap=pkl.load(open(path_names+'model_cmap.pkl','rb'))
obs_col_cmap=pkl.load(open(path_names+'obs_col_cmap.pkl','rb'))
metric_background_cmap=pkl.load(open(path_names+'metric_background_cmap.pkl','rb'))

# data
dataset_path=pkl.load(open(path_names+'dataset_path.pkl','rb'))
dataset_h5ad_path=pkl.load(open(path_names+'dataset_h5ad_path.pkl','rb'))

# %% [markdown]
# ## All Runs

# %%
dataset_map

# %%
# Load data and keep relevant runs
ress=[]
for dataset,dataset_name in dataset_map.items():
    print(dataset_name)

    top_settings=pkl.load(open(f'{path_data}eval/{dataset}/integration_summary/top_settings.pkl','rb'))
    top_runs = sum([v['runs'] for k, v in top_settings.items()], [])

    path_integration=f'{path_data}eval/{dataset}/integration/'
    res=[]
    for run in glob.glob(path_integration+'*/'):
        if (os.path.exists(run+'args.pkl') or os.path.exists(run+'args.yml')) and \
            os.path.exists(run+'scib_metrics.pkl'):
            if os.path.exists(run+'args.pkl'):
                args_=pd.Series(vars(pkl.load(open(run+'args.pkl','rb'))))
            if os.path.exists(run+'args.yml'):
                args_=pd.Series(yaml.safe_load(open(run+'args.yml','rb')))
            metrics_=pd.Series(pkl.load(open(run+'scib_metrics.pkl','rb')))
            run_id = run.strip('/').rsplit('/', 1)[-1]
            run_stats_ = pd.Series({
                'is_top': run_id in top_runs,
                'run_id': run_id,
                'run_path': run,
            })
            data=pd.concat([args_,metrics_, run_stats_])
            name=run.split('/')[-2]
            data.name=name
            res.append(data)
    res=pd.concat(res,axis=1).T

    # Parse res table

    # Parse params
    res['params_opt']=res.params_opt.replace(params_opt_correct_map)
    res['param_opt_col']=res.params_opt.replace(param_opt_col_map)
    res['param_opt_val']=res.apply(
        lambda x: (x[x['param_opt_col']] if not isinstance(x[x['param_opt_col']],dict)
                  else x[x['param_opt_col']]['weight_end']) 
                  if x['param_opt_col'] is not None else 0,axis=1)
    # Param opt val for plotting - converted to str categ below
    res['param_opt_val_str']=res.apply(
        lambda x: x[x['param_opt_col']] if x['param_opt_col'] is not None else np.nan,axis=1)

    ####
    res['params_opt']=np.where(res.index.str.contains('harmonypy'), 
                               res['params_opt'].replace({'harmony_theta': 'harmonypy_theta'}),
                               res['params_opt'])
    res['param_opt_col']=np.where(res.index.str.contains('harmonypy'), 
                                  res['param_opt_col'].replace({'harmony_theta': 'harmonypy_theta'}),
                                  res['param_opt_col'])
    res['harmonypy_theta'] = res['harmony_theta']
    ####
    
    res['params_opt']=pd.Categorical(res['params_opt'],sorted(res['params_opt'].unique()), True)

    # Keep relevant runs
    params_opt_vals=set(params_opt_map.keys())
    res_sub=res.query('params_opt in @params_opt_vals').copy()
    # Name models
    res_sub['model']=res_sub.params_opt.replace(params_opt_map).astype(str)   
    # Models present in data but have no params opt
    nonopt_models=list(
        (set(params_opt_map.values()) & set(res_sub['model'].unique()))-set(
        [model for models,params_vals in param_opt_vals for model in models]))
    # Query: a.) model not optimized OR b.) model belongs to one of the models that have 
    # optimized params and the optimized param is within list of param values
    res_query=[f'model in {nonopt_models}']
    # Models with opt params
    for models,params_vals in param_opt_vals:
        res_query_sub=[]
        # Param value in vals to keep if the param was optimised
        for param,vals in params_vals:
            # For param check if it was opt in data setting as else there will be no col for it
            if param in res_sub.columns:
                res_query_sub.append(f'({param} in {vals} & "{param}"==param_opt_col)')
        # Only add to the query the models for which any param was opt
        if len(res_query_sub)>0:
            res_query_sub='(('+' | '.join(res_query_sub)+f') & model in {models})'
            res_query.append(res_query_sub)
    res_query=' | '.join(res_query)
    res_sub=res_sub.query(res_query).copy()

    # Add pretty model names
    res_sub['model_parsed']=pd.Categorical(
        values=res_sub['model'].map(model_map),
        categories=model_map.values(), ordered=True)
    # Add prety param names
    res_sub['param_parsed']=pd.Categorical(
        values=res_sub['param_opt_col'].map(param_map),
        categories=param_map.values(), ordered=True)
    # Add gene setting names
    res_sub['genes_parsed']=pd.Categorical(
        values=res_sub['params_opt'].map(params_opt_gene_map),
         categories=list(dict.fromkeys(params_opt_gene_map.values())), ordered=True)
    
    # display(res_sub.groupby(['model_parsed','param_parsed','genes_parsed'],observed=True).size())
    
    # Store
    res_sub['dataset_parsed']=dataset_name
    ress.append(res_sub)

# Combine results of all datasets
ress=pd.concat(ress)

# Order datasets
ress['dataset_parsed']=pd.Categorical(
    values=ress['dataset_parsed'],
    categories=list(dataset_map.values()), ordered=True)

# Parse param valuse for plotting
ress['param_opt_val_str']=pd.Categorical(
    values=ress['param_opt_val_str'].fillna('none').astype(str),
    categories=[str(i) for i in 
                sorted([i for i in ress['param_opt_val_str'].unique() if not np.isnan(i)])
               ]+['none'],
    ordered=True)

# %%
drop_models = [
    "SysVI",
    "SysVI-stable",
]
ress.query('model not in @drop_models', inplace=True)
ress

# %%

# %% [markdown]
# ### Metric scores

# %%
ds_name = 'Pancreas Mouse-Human'
model_subset = ['GLUE', 'cVAE', 'ADV']
params_opt_subset = [
    'adv_kappa', 
    'kl_weight', 
    'scglue_lam_align', 
    # 'scglue_rel_gene_weight',
]

# %%
models_to_plot = defaultdict(lambda: defaultdict(dict))
for i, row in (
    ress
    .query('seed == 1')
    .query('dataset_parsed == @ds_name')
    .query('params_opt in @params_opt_subset')
    [['model_parsed', 'dataset_parsed', 'seed', 'params_opt', 'run_path', 'param_parsed', 'group_key', 'param_opt_val']]
).iterrows():
    models_to_plot[row['model_parsed']]['param_name'] = row['params_opt']
    models_to_plot[row['model_parsed']]['param_dict'][row['param_opt_val']] = row.to_dict()
models_to_plot

# %%
for model_name, multi_model_info in models_to_plot.items():
    params = multi_model_info['param_dict']
    ncols = 2
    nrows = math.ceil(len(params) / ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=(2.7*ncols, 2.7*nrows))

    for i, (ax, param_val) in enumerate(zip(axs.flatten(), sorted(params.keys()))):
        model_info = params[param_val]
        embed = sc.read_h5ad(os.path.join(model_info['run_path'], "embed.h5ad"))

        col = model_info['group_key']
        dataset_name = model_info['dataset_parsed']
        dataset = {v: k for k, v in dataset_map.items()}[dataset_name]
        cmap=obs_col_cmap[dataset_map_rev[dataset_name]][col]
        embed.obs[col+'_parsed']=embed.obs[col].astype(str).map(cell_type_map[dataset])
        cmap={cell_type_map[dataset][k]:v for k,v in cmap.items()}
        sc.pp.subsample(embed, fraction=1.)
        sc.pl.umap(
            embed,
            color=col+'_parsed',
            palette=cmap,
            ax=ax,
            show=False,
            frameon=False,
            legend_loc='none',
            # legend_loc='none' if i < (ncols - 1) else 'right margin',
            title=f"{model_info['param_parsed']}={param_val}"
        )
        if i == 0:
            ax.set_ylabel(model_name)

    # Hide any empty axes
    for ax in axs.flat[len(params):]:
        ax.axis("off")

    fig.supylabel(model_name)
    fig.tight_layout()
    plt.savefig(path_fig + f'adv_param_effect_on_umap_{model_name}.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(path_fig + f'adv_param_effect_on_umap_{model_name}.png', dpi=300, bbox_inches='tight')
    plt.show()

# %%
fig = sc.pl.umap(
    embed,
    color=col+'_parsed',
    palette=cmap,
    show=False,
    frameon=False,
    legend_loc='right margin',
)
plt.savefig(path_fig + f'adv_param_effect_on_umap___legend.pdf', dpi=300, bbox_inches='tight')
plt.savefig(path_fig + f'adv_param_effect_on_umap___legend.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
