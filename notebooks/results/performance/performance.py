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
from copy import deepcopy

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
# ## Top settings
# Best performing parameter setting results

# %%
load_embed = True

# %%
# Load metrics and embeddings
metrics={}
if load_embed:
    embeds={}
args={}
for dataset,dataset_name in dataset_map.items():
    top_settings=pkl.load(open(f'{path_data}eval/{dataset}/integration_summary/top_settings.pkl','rb'))
    metrics[dataset_name]=[]
    if load_embed:
        embeds[dataset_name]={}
    args[dataset_name]={}
    path_integration=f'{path_data}eval/{dataset}/integration/'
    for model,model_setting in top_settings.items():
        model=model_map[model]
        for run in model_setting['runs']:
            if os.path.exists(path_integration+run+'/args.pkl'):
                args_run=vars(pkl.load(open(path_integration+run+'/args.pkl','rb')))
            else:
                args_run=yaml.safe_load(open(path_integration+run+'/args.yml','rb'))
            metrics_data=pd.Series(
                pkl.load(open(path_integration+run+'/scib_metrics.pkl','rb')),
                name=model)
            metrics_data['seed']=args_run['seed']
            metrics[dataset_name].append(metrics_data)
            if run==model_setting['mid_run']:
                if load_embed:
                    print(dataset_name, model, path_integration+run+'/embed.h5ad')
                    embed=sc.read(path_integration+run+'/embed.h5ad')
                    sc.pp.subsample(embed, fraction=1.)
                    embeds[dataset_name][model]=embed
                args[dataset_name][model]=args_run
    metrics[dataset_name]=pd.DataFrame(metrics[dataset_name])
    metrics[dataset_name]['model']=pd.Categorical(
        values=metrics[dataset_name].index,
        categories=[m for m in model_map.values() if m in metrics[dataset_name].index],
        ordered=True)
    metrics[dataset_name].rename(metric_map,axis=1,inplace=True)
    # Seed to str for plotting
    metrics[dataset_name]['seed']=metrics[dataset_name]['seed'].astype(str)

# %%
with open("/home/moinfar/io/csi/tmp/metrics.pkl", "wb") as f:
    pkl.dump(metrics, f)

# %%
with open("/home/moinfar/io/csi/tmp/metrics.pkl", "rb") as f:
    metrics = pkl.load(f)

# %%
# Add non-integrated embeds
if load_embed:
    dataset_embed_fns={
        dataset: os.path.join(dataset_path[dataset], dataset_h5ad_path[dataset])
        for dataset in dataset_path.keys()
    }
    for dataset,dataset_name in dataset_map.items():
        adata = sc.read(dataset_embed_fns[dataset][:-len('.h5ad')] + '_embed.h5ad')
        if adata.n_obs > 100_000:
            sc.pp.subsample(adata, n_obs=100_000)
        else:
            sc.pp.subsample(adata, fraction=1.)
        embeds[dataset_name][model_map['non-integrated']] = adata

# %%
# Add scGEN example embeds to retina
if load_embed:
    dataset='retina_adult_organoid'
    dataset_name=dataset_map[dataset]
    example_runs=pkl.load(open(f'{path_data}eval/{dataset}/integration_summary/example_runs.pkl','rb'))
    path_integration=f'{path_data}eval/{dataset}/integration/'
    for model,run in example_runs.items():
        model=model_map[model]
        embed=sc.read(path_integration+run+'/embed.h5ad')
        sc.pp.subsample(embed, fraction=1.)
        embeds[dataset_name][model]=embed



# %%
# Display loaded embeddings
if load_embed:
    for i,j in embeds.items():
        print('*** '+i)
        print(j.keys())

# %% [markdown]
# ### Metric scores

# %%
model_cmap

# %%
# Remove harmony (R) as it performs worse and does not scale to large datasets. We use harmonypy instead.

all_models = [
    "SysVI",
    "SysVI-stable",
    "VAMP+CYC",
    "CYC", "VAMP",
    "cVAE", "scVI",
    "Harmony",
    "Harmony", 
    "Seaurat", 
    "Harmony-py", 
    "GLUE",
    "SATURN", "SATURN-CT"
]
model_order = [
    # "SysVI",
    # "SysVI-stable",
    "VAMP+CYC",
    "CYC", "VAMP",
    "cVAE", "scVI",
    "Seaurat", 
    "Harmony", 
    "Harmony-py", 
    "GLUE",
    "SATURN", "SATURN-CT"
]
drop_models = [model_ for model_ in all_models if model_ not in model_order]

# %%
# Remove harmony (R) as it performs worse and does not scale to large datasets. We use harmonypy instead.

for metrics_sub in metrics.values():
    metrics_sub.query('model not in @drop_models', inplace=True)
    metrics_sub["model"] = pd.Categorical(metrics_sub["model"].astype(str), categories=model_order)


# %%

# %%
# total_score_formula = lambda metrics_df: 0.3 * metrics_df["NMI"] + 0.3 * metrics_df["Moran's I"] + 0.4 * metrics_df["iLISI"]
# total_score_formula = lambda metrics_df: (0.3 * metrics_df["NMI"]**2 + 0.3 * metrics_df["Moran's I"]**2 + 0.4 * metrics_df["iLISI"]**2)**0.5
def add_total_score_formula(metrics_df):
    min_nmi, max_nmi = metrics_df["NMI"].min(), metrics_df["NMI"].max()
    min_mori, max_mori = metrics_df["Moran's I"].min(), metrics_df["Moran's I"].max()
    min_ilisi, max_ilisi = metrics_df["iLISI"].min(), metrics_df["iLISI"].max()

    bio_score = 1 / 2 * (
        (metrics_df["NMI"] - min_nmi) / (max_nmi - min_nmi) + 
        (metrics_df["Moran's I"] - min_mori) / (max_mori - min_mori)
    )
    batch_score = (metrics_df["iLISI"] - min_ilisi) / (max_ilisi - min_ilisi)
    metrics_df['bio'] = bio_score
    metrics_df['batch'] = batch_score
    metrics_df['total_score'] = 0.6 * bio_score + 0.4 * batch_score
    

for metrics_sub in metrics.values():
    # metrics_sub['total_score'] = total_score_formula(metrics_sub)
    add_total_score_formula(metrics_sub)
    metrics_sub['This paper'] = metrics_sub.index.isin(['CYC', 'VAMP', 'VAMP+CYC'])

# %%
xlims = {}
for metric in list(metric_map.values()) + ['total_score']:
    mins = []
    maxs = []
    for data in metrics.values():
        mins.append(data[metric].min())
        maxs.append(data[metric].max())
    x_min = min(mins)
    x_max = max(maxs)
    x_buffer = (x_max - x_min) * 0.15
    x_min = x_min - x_buffer
    x_max = x_max + x_buffer
    xlims[metric] = (x_min, x_max)

# Plots
n_rows = len(metrics)
n_cols = len(metric_map)# + 1  # Add one column for the scatter plot
fig, axs = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, len(metric_map) / 1.5 * n_rows),
                        sharey=False, sharex='col')

for row, (dataset_name, metrics_sub) in enumerate(metrics.items()):
    for col, metric in enumerate(list(metric_map.values())):
        ax = axs[row, col]
        means = metrics_sub.groupby('model')[metric].mean().reset_index()
        sb.swarmplot(y='model', x=metric, data=metrics_sub, ax=ax,
                     edgecolor='k', linewidth=0.25,
                     hue='model', palette=model_cmap, s=5, zorder=1)
        sb.scatterplot(y='model', x=metric, data=means, ax=ax,
                       edgecolor='k', linewidth=2.5,
                       color='k', s=150, marker='|', zorder=2)
        # Make pretty
        ax.set_xlim(xlims[metric][0], xlims[metric][1])
        ax.get_legend().remove()
        ax.tick_params(axis='y', which='major', length=0)
        ax.set(facecolor=(0, 0, 0, 0))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if row != (n_rows - 1):
            ax.set_xlabel('')
            ax.tick_params(axis='x', which='major', length=0)
        else:
            ax.locator_params(axis='x', nbins=5)
        if row == 0:
            title = metric_meaning_map[metric_map_rev[metric]] if metric != 'total_score' else 'Total Score'
            ax.set_title(title, fontsize=10)
        if col == 0:
            ax.set_ylabel(dataset_name.replace(" ", "\n"))
        else:
            ax.set_ylabel('')
            ax.tick_params(axis='y', which='both', labelleft=False)
        ax.grid(which='major', axis='x', linestyle=(0, (5, 10)), lw=0.5)

fig.set(facecolor=(0, 0, 0, 0))
plt.subplots_adjust(wspace=0.3, hspace=0.05)

# Save
plt.savefig(path_fig + 'combined_performance.pdf', dpi=300, bbox_inches='tight')
plt.savefig(path_fig + 'combined_performance.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
xlims = {}
for metric in list(metric_map.values()) + ['total_score']:
    mins = []
    maxs = []
    for data in metrics.values():
        mins.append(data[metric].min())
        maxs.append(data[metric].max())
    x_min = min(mins)
    x_max = max(maxs)
    x_buffer = (x_max - x_min) * 0.15
    x_min = x_min - x_buffer
    x_max = x_max + x_buffer
    xlims[metric] = (x_min, x_max)

# Plots
n_rows = len(metrics)
n_cols = len(metric_map) + 1  # Add one column for the scatter plot
fig, axs = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, len(metric_map) / 1.5 * n_rows),
                        sharey=False, sharex='col')

for row, (dataset_name, metrics_sub) in enumerate(metrics.items()):
    for col, metric in enumerate(list(metric_map.values())):
        ax = axs[row, col]
        means = metrics_sub.groupby('model')[metric].mean().reset_index()
        sb.swarmplot(y='model', x=metric, data=metrics_sub, ax=ax,
                     edgecolor='k', linewidth=0.25,
                     hue='model', palette=model_cmap, s=5, zorder=1)
        sb.scatterplot(y='model', x=metric, data=means, ax=ax,
                       edgecolor='k', linewidth=2.5,
                       color='k', s=150, marker='|', zorder=2)
        # Make pretty
        ax.set_xlim(xlims[metric][0], xlims[metric][1])
        ax.get_legend().remove()
        ax.tick_params(axis='y', which='major', length=0)
        ax.set(facecolor=(0, 0, 0, 0))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if row != (n_rows - 1):
            ax.set_xlabel('')
            ax.tick_params(axis='x', which='major', length=0)
        else:
            ax.locator_params(axis='x', nbins=5)
        if row == 0:
            title = metric_meaning_map[metric_map_rev[metric]] if metric != 'total_score' else 'Total Score'
            ax.set_title(title, fontsize=10)
        if col == 0:
            ax.set_ylabel(dataset_name.replace(" ", "\n"))
        else:
            ax.set_ylabel('')
            ax.tick_params(axis='y', which='both', labelleft=False)
        ax.grid(which='major', axis='x', linestyle=(0, (5, 10)), lw=0.5)

    ax = axs[row, -1]
    if row == 0:
        ax.set_title("Bio vs. Batch", fontsize=10)
    # Add scatter plot for bio vs batch as the last column
    # Make runs in small dots and avg in large dots
    sb.scatterplot(data=metrics_sub, x='batch', y='bio', hue='model', ax=ax, style="This paper",
                   palette=model_cmap, s=10, edgecolor='k', linewidth=0.1)
    metrics_sub_avg=metrics_sub.groupby('model')[['bio','batch','This paper']].mean().reset_index().rename({'model':'Model'},axis=1)
    metrics_sub_avg['This paper']=metrics_sub_avg['This paper'].map({1:'Yes',0:'No'})
    sb.scatterplot(
        data=metrics_sub_avg,
        x='batch', y='bio', hue='Model', ax=ax, style="This paper",
        palette=model_cmap, s=40, edgecolor='k', linewidth=0.3)

    # Calculate and plot the average for each method
    # for model_name in metrics_sub['model'].unique():
    #     model_data = metrics_sub[metrics_sub['model'] == model_name]
    #     avg_batch = model_data['batch'].mean()
    #     avg_bio = model_data['bio'].mean()
    #     ax.scatter(avg_batch, avg_bio, color='k', s=20, marker="D", zorder=-3)  # Black dot for average
    
    #     # Annotate with the model name
    #     ax.annotate(model_name, (avg_batch, avg_bio), textcoords="offset points", xytext=(7, 0),
    #                 ha='left', fontsize=8, color='k', zorder=-4)

    
    ax.set_ylabel('Bio')
    ax.set_xlabel('Batch')
    # Remove top and right solid border lines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, which='major', linestyle='--', linewidth=0.5)
    if row>0:
        ax.get_legend().remove()
    else:
        # Add plot b legend
        handles,labels=ax.get_legend_handles_labels()
        keep_legend_part=int(len(handles)/2)
        handles=handles[keep_legend_part:]
        labels=labels[keep_legend_part:]
        handle_run=deepcopy(handles[1])
        handle_mean=deepcopy(handles[1])
        # handle_run.set_markerfacecolor('k')
        # handle_run.set_markersize(3)
        # handle_mean.set_markerfacecolor('k')
        handles=handles+[
            deepcopy(handles[0]),
            handle_run,
            handle_mean,
            
        ]
        labels=labels+[
            'Measured',
            'Run',
            'Average'
        ]
        ax.legend(handles=handles,labels=labels, 
                  bbox_to_anchor=(1.05,0.95))


fig.set(facecolor=(0, 0, 0, 0))
fig.subplots_adjust(wspace=0.1, hspace=0.05)

# Move the last column away from the rest of the plot
for ax in axs[:,-1]:
    ax.set_position([ax.get_position().x0+0.07,ax.get_position().y0,ax.get_position().width,ax.get_position().height])

# Save
fig.savefig(path_fig + 'combined_performance_and_scatter.pdf', dpi=300, bbox_inches='tight')
fig.savefig(path_fig + 'combined_performance_and_scatter.png', dpi=300, bbox_inches='tight')
fig.show()

# %%

# %%

# %%
metric_meaning_map={'nmi_opt': 'Bio (coarse)', 'moransi': 'Bio (fine)', 'ilisi_system': 'Batch'}
metric_map_rev={'NMI': 'nmi_opt', "Moran's I": 'moransi', 'iLISI': 'ilisi_system'}
metric_map={'nmi_opt': 'NMI', 'moransi': "Moran's I", 'ilisi_system': 'iLISI'}

# %%

# %% [markdown]
# Significance of difference between pairs of models per metric and datatset. Pvalue adjustment is performed per metric and dataset across model pairs.

# %%
# Significance of model differences
pvals=[]
for dataset_name,metrics_sub in metrics.items():
    models = metrics_sub['model'].cat.categories
    n_models = len(models)
    for metric in metric_map.values():
        pvals_sub=[]
        for m1 in range(n_models-1):
            for m2 in range(m1+1,n_models):
                model1=models[m1]
                model2=models[m2]
                try:
                    t,p=ttest_ind(
                        metrics_sub.loc[model1,metric], metrics_sub.loc[model2,metric], 
                        equal_var=False, alternative='two-sided')
                except Exception as e:
                    print(e)
                    print(f"{dataset_name}: Could not find metrics for {metric} for at least {model1} or {model2}")
                    t,p=0., 1.
                pvals_sub.append(dict(
                    dataset=dataset_name,metric=metric,p=p,t=t, 
                    model_cond=model1,model_ctrl=model2,
                    # This does not account if t is exactly 0, but this would be very unlikely
                    higher=model1 if t>0 else model2))
        pvals_sub=pd.DataFrame(pvals_sub)
        padj_method='fdr_tsbh'
        pvals_sub['padj_'+padj_method]=multipletests(pvals_sub['p'].values, method=padj_method)[1]
        pvals.append(pvals_sub)
pvals=pd.concat(pvals)
pvals.to_csv(path_tab+'performance-score_topsettings-pairwise_significance.tsv',sep='\t',index=None)

# %% [markdown]
# ### UMAPs
# UMAP of representative run from top setting

# %%
# UMAP plot per data setting 

# Column names of colored covariates
ct_col_name='Cell type'
sys_col_name='System'
sample_col_name='Sample'
for dataset,dataset_name in dataset_map.items():
    embeds_ds={k:v for k,v in embeds[dataset_name].items() if k not in drop_models}
    ncols=3
    nrows=len(embeds_ds)
    fig,axs=plt.subplots(nrows,ncols,figsize=(2*ncols,2*nrows))
    # Args for ct, system, sample col names - same in all models
    args_sub=list(args[dataset_name].values())[0]
    # Plot every model
    for irow,model_name in enumerate([m for m in model_map.values() if m in embeds_ds]):
        embed=embeds_ds[model_name]
        if 'X_umap' not in embed.obsm:
            print(f"'X_umap' not in embed.obsm for {model_name}")
            continue
            
        for icol,(col_name,col) in enumerate(zip(
            [ct_col_name,sys_col_name,sample_col_name],
            [args_sub['group_key'],args_sub['system_key'],args_sub['batch_key']])):

            # Set cmap and col val names
            cmap=obs_col_cmap[dataset_map_rev[dataset_name]][col]
            if col_name==sys_col_name:
                # Map system to str as done in integrated embeds but not in non-int
                embed.obs[col+'_parsed']=embed.obs[col].astype(str).map(system_map[dataset])
                cmap={system_map[dataset][k]:v for k,v in cmap.items()}
            elif col_name==ct_col_name:
                # Map system to str as done in integrated embeds but not in non-int
                embed.obs[col+'_parsed']=embed.obs[col].astype(str).map(cell_type_map[dataset])
                cmap={cell_type_map[dataset][k]:v for k,v in cmap.items()}
            else:
                embed.obs[col+'_parsed']=embed.obs[col]

            # Plot
            ax=axs[irow,icol]
            sc.pp.subsample(embed, fraction=1.)
            sc.pl.umap(embed,color=col+'_parsed',ax=ax,show=False,
                      palette=cmap, frameon=False,title='')

            # Make pretty
            if irow==0:
                ax.set_title(col_name+'\n',fontsize=10)

            if icol==0:
                ax.axis('on')
                ax.tick_params(
                        top='off', bottom='off', left='off', right='off', 
                        labelleft='on', labelbottom='off')
                ax.set_ylabel(model_name+'\n',rotation=90)
                ax.set_xlabel('')
                ax.set(frame_on=False)

            if irow!=(nrows-1) or col_name==sample_col_name:
                ax.get_legend().remove()
            else:
                ax.legend(bbox_to_anchor=(0.4,-1),frameon=False,title=col_name,
                          ncol=math.ceil(embed.obs[col].nunique()/10))

    fig.set(facecolor = (0,0,0,0))
    
    # Save
    plt.savefig(path_fig+f'performance-embed_{dataset}_topsettings-umap.pdf',
                dpi=300,bbox_inches='tight')
    plt.savefig(path_fig+f'performance-embed_{dataset}_topsettings-umap.png',
                dpi=300,bbox_inches='tight')

# %% [markdown]
# ## All runs
# Results of all runs

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
            run_stats_ = pd.Series({'is_top': run_id in top_runs})
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
    
    display(res_sub.groupby(['model_parsed','param_parsed','genes_parsed'],observed=True).size())
    
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
ress = ress.query('testing == False')
ress.drop_duplicates(['model_parsed', 'param_parsed', 'genes_parsed', 'dataset_parsed', 'name', 'seed', 'params_opt', 'param_opt_val'], inplace=True)

# %% [markdown]
# ### Metric scores
# Overview of all metric results

# %%
(
    ress.query('model_parsed not in @drop_models')
    .groupby(['genes_parsed', 'model_parsed','param_parsed'],observed=True,sort=True
            ).size().index.to_frame().reset_index(drop=True)
)

# %%
# Plot model+opt_param * metrics+dataset
params=(
    ress.query('model_parsed not in @drop_models')
    .groupby(['genes_parsed', 'model_parsed','param_parsed'],observed=True,sort=True
            ).size().index.to_frame().reset_index(drop=True)
)
nrow=params.shape[0]
n_metrics=len(metric_map)
ncol=ress['dataset_parsed'].nunique()*n_metrics
fig,axs=plt.subplots(nrow,ncol,figsize=(ncol*1.9,nrow*2),sharex='col',sharey='row')
for icol_ds, (dataset_name,res_ds) in enumerate(ress.groupby('dataset_parsed')):
    
    # Max row for ds - some models not in all ds
    models_parsed_ds=set(res_ds.model_parsed)
    params_parsed_ds=set(res_ds.param_parsed)
    genes_parsed_ds=set(res_ds.genes_parsed)
    irow_max_ds=max([irow for irow,row in params.iterrows() if 
     row.model_parsed in models_parsed_ds and 
     row.param_parsed in params_parsed_ds and
     row.genes_parsed in genes_parsed_ds])
    
    # Plot metric + opt param settings
    for icol_metric,(metric,metric_name) in enumerate(metric_map.items()):
        icol=icol_ds*n_metrics+icol_metric
        for irow,(_,param_data) in enumerate(params.iterrows()):
            ax=axs[irow,icol]
            res_sub=res_ds.query(
                f'model_parsed=="{param_data.model_parsed}" & '+\
                f'param_parsed=="{param_data.param_parsed}" & '+\
                f'genes_parsed=="{param_data.genes_parsed}"')
            if res_sub.shape[0]>0:
                res_sub=res_sub.copy()
                res_sub['param_opt_val_str']=\
                    res_sub['param_opt_val_str'].cat.remove_unused_categories()
                # Plot
                sb.swarmplot(x=metric,y='param_opt_val_str',
                             hue="is_top",
                             # hue='param_opt_val_str',
                             data=res_sub,ax=ax, 
                             palette='tab10')
                
                # Make pretty
                ax.set(facecolor = metric_background_cmap[metric])
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.grid(axis='x', linestyle='--', color='gray')
                ax.get_legend().remove()
                if irow!=irow_max_ds:
                    ax.set_xlabel('')
                else:
                    # Add xaxis
                    # Must turn label to visible as sb will set it off if sharex
                    # Must reset ticks as will be of due to sharex
                    ax.set_xlabel(metric_name,visible=True)
                    ax.xaxis.set_ticks_position('bottom')
                if irow==0:
                    title=''
                    if icol%3==0:
                        title=title+dataset_name+'\n\n'
                    ax.set_title(title+metric_meaning_map[metric]+'\n',fontsize=10)
                if icol==0:
                    ax.set_ylabel(
                        param_data.model_parsed+' '+param_data.genes_parsed+'\n'+\
                        param_data.param_parsed+'\n')
                else:
                    ax.set_ylabel('')
            else:
                ax.remove()
            

plt.subplots_adjust(wspace=0.2,hspace=0.2)
fig.set(facecolor = (0,0,0,0))

# Turn off tight layout as it messes up spacing if adding xlabels on intermediate plots
#fig.tight_layout()

# Save
plt.savefig(path_fig+'performance-score_all-swarm.pdf',
            dpi=300,bbox_inches='tight')
plt.savefig(path_fig+'performance-score_all-swarm.png',
            dpi=300,bbox_inches='tight')


# %%

# %%

# %%
