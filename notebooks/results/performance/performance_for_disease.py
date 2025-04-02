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
# ## All Runs

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

# %% [markdown]
# ### Metric scores

# %%
model_cmap

# %%
model_order = [
    "VAMP+CYC", "CYC", "VAMP",
    "cVAE", "scVI",
    "Harmony",
    "Seaurat", "Harmony-py", "GLUE",
    "SATURN", "SATURN-CT"
]
drop_models = [
    "SysVI",
    "SysVI-stable",
]

# %%
ress.query('model not in @drop_models', inplace=True)

# %%

# %% [markdown]
# ## Add disease annotations for pancreas data

# %%
mm_pancreas_adata = sc.read(os.path.expanduser("~/data/cxg/f6044f53-41de-4654-8437-0d22b17dfd31.h5ad"), backed='r')
hs_pancreas_adata = sc.read("/om2/user/khrovati/data/cross_system_integration/pancreas_conditions_MIA_HPAP2/combined_orthologuesHVG.h5ad", backed='r')
mm_pancreas_adata, hs_pancreas_adata

# %%

# %% [markdown]
# ## Calculate metrics for different setups

# %%
setup_dicts = {
    'test_1': {
        'dataset_name': 'Retina Organoid-Tissue',
        'description': 'Compare organoid with periphery (similar) and fovea (dissimilar).',
        'description_short': 'Organoid vs \nperiphery (+) and fovea (-).',
        'embed_subset': lambda embed: embed[embed.obs['cell_type'].astype(str) == 'Mueller cell'],
        'group_similar': lambda embed_subset: embed_subset[(embed_subset.obs['system'].astype(str) == '0') | (embed_subset.obs['region'].astype(str) == 'periphery')],
        'group_dissimilar': lambda embed_subset: embed_subset[(embed_subset.obs['system'].astype(str) == '0') | (embed_subset.obs['region'].astype(str) == 'fovea')],
        'ignore_models': ['Non-integrated', 'Harmony'],
        'plot_cols': ['system', 'region'],
    },
    'test_2': {
        'dataset_name': 'Pancreas Mouse-Human',
        'description': 'Compare mouse T2D with T2D human (similar) and normal human (dissimilar).',
        'description_short': 'Mouse T2D vs \nhuman T2D (+) and healthy (-)',
        'embed_subset': lambda embed: embed[embed.obs['cell_type_eval'].astype(str) == 'beta'],
        'group_similar': lambda embed_subset: embed_subset[
            ((embed_subset.obs['system'].astype(str) == '0') & (embed_subset.obs['mm_Diabetes Status'].astype(str) == 'type 2 diabetes mellitus')) | 
            ((embed_subset.obs['system'].astype(str) == '1') & (embed_subset.obs['hs_Diabetes Status'].astype(str) == 'T2D'))
        ],
        'group_dissimilar': lambda embed_subset: embed_subset[
            ((embed_subset.obs['system'].astype(str) == '0') & (embed_subset.obs['mm_Diabetes Status'].astype(str) == 'type 2 diabetes mellitus')) | 
            ((embed_subset.obs['system'].astype(str) == '1') & (embed_subset.obs['hs_Diabetes Status'].astype(str) == 'ND'))
        ],
        'ignore_models': ['Non-integrated', 'Harmony'],
        'plot_cols': ['system', 'mm_Diabetes Status', 'hs_Diabetes Status'],
    },
    'test_3': {
        'dataset_name': 'Pancreas Mouse-Human',
        'description': 'Compare T2D human with T2D mouse (similar) and normal mouse (dissimilar).',
        'description_short': 'Human T2D vs \nmouse T2D (+) and healthy (-)',
        'embed_subset': lambda embed: embed[embed.obs['cell_type_eval'].astype(str) == 'beta'],
        'group_similar': lambda embed_subset: embed_subset[
            ((embed_subset.obs['system'].astype(str) == '0') & (embed_subset.obs['mm_Diabetes Status'].astype(str) == 'type 2 diabetes mellitus')) | 
            ((embed_subset.obs['system'].astype(str) == '1') & (embed_subset.obs['hs_Diabetes Status'].astype(str) == 'T2D'))
        ],
        'group_dissimilar': lambda embed_subset: embed_subset[
            ((embed_subset.obs['system'].astype(str) == '0') & (embed_subset.obs['mm_Diabetes Status'].astype(str) == 'normal')) | 
            ((embed_subset.obs['system'].astype(str) == '1') & (embed_subset.obs['hs_Diabetes Status'].astype(str) == 'T2D'))
        ],
        'ignore_models': ['Non-integrated', 'Harmony'],
        'plot_cols': ['system', 'mm_Diabetes Status', 'hs_Diabetes Status'],
    },
    'test_4': {
        'dataset_name': 'Skin Mouse-Human',
        'description': 'Compare mouse epidermolysis bullosa with human psoriasis (similar) and normal human (dissimilar).',
        'description_short': 'Mouse disease vs \nhuman disease (+) and healthy (-)',
        'embed_subset': lambda embed: embed[embed.obs['cell_type_eval'].astype(str) == 'Fibroblast'],
        'group_similar': lambda embed_subset: embed_subset[
            ((embed_subset.obs['system'].astype(str) == '0') & (embed_subset.obs['mm_disease'].astype(str) == 'epidermolysis bullosa')) | 
            ((embed_subset.obs['system'].astype(str) == '1') & (embed_subset.obs['hs_disease'].astype(str) == 'psoriasis'))
        ],
        'group_dissimilar': lambda embed_subset: embed_subset[
            ((embed_subset.obs['system'].astype(str) == '0') & (embed_subset.obs['mm_disease'].astype(str) == 'epidermolysis bullosa')) | 
            ((embed_subset.obs['system'].astype(str) == '1') & (embed_subset.obs['hs_disease'].astype(str) == 'normal'))
        ],
        'ignore_models': ['Non-integrated', 'Harmony'],
        'plot_cols': ['system', 'mm_disease', 'hs_disease'],
    },
    'test_5': {
        'dataset_name': 'Skin Mouse-Human',
        'description': 'Compare human psoriasis with mouse epidermolysis bullosa (similar) and normal mouse (dissimilar).',
        'description_short': 'Human disease vs \nmouse disease (+) and healthy (-)',
        'embed_subset': lambda embed: embed[embed.obs['cell_type_eval'].astype(str) == 'Fibroblast'],
        'group_similar': lambda embed_subset: embed_subset[
            ((embed_subset.obs['system'].astype(str) == '0') & (embed_subset.obs['mm_disease'].astype(str) == 'epidermolysis bullosa')) | 
            ((embed_subset.obs['system'].astype(str) == '1') & (embed_subset.obs['hs_disease'].astype(str) == 'psoriasis'))
        ],
        'group_dissimilar': lambda embed_subset: embed_subset[
            ((embed_subset.obs['system'].astype(str) == '0') & (embed_subset.obs['mm_disease'].astype(str) == 'normal')) | 
            ((embed_subset.obs['system'].astype(str) == '1') & (embed_subset.obs['hs_disease'].astype(str) == 'psoriasis'))
        ],
        'ignore_models': ['Non-integrated', 'Harmony'],
        'plot_cols': ['system', 'mm_disease', 'hs_disease'],
    },
    'test_6': {
        'dataset_name': 'Limited Skin Mouse-Human',
        'description': 'Compare normal human with and normal mouse (similar) and mouse epidermolysis bullosa (dissimilar).',
        'description_short': 'Human healthy vs \nmouse healthy (+) and disease (-)',
        'embed_subset': lambda embed: embed[embed.obs['cell_type_eval'].astype(str) == 'Fibroblast'],
        'group_similar': lambda embed_subset: embed_subset[
            ((embed_subset.obs['system'].astype(str) == '0') & (embed_subset.obs['mm_disease'].astype(str) == 'normal')) | 
            ((embed_subset.obs['system'].astype(str) == '1') & (embed_subset.obs['hs_disease'].astype(str) == 'normal'))
        ],
        'group_dissimilar': lambda embed_subset: embed_subset[
            ((embed_subset.obs['system'].astype(str) == '0') & (embed_subset.obs['mm_disease'].astype(str) == 'epidermolysis bullosa')) | 
            ((embed_subset.obs['system'].astype(str) == '1') & (embed_subset.obs['hs_disease'].astype(str) == 'normal'))
        ],
        'ignore_models': ['Non-integrated', 'Harmony'],
        'plot_cols': ['system', 'mm_disease', 'hs_disease'],
    },
    'test_7': {
        'dataset_name': 'Skin Mouse-Human',
        'description': 'Compare normal human with normal mouse (similar) and mouse epidermolysis bullosa (dissimilar).',
        'description_short': 'Human healthy vs \nmouse healthy (+) and disease (-)',
        'embed_subset': lambda embed: embed[embed.obs['cell_type_eval'].astype(str) == 'Fibroblast'],
        'group_similar': lambda embed_subset: embed_subset[
            ((embed_subset.obs['system'].astype(str) == '0') & (embed_subset.obs['mm_disease'].astype(str) == 'normal')) | 
            ((embed_subset.obs['system'].astype(str) == '1') & (embed_subset.obs['hs_disease'].astype(str) == 'normal'))
        ],
        'group_dissimilar': lambda embed_subset: embed_subset[
            ((embed_subset.obs['system'].astype(str) == '0') & (embed_subset.obs['mm_disease'].astype(str) == 'epidermolysis bullosa')) | 
            ((embed_subset.obs['system'].astype(str) == '1') & (embed_subset.obs['hs_disease'].astype(str) == 'normal'))
        ],
        'ignore_models': ['Non-integrated', 'Harmony'],
        'plot_cols': ['system', 'mm_disease', 'hs_disease'],
    },
    'test_8': {
        'dataset_name': 'Skin Mouse-Human',
        'description': 'Compare human psoriasis with normal mouse (similar) and mouse epidermolysis bullosa (dissimilar).',
        'description_short': 'Human Disease vs \nmouse healthy (+) and disease (-)',
        'embed_subset': lambda embed: embed[embed.obs['cell_type_eval'].astype(str) == 'Fibroblast'],
        'group_similar': lambda embed_subset: embed_subset[
            ((embed_subset.obs['system'].astype(str) == '0') & (embed_subset.obs['mm_disease'].astype(str) == 'normal')) | 
            ((embed_subset.obs['system'].astype(str) == '1') & (embed_subset.obs['hs_disease'].astype(str) == 'psoriasis'))
        ],
        'group_dissimilar': lambda embed_subset: embed_subset[
            ((embed_subset.obs['system'].astype(str) == '0') & (embed_subset.obs['mm_disease'].astype(str) == 'epidermolysis bullosa')) | 
            ((embed_subset.obs['system'].astype(str) == '1') & (embed_subset.obs['hs_disease'].astype(str) == 'psoriasis'))
        ],
        'ignore_models': ['Non-integrated', 'Harmony'],
        'plot_cols': ['system', 'mm_disease', 'hs_disease'],
    },
}

# %%
from scib_metrics.nearest_neighbors import pynndescent
from scib_metrics import lisi_knn, ilisi_knn

def calculate_neighbors(adata, obsm_emb_key, k=100, n_jobs=1, neighbor_computer=None):
    # Calculates `k` nearest neighbors in the space defined by `obsm_emb_key` of `adata`
    # One can pass `neighbor_computer` as in scib_metrics. Default is good enough.
    if neighbor_computer is not None:
        neigh_output = neighbor_computer(adata.obsm[obsm_emb_key], k)
    neigh_output = pynndescent(adata.obsm[obsm_emb_key], n_neighbors=k, random_state=0, n_jobs=n_jobs)
    #indices, distances = neigh_output.indices, neigh_output.distances
    return neigh_output


# %%

# %%

# %%
for setup_name, setup_dict in setup_dicts.items():
    print(setup_name, setup_dict)

    dataset_name = setup_dict['dataset_name']
    fn_embed_subset = setup_dict['embed_subset']
    fn_group_similar = setup_dict['group_similar']
    fn_group_dissimilar = setup_dict['group_dissimilar']
    ignore_models = setup_dict['ignore_models']
    plot_cols = setup_dict['plot_cols']

    ress_subset = ress[ress['dataset_parsed'] == setup_dict['dataset_name']]
    for i, run_info in ress_subset.sample(frac=1).iterrows():
        print(run_info['run_path'])
        print(datetime.now())

        (Path(path_data) / "performance_disease" / run_info['dataset_parsed']).mkdir(parents=True, exist_ok=True)
        result_filename = Path(path_data) / "performance_disease" / run_info['dataset_parsed'] / f"results_{run_info['run_id']}_{setup_name}.pkl"
        if result_filename.exists():
            print("Already calculated")
            continue

        try:
            print(f">>> Writing to {result_filename}")
            result_filename.touch()
            if Path(run_info['run_path'] + '/embed_full.h5ad').exists():
                embed = sc.read_h5ad(run_info['run_path'] + '/embed_full.h5ad')
            else:
                embed = sc.read_h5ad(run_info['run_path'] + '/embed.h5ad')
            if dataset_name == 'Pancreas Mouse-Human':
                embed.obs['mm_Diabetes Status'] = mm_pancreas_adata.obs['disease']
                embed.obs['hs_Diabetes Status'] = hs_pancreas_adata.obs['hs_Diabetes Status']
            embed_subset = fn_embed_subset(embed).copy()
            embed_subset.obsm['method'] = embed_subset.X.copy()
    
            embed_subset_s = fn_group_similar(embed_subset).copy()
            embed_subset_d = fn_group_dissimilar(embed_subset).copy()
    
            result_dict = {
                'setup_name': setup_name,
                'dataset_name': dataset_name,
                'run_id': run_info['run_id'],
            }
            try:
                results_dict['seed'] = args_run['seed']
                results_dict['name'] = args_run['name']
            except:
                pass
            for embed_, key in [(embed_subset_s, "similar"), (embed_subset_d, "dissimilar")]:
                ng_output = calculate_neighbors(embed_, obsm_emb_key='method')
                sp_distances, sp_conns = sc.neighbors._compute_connectivities_umap(
                    ng_output.indices, ng_output.distances, 
                    ng_output.indices.shape[0], n_neighbors=ng_output.indices.shape[1],
                )
                ilisi_value = ilisi_knn(sp_distances, embed_.obs['system'].values)
                result_dict[f"ilisi_{key}"] = ilisi_value
            
            print(result_dict)
            with open(result_filename, 'wb') as f:
                pkl.dump(result_dict, f)
        except Exception as e:
            print(e)
            result_filename.unlink()
            raise e

# %%

# %%

# %%
results_list = []

# %%
for setup_name, setup_dict in setup_dicts.items():
    print(setup_name, setup_dict)

    dataset_name = setup_dict['dataset_name']
    fn_embed_subset = setup_dict['embed_subset']
    fn_group_similar = setup_dict['group_similar']
    fn_group_dissimilar = setup_dict['group_dissimilar']
    ignore_models = setup_dict['ignore_models']
    plot_cols = setup_dict['plot_cols']

    ress_subset = ress[ress['dataset_parsed'] == setup_dict['dataset_name']]
    for i, run_info in ress_subset.sample(frac=1).iterrows():
        print(run_info['run_path'])
        print(datetime.now())

        (Path(path_data) / "performance_disease" / run_info['dataset_parsed']).mkdir(parents=True, exist_ok=True)
        result_filename = Path(path_data) / "performance_disease" / run_info['dataset_parsed'] / f"results_{run_info['run_id']}_{setup_name}.pkl"
        if not result_filename.exists():
            print(f"File does not exists: {result_filename}")
            continue

        print(">>", result_filename)
        with open(result_filename, "rb") as f:
            result_dict = pkl.load(f)
            print(result_dict)
        ress_row = ress.loc[result_dict['run_id']]
        result_dict = {
            **result_dict,
            **ress_row.to_dict(),
        }

        results_list.append(result_dict)

# %%

# %%
results_df = pd.DataFrame(results_list)
results_df.to_csv(os.path.expanduser("~/tmp/sysvi_paired_comparison_lisi_scores__results_df.csv"))


# %%
results_df

# %%

# %%

# %%

# %%

# %% [markdown]
# # Plot results

# %%
import pandas as pd
results_df = pd.read_csv(os.path.expanduser("~/tmp/sysvi_paired_comparison_lisi_scores__results_df.csv"))
results_df['model_parsed'] = pd.Categorical(values=results_df['model_parsed'], categories=model_map.values(), ordered=True)
results_df

# %%
results_df = results_df.query('testing == False')
results_df.drop_duplicates(['setup_name', 'model_parsed', 'param_parsed', 'genes_parsed', 'dataset_parsed', 'name', 'seed', 'params_opt', 'param_opt_val'], inplace=True)

# %%
keep_setups = ['test_1', 'test_2', 'test_3', 'test_7', 'test_8', 'test_6']
results_df = results_df[results_df['setup_name'].astype(str).isin(keep_setups)].copy()
results_df['setup_name'] = pd.Categorical(values=results_df['setup_name'], categories=keep_setups, ordered=True)
results_df['setup_name'].unique()

# %%

# %%
(
    results_df.query('model_parsed not in @drop_models')
    .query('genes_parsed == "OTO"')
    .groupby(['model_parsed','param_parsed','genes_parsed'],observed=True,sort=True
            ).size().index.to_frame().reset_index(drop=True)
)

# %%
metric_meaning_map_ = {
    'ilisi_similar': 'ilisi Similar',
    'ilisi_dissimilar': 'ilisi Dissimilar',
}

# %%
# Plot model+opt_param * metrics+dataset
params=(
    results_df
    .query('model_parsed not in @drop_models')
    .query('genes_parsed == "OTO"')
    .groupby(['model_parsed','param_parsed','genes_parsed'],observed=True,sort=True
            ).size().index.to_frame().reset_index(drop=True)
)
nrow=params.shape[0]
n_metrics=2
ncol=results_df['setup_name'].nunique()*n_metrics
fig,axs=plt.subplots(nrow,ncol,figsize=(ncol*1.9,nrow*2),sharex='col',sharey='row')
for icol_ds, (setup_name,res_ds) in enumerate(results_df.groupby('setup_name')):
    # Max row for ds - some models not in all ds
    models_parsed_ds=set(res_ds.model_parsed)
    params_parsed_ds=set(res_ds.param_parsed)
    genes_parsed_ds=set(res_ds.genes_parsed)
    irow_max_ds=max([irow for irow,(model_parsed,param_parsed,genes_parsed) in params.iterrows() if 
     model_parsed in models_parsed_ds and 
     param_parsed in params_parsed_ds and
     genes_parsed in genes_parsed_ds])
    
    # Plot metric + opt param settings
    for icol_metric,(metric,metric_name) in enumerate([('ilisi_similar', 'ilisi Similar'),
                                                       ('ilisi_dissimilar', 'ilisi Dissimilar'),]):
        icol=icol_ds*n_metrics+icol_metric
        for irow,(_,param_data) in enumerate(params.iterrows()):
            dataset_name = setup_dicts[setup_name]['dataset_name']
            ax=axs[irow,icol]
            res_sub=res_ds.query(
                f'model_parsed=="{param_data.model_parsed}" & '+\
                f'param_parsed=="{param_data.param_parsed}" & '+\
                f'genes_parsed=="{param_data.genes_parsed}"')
            if res_sub.shape[0]>0:
                res_sub=res_sub.copy()
                res_sub['param_opt_val_str']=\
                    res_sub['param_opt_val_str'].astype('category').cat.remove_unused_categories()
                # Plot
                sb.swarmplot(x=metric,y='param_opt_val_str',
                             hue="is_top",
                             # hue='param_opt_val_str',
                             data=res_sub,ax=ax, 
                             palette='tab10')
                
                # Make pretty
                ax.set(facecolor = '#F9F3F3')
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
                    if icol%2==0:
                        title=title+dataset_name+'\n'+setup_dicts[setup_name]['description_short']+"\n\n"
                    ax.set_title(title+metric_meaning_map_[metric]+'\n',fontsize=10)
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
plt.savefig(path_fig+'performance-disease-score_all-swarm.pdf',
            dpi=300,bbox_inches='tight')
plt.savefig(path_fig+'performance-disease-score_all-swarm.png',
            dpi=300,bbox_inches='tight')

# %%

# %%

# %%

# %%
top_results_df = (
    results_df
    .query('is_top == True')
    .query('model_parsed not in @drop_models')
).copy()
top_results_df['model_parsed'] = pd.Categorical(values=top_results_df['model_parsed'], categories=[c for c in model_map.values() if c in top_results_df['model_parsed'].unique().tolist()], ordered=True)

# %%
import matplotlib.pyplot as plt
import seaborn as sb
import matplotlib.cm as cm
from matplotlib.lines import Line2D


# Define xlims dynamically for uniform axis limits
xlims = {}
metrics_list = [("ilisi_similar", "iLISI similar"), ("ilisi_dissimilar", "iLISI dissimilar")]

for metric, _ in metrics_list:
    mins = []
    maxs = []
    for _, subset_df in top_results_df.groupby('setup_name'):
        mins.append(subset_df[metric].min())
        maxs.append(subset_df[metric].max())
    x_min = min(mins)
    x_max = max(maxs)
    x_buffer = (x_max - x_min) * 0.15
    xlims[metric] = (x_min - x_buffer, x_max + x_buffer)

# Setup figure and axes
n_rows = len(top_results_df["setup_name"].unique())
n_cols = len(metrics_list) + 1  # +1 for the combined scatter plot
fig, axs = plt.subplots(n_rows, n_cols, figsize=(3.4 * n_cols, n_rows * 3.4),
                        sharey=False, sharex=False)

# Define colormap for this setup
unique_models = top_results_df["model_parsed"].cat.categories
cmap = cm.get_cmap("tab20", len(unique_models))
model_cmap = {model: cmap(i) for i, model in enumerate(unique_models)}
top_results_df['This paper'] = top_results_df['model_parsed'].isin(['CYC', 'VAMP', 'VAMP+CYC'])

# Generate plots
for row, (setup_name, subset_df) in enumerate(top_results_df.groupby('setup_name')):
    x_min_ = min(list(subset_df['ilisi_similar']) + list(subset_df['ilisi_dissimilar']))
    x_max_ = max(list(subset_df['ilisi_similar']) + list(subset_df['ilisi_dissimilar']))
    x_buffer_ = (x_max_ - x_min_) * 0.1
    xlims_ = (x_min_ - x_buffer_, x_max_ + x_buffer_)
    
    # Metric-specific plots
    for col, (metric, metric_title) in enumerate(metrics_list):
        ax = axs[row, col]
        means = subset_df.groupby('model_parsed')[metric].mean().reset_index()

        sb.swarmplot(y='model_parsed', x=metric, data=subset_df, ax=ax,
                     hue='model_parsed', palette=model_cmap, s=5, zorder=1, edgecolor='k', linewidth=0.25)
        sb.scatterplot(y='model_parsed', x=metric, data=means, ax=ax,
                       color='k', s=150, marker='|', zorder=2, edgecolor='k', linewidth=2.5)

        # Formatting
        ax.set_xlim(xlims_)
        ax.grid(True, which='major', axis='x', linestyle='--', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if row == 0:
            ax.set_title(f"{metric_title}", fontsize=10)
        if col == 0:
            ax.set_ylabel(
                setup_dicts[setup_name]["dataset_name"].replace(" ", "\n") + "\n" + 
                setup_dicts[setup_name]['description_short']+"\n"
            )
            ax.set_xlabel('')
        else:
            ax.set_ylabel('')
            ax.set_xlabel('')
            ax.tick_params(axis='y', which='both', labelleft=False)
        ax.get_legend().remove()

    # Combined scatter plot
    ax = axs[row, -1]
    scatter = sb.scatterplot(data=subset_df, x="ilisi_similar", y="ilisi_dissimilar", style="This paper", style_order=[0, 1],
                             hue="model_parsed", palette=model_cmap, s=10, ax=ax, edgecolor='k', linewidth=0.3, legend=False)
    
    subset_df_avg=subset_df.groupby('model_parsed')[['ilisi_similar','ilisi_dissimilar','This paper']].mean().reset_index().rename({'model_parsed':'Model'},axis=1)
    subset_df_avg['This paper']=subset_df_avg['This paper'].map({1:'Yes',0:'No'})
    scatter = sb.scatterplot(data=subset_df_avg, x="ilisi_similar", y="ilisi_dissimilar", style="This paper", style_order=['No', 'Yes'],
                             hue="Model", palette=model_cmap, s=30, ax=ax, edgecolor='k', linewidth=0.3)
    
    ax.plot([xlims_[0], xlims_[1]],
            [xlims_[0], xlims_[1]], 'k--', linewidth=0.8)
    ax.set_xlim(xlims_)
    ax.set_ylim(xlims_)
    ax.set_title("", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_xlabel("iLISI similar")
    ax.set_ylabel("iLISI Dissimilar")
    ax.get_legend().remove()

# Add legend outside the plots
handles, labels = scatter.get_legend_handles_labels()
# fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1.02, 0.5), title="Models", fontsize=8)
handles, labels = plt.gca().get_legend_handles_labels()

points = [
    Line2D([0], [0], label='Style guide', marker='s', markersize=0, markeredgecolor='black', markerfacecolor='k', linestyle=''),
    Line2D([0], [0], label='Run', marker='s', markersize=3, markeredgecolor='black', markerfacecolor='k', linestyle=''),
    Line2D([0], [0], label='Average', marker='s', markersize=7, markeredgecolor='black', markerfacecolor='k', linestyle=''),
]
handles.extend(points)

plt.legend(handles=handles, loc='center left', bbox_to_anchor=(1.03, 4.5), fontsize=8)

# Final adjustments
plt.subplots_adjust(wspace=0.3, hspace=0.3)
plt.savefig(path_fig + "performance_disease_metrics_combined_scatter.pdf", dpi=300, bbox_inches="tight")
plt.savefig(path_fig + "performance_disease_metrics_combined_scatter.png", dpi=300, bbox_inches="tight")
plt.show()


# %%

# %%

# %%

# %%

# %%

# %%
