# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.6
#   kernelspec:
#     display_name: cs_integration
#     language: python
#     name: cs_integration
# ---

# %%
import anndata as ad
import scanpy as sc
import pickle as pkl
import pandas as pd
import numpy as np
import argparse
import os
import string
import subprocess

from matplotlib.pyplot import rcParams
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import random
import scglue
import networkx as nx

# %%
parser = argparse.ArgumentParser()
def intstr_to_bool(x):
    return bool(int(x))
def str_to_float_zeronone(x):
    if x is None or x=="0":
        return None
    else:
        return float(x)
parser.add_argument('-n', '--name', required=False, type=str, default=None,
                    help='name of replicate, if unspecified set to rSEED if seed is given '+\
                    'and else to blank string')
parser.add_argument('-s', '--seed', required=False, type=int, default=None,
                    help='random seed, if none it is randomly generated')
parser.add_argument('-po', '--params_opt', required=False, type=str, default='',
                    help='name of optimized params/test purpose')
parser.add_argument('-pa', '--path_adata', required=True, type=str,
                    help='full path to adata obj')
parser.add_argument('-ps', '--path_save', required=True, type=str,
                    help='directory path for saving, creates subdir within it')
parser.add_argument('-sk', '--system_key', required=True, type=str,
                    help='obs col with system info')
parser.add_argument('-gk', '--group_key', required=True, type=str,
                    help='obs col with group info')
parser.add_argument('-bk', '--batch_key', required=True, type=str,
                    help='obs col with batch info')
parser.add_argument('-me', '--max_epochs', required=False, type=int,default=200,
                    help='max_epochs for training')
parser.add_argument('-edp', '--epochs_detail_plot', required=False, type=int, default=20,
                    help='Loss subplot from this epoch on')

parser.add_argument('-nce', '--n_cells_eval', required=False, type=int, default=-1,  
                    help='Max cells to be used for eval, if -1 use all cells. '+\
                   'For cell subsetting seed 0 is always used to be reproducible accros '+\
                   'runs with different seeds.')

parser.add_argument('-t', '--testing', required=False, type=intstr_to_bool,default='0',
                    help='Testing mode')



parser.add_argument('--rel_gene_weight', required=False, type=float, default=1.,
                    help='Weight to connect a gene to another relevant gene in scGLUE')
parser.add_argument('--latent_dim', required=False, type=int, default=50,
                    help='Latent dim in scGLUE')
parser.add_argument('--h_depth', required=False, type=int, default=2,
                    help='depth of encoder in scGLUE')
parser.add_argument('--h_dim', required=False, type=int, default=256,
                    help='Dim of hidden layers in encoder of scGLUE')

parser.add_argument('--lam_data', required=False, type=float, default=1.0,
                    help='lam_data in scGLUE')
parser.add_argument('--lam_kl', required=False, type=float, default=1.0,
                    help='lam_kl in scGLUE')
parser.add_argument('--lam_graph', required=False, type=float, default=0.02,
                    help='lam_graph in scGLUE')
parser.add_argument('--lam_align', required=False, type=float, default=0.05,
                    help='lam_align in scGLUE')
# %%
if False:
    args= parser.parse_args(args=[
        '-pa','/Users/amirali.moinfar/Downloads/pancreas_conditions_MIA_HPAP2/combined_orthologuesHVG.h5ad',
        '-ps','/Users/amirali.moinfar/tmp/cross_species_prediction/eval/test/integration/',
        # '-pa','/om2/user/khrovati/data/cross_species_prediction/pancreas_healthy/combined_orthologuesHVG2000.h5ad',
        # '-ps','/om2/user/khrovati/data/cross_system_integration/eval/test/integration/',
        '-sk','system',
        '-gk','cell_type_eval',
        '-bk','batch',
        '-me','2',
        '-edp','0',
        
        '-s','1',
                
        '-nce','1000',
        
        '-t','1'
    ])
# Read command line args
else:
    args = parser.parse_args()
    
print(args)

TESTING=args.testing

if args.name is None:
    if args.seed is not None:
        args.name='r'+str(args.seed)

# %%
# scglue params
rel_gene_weight = args.rel_gene_weight
latent_dim=args.latent_dim
h_depth=args.h_depth
h_dim=args.h_dim

lam_data=args.lam_data
lam_kl=args.lam_kl
lam_graph=args.lam_graph
lam_align=args.lam_align

# %%
# Make folder for saving
path_save=args.path_save+'scglue'+\
    '_'+''.join(np.random.permutation(list(string.ascii_letters)+list(string.digits))[:8])+\
    ('-TEST' if TESTING else '')+\
    os.sep

os.mkdir(path_save)
print(path_save)

# %%
# Set seed for eval
# Set only here below as need randomness for generation of out directory name (above)
if args.seed is not None:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

# %%
# Save args
pkl.dump(args,open(path_save+'args.pkl','wb'))

# %% [markdown]
# ## Integration

# %% [markdown]
# ### Prepare data

# %%
# Load data
adata=sc.read(args.path_adata)
adata.obs[args.system_key] = adata.obs[args.system_key].astype("str")
adata

# %%
if TESTING:
    # Make data smaller if testing the script
    random_idx=np.random.permutation(adata.obs_names)[:5000]
    adata=adata[random_idx,:].copy()
    print(adata.shape)
    # Set some groups to nan for testing if this works
    adata.obs[args.group_key]=[np.nan]*10+list(adata.obs[args.group_key].iloc[10:])

# %%
total_mods = list(adata.obs[args.system_key].unique())
total_mods

# %%
mods_adata = {}
for mod in total_mods:
    mods_adata[mod] = adata[adata.obs[args.system_key] == mod].copy()
    mods_adata[mod].var.index = mods_adata[mod].var.index + f'-{mod}'
    print(f"mod: {mod}\n", mods_adata[mod])

# %% [markdown]
# ### Training

# %%
print('Train')

# %%
for adata_spc in mods_adata.values():
    adata_spc.X = adata_spc.layers['counts'].copy()
    sc.pp.normalize_total(adata_spc)
    sc.pp.log1p(adata_spc)
    sc.pp.scale(adata_spc)
    sc.tl.pca(adata_spc)
    
    scglue.models.configure_dataset(
        adata_spc, "NB", use_highly_variable=False, use_batch=args.batch_key,
        use_layer="counts", use_rep="X_pca"
    )

# %%
my_guidance = nx.DiGraph()
for mod in total_mods:
    adata_mod = mods_adata[mod]
    print(f"connecting {mod} to itself")
    for g in adata_mod.var.index:
        my_guidance.add_node(g)
        my_guidance.add_edge(g, g, **{'weight': 1.0, 'sign': 1, 'type': 'loop', 'id': '0'})

for mod in total_mods:
    adata_mod = mods_adata[mod]
    other_mods = [m for m in total_mods if m != mod]
    for omod in other_mods:
        adata_omod = mods_adata[omod]
        print(f"connecting {mod} to {omod}")
        for g in adata_mod.var.index:
            my_guidance.add_edge(g, g.split("-")[0]+f"-{omod}", **{'weight': rel_gene_weight, 'sign': 1, 'type': 'normal', 'id': '0'})

# %%
filename = path_save + "scglue_model"

# %%
glue = scglue.models.fit_SCGLUE(
    mods_adata, my_guidance,
    init_kws=dict(latent_dim=latent_dim, h_depth=h_depth, h_dim=h_dim),
    fit_kws={"directory": "glue", "max_epochs": args.max_epochs},
    compile_kws=dict(lam_data=lam_data, lam_kl=lam_kl, lam_graph=lam_graph, lam_align=lam_align), 
    balance_kws=None,
)

glue.save(filename)

# %%
glue = scglue.models.load_model(filename)

# %%
try:
    dx = scglue.models.integration_consistency(
        glue, mods_adata, my_guidance
    )
    _ = sns.lineplot(x="n_meta", y="consistency", data=dx).axhline(y=0.05, c="darkred", ls="--")
    print(dx)
    with open(path_save + "integration_consistency", 'wb') as f:
        pkl.dump(dx, f)
except Exception as e:
    print("Error:", e)

# %%

# %%

# %%

# %%

# %% [markdown]
# ### Eval

# %% [markdown]
# #### Losses

# %%

# %% [markdown]
# #### Embedding

# %%
print('Plot embedding')

# %%
for mod, adata_mod in mods_adata.items():
    adata_mod.obsm["X_glue"] = glue.encode_data(mod, adata_mod)
combined = ad.concat(mods_adata.values())

# %%
# Compute embedding
cells_eval=adata.obs_names if args.n_cells_eval==-1 else \
    np.random.RandomState(seed=0).permutation(adata.obs_names)[:args.n_cells_eval]
print('N cells for eval:',cells_eval.shape[0])
embed = sc.AnnData(combined[cells_eval].obsm['X_glue'], obs=combined[cells_eval,:].obs.copy())

# %%
# Use 90 neighbours so that this can be also used for lisi metrics
sc.pp.neighbors(embed, use_rep='X', n_neighbors=90)
sc.tl.umap(embed)

# %%
# Make system categorical, also for metrics below
embed.obs[args.system_key]=embed.obs[args.system_key].astype(str)

# %%
# Save embed
embed.write(path_save+'embed.h5ad')

# %%
# Plot embedding
rcParams['figure.figsize']=(8,8)
cols=[args.system_key,args.group_key,args.batch_key]
fig,axs=plt.subplots(len(cols),1,figsize=(8,8*len(cols)))
for col,ax in zip(cols,axs):
    sc.pl.umap(embed,color=col,s=10,ax=ax,show=False,sort_order=False)
plt.savefig(path_save+'umap.png',dpi=300,bbox_inches='tight')

# %% [markdown]
# #### Integration metrics

# %%
print('Run integration metrics')

# %%
if TESTING:
    args_metrics=[
        '-p','/om2/user/khrovati/data/cross_system_integration/eval/test/integration/example/',
        '-sk','system',
        '-gk','cell_type',
        '-bk','sample'
    ]

else:
    args_metrics=[
        '--path',path_save,
        '--system_key',args.system_key,
        '--group_key',args.group_key,
        '--batch_key',args.batch_key
    ]
process = subprocess.Popen(['python','run_metrics.py']+args_metrics, 
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
# Make sure that process has finished
res=process.communicate()
# Save stdout from the child script
for line in res[0].decode(encoding='utf-8').split('\n'):
     print(line)
# Check that child process did not fail - if this was not checked then
# the status of the whole job would be succesfull 
# even if the child failed as error wouldn be passed upstream
if process.returncode != 0:
    raise ValueError('Process failed with', process.returncode)

# %% [markdown]
# # End

# %%
print('Finished integration!')

# %%
