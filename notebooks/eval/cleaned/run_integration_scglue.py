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
#     display_name: scglue
#     language: python
#     name: scglue
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
parser.add_argument('-pa2', '--path_adata_2', required=False, type=str, default="",
                    help='full path to second adata obj')
parser.add_argument('-gg', '--gene_graph', required=False, type=str, default="",
                    help='path to tsv containing gene graph. columns are accessed by numbers and in the same order as given system_key in adata files.')
parser.add_argument('-ps', '--path_save', required=True, type=str,
                    help='directory path for saving, creates subdir within it')
parser.add_argument('-sk', '--system_key', required=True, type=str,
                    help='obs col with system info')
parser.add_argument('-gk', '--group_key', required=True, type=str,
                    help='obs col with group info')
parser.add_argument('-bk', '--batch_key', required=True, type=str,
                    help='obs col with batch info')
parser.add_argument('-pk', '--pca_key', required=False, type=str, default="",
                    help='key to obsm that contains X_pca calculated to each system.')
parser.add_argument('-me', '--max_epochs', required=False, type=int, default=-1,
                    help='max_epochs for training. -1 for AUTO detection by scGLUE.')
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
parser.add_argument('--n_latent', required=False, type=int, default=50,
                    help='Latent dim in scGLUE')
parser.add_argument('--n_layers', required=False, type=int, default=2,
                    help='depth of encoder in scGLUE')
parser.add_argument('--n_hidden', required=False, type=int, default=256,
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
        # Amir
        # With one data with common vars
        # '-pa','/Users/amirali.moinfar/Downloads/pancreas_conditions_MIA_HPAP2/combined_orthologuesHVG.h5ad',
        # With two data with gene graph
        #'-pa','/Users/amirali.moinfar/Downloads/pancreas_conditions_MIA_HPAP2/combined-hsPart_nonortholHVG.h5ad',
        #'-pa2','/Users/amirali.moinfar/Downloads/pancreas_conditions_MIA_HPAP2/combined-mmPart_nonortholHVG.h5ad',
        #'-gg','/Users/amirali.moinfar/Downloads/pancreas_conditions_MIA_HPAP2/combined_nonortholHVG_geneMapping.tsv',
        #'-ps','/Users/amirali.moinfar/tmp/cross_species_prediction/eval/test/integration/',
        
        # Karin
        '-ps','/om2/user/khrovati/data/cross_system_integration/eval/test/integration/',
        # 1 adata
        '-pa','/om2/user/khrovati/data/cross_system_integration/pancreas_conditions_MIA_HPAP2/test/combined_orthologuesHVG.h5ad',
        '-gg','/om2/user/khrovati/data/cross_system_integration/pancreas_conditions_MIA_HPAP2/combined_orthologuesHVG_geneMapping.tsv',
        # 2 adatas
        # '-pa','/om2/user/khrovati/data/cross_system_integration/pancreas_conditions_MIA_HPAP2/test/combined-mmPart_nonortholHVG.h5ad',
        # '-pa2','/om2/user/khrovati/data/cross_system_integration/pancreas_conditions_MIA_HPAP2/test/combined-hsPart_nonortholHVG.h5ad',
        # '-gg','/om2/user/khrovati/data/cross_system_integration/pancreas_conditions_MIA_HPAP2/combined_nonortholHVG_geneMapping.tsv',
        
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
    args, args_unknown = parser.parse_known_args()
    
print(args)

TESTING=args.testing

if args.name is None:
    if args.seed is not None:
        args.name='r'+str(args.seed)

# %%
SINGLE_ADATA = True
if args.path_adata_2 != "":
    assert args.gene_graph != ""
    SINGLE_ADATA = False

# %%
# scglue params
rel_gene_weight = args.rel_gene_weight
latent_dim=args.n_latent
h_depth=args.n_layers
h_dim=args.n_hidden

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
print("PATH_SAVE=",path_save)

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
# print('adata')
# print(adata)

# %%
adata_2 = None
if not SINGLE_ADATA:
    adata_2=sc.read(args.path_adata_2)
    adata_2.obs[args.system_key] = adata_2.obs[args.system_key].astype("str")
# print('adata 2')
# print(adata_2)

# %%
given_gene_graph = None
if not SINGLE_ADATA:
    given_gene_graph = pd.read_csv(args.gene_graph, sep="\t")
#given_gene_graph

# %%
if TESTING:
    # Make data smaller if testing the script
    random_idx=np.random.permutation(adata.obs_names)[:5000]
    adata=adata[random_idx,:].copy()
    # Set some groups to nan for testing if this works
    adata.obs[args.group_key]=[np.nan]*10+list(adata.obs[args.group_key].iloc[10:])

    if not SINGLE_ADATA:
        adata=adata[:2500, :].copy()
        random_idx=np.random.permutation(adata_2.obs_names)[:2500]
        adata_2=adata_2[random_idx,:].copy()
        # Set some groups to nan for testing if this works
        adata_2.obs[args.group_key]=[np.nan]*10+list(adata_2.obs[args.group_key].iloc[10:])

# %%
if SINGLE_ADATA:
    total_mods = list(adata.obs[args.system_key].unique())
else:
    total_mods = list(sorted(
        list(adata.obs[args.system_key].unique()) +
        list(adata_2.obs[args.system_key].unique())))

#total_mods

# %%
mods_adata = {}
if SINGLE_ADATA:
    for mod in total_mods:
        mods_adata[mod] = adata[adata.obs[args.system_key] == mod]
        mods_adata[mod].var['original_index'] = mods_adata[mod].var.index
        mods_adata[mod].var.index = mods_adata[mod].var.index + f'-{mod}'
        print(f"mod: {mod}\n", mods_adata[mod])
else:
    for adata_current in [adata, adata_2]:
        mod = adata_current.obs[args.system_key].unique()[0]
        mods_adata[mod] = adata_current
        mods_adata[mod].var['original_index'] = mods_adata[mod].var.index
        mods_adata[mod].var.index = mods_adata[mod].var.index + f'-{mod}'
        print(f"mod: {mod}\n", mods_adata[mod])

# %% [markdown]
# ### Training

# %%
print('Train')

# %%
for adata_spc in mods_adata.values():
    if args.pca_key == "":
        # X contains normalized+log data
        # adata_spc.X = adata_spc.layers['counts'].copy()
        # sc.pp.normalize_total(adata_spc)
        # sc.pp.log1p(adata_spc)
        sc.pp.scale(adata_spc)
        sc.tl.pca(adata_spc)
        pca_key = "X_pca"
    else:
        pca_key = args.pca_key
    
    scglue.models.configure_dataset(
        adata_spc, "NB", use_highly_variable=False, use_batch=args.batch_key,
        use_layer="counts", use_rep=pca_key,
    )

# %%
my_guidance = nx.DiGraph()
for mod in total_mods:
    adata_mod = mods_adata[mod]
    print(f"connecting {mod} to itself")
    for g in adata_mod.var.index:
        my_guidance.add_node(g)
        my_guidance.add_edge(g, g, **{'weight': 1.0, 'sign': 1, 'type': 'loop', 'id': '0'})

for i, mod in enumerate(total_mods):
    adata_mod = mods_adata[mod]
    for j, omod in enumerate(total_mods):
        if i == j:
            continue
        adata_omod = mods_adata[omod]
        print(f"connecting {mod} to {omod}")
        if SINGLE_ADATA:
            for g in adata_mod.var.index:
                my_guidance.add_edge(g, g.split("-")[0]+f"-{omod}", **{'weight': rel_gene_weight, 'sign': 1, 'type': 'normal', 'id': '0'})
        else:
            # Attention: we access columns of graph df by numbers (system_key in adatas should be given in the correct order)
            edge_mod = given_gene_graph[given_gene_graph.columns[i]] + f"-{mod}"
            edge_omod = given_gene_graph[given_gene_graph.columns[j]] + f"-{omod}"

            assert np.all(edge_mod.isin(adata_mod.var.index))
            assert np.all(edge_omod.isin(adata_omod.var.index))

            for g1, g2 in zip(list(edge_mod), list(edge_omod)):
                my_guidance.add_edge(g1, g2, **{'weight': rel_gene_weight, 'sign': 1, 'type': 'normal', 'id': '0'})
print("Done")

# %%
filename = path_save + "scglue_model"

# %%
max_epochs = args.max_epochs
if max_epochs == -1:
    max_epochs = scglue.utils.AUTO

glue = scglue.models.fit_SCGLUE(
    mods_adata, my_guidance,
    init_kws=dict(latent_dim=latent_dim, h_depth=h_depth, h_dim=h_dim, random_seed=args.seed),
    fit_kws={
        # Setting dir causes error as multiple runs want to be saved in the same dir
        #"directory": "glue", 
        # The below would likely work
        "directory": path_save+'glue_logs',
        "max_epochs": max_epochs},
    compile_kws=dict(lam_data=lam_data, lam_kl=lam_kl, lam_graph=lam_graph, lam_align=lam_align), 
    balance_kws=None,
)

#glue.save(filename)

# %%
# glue = scglue.models.load_model(filename)

# %%
if False:
    try:
        mods_adata_temp={}
        for k, a in mods_adata.items():
            a=a.copy()
            a.X=a.layers['counts']
            mods_adata_temp[k]=a
        del a
        dx = scglue.models.integration_consistency(
            glue, mods_adata_temp, my_guidance
        )
        del mods_adata_temp
        #_ = sns.lineplot(x="n_meta", y="consistency", data=dx).axhline(y=0.05, c="darkred", ls="--")
        print('Consistency scores with min:', dx['consistency'].min())
        print(dx)
        with open(path_save + "integration_consistency.pkl", 'wb') as f:
            pkl.dump(dx, f)
    except Exception as e:
        print("Error:", e)

# %% [markdown]
# ### Eval

# %% [markdown]
# #### Losses

# %% [markdown]
# #### Embedding

# %%
print('Get embedding')

# %%
for mod, adata_mod in mods_adata.items():
    adata_mod.obsm["X_glue"] = glue.encode_data(mod, adata_mod)
combined = ad.concat(mods_adata.values())

# %%
# Compute embedding
if SINGLE_ADATA:
    all_obs_names = list(adata.obs_names)
else:
    all_obs_names = list(adata.obs_names) + list(adata_2.obs_names)

embed_full = sc.AnnData(combined[all_obs_names].obsm['X_glue'], obs=combined[all_obs_names,:].obs.copy())
cells_eval = all_obs_names if args.n_cells_eval==-1 else \
    np.random.RandomState(seed=0).permutation(all_obs_names)[:args.n_cells_eval]
print('N cells for eval:',cells_eval.shape[0])
embed = embed_full[cells_eval].copy()

# %%
# Make system categorical, also for metrics below
embed.obs[args.system_key]=embed.obs[args.system_key].astype(str)
embed_full.obs[args.system_key]=embed_full.obs[args.system_key].astype(str)

# %%
# Save embed
embed.write(path_save+'embed.h5ad')
embed_full.write(path_save+'embed_full.h5ad')

# 
# %%
print('Finished integration!')

# %%
