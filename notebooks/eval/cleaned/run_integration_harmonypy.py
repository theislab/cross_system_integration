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
import sys
import scanpy as sc
import scanpy.external as sce
import pickle as pkl
import pandas as pd
import numpy as np
import random
seed=np.random.randint(0,1000000)
import argparse
import os
import pathlib
import string
import subprocess

from matplotlib.pyplot import rcParams
import matplotlib.pyplot as plt
import seaborn as sb

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
parser.add_argument('-nce', '--n_cells_eval', required=False, type=int, default=-1,  
                    help='Max cells to be used for eval, if -1 use all cells. '+\
                   'For cell subsetting seed 0 is always used to be reproducible accros '+\
                   'runs with different seeds.')
parser.add_argument('-ht', '--harmony_theta', required=False, type=float, default=1.0,  
                    help='Controls the strength of batch effect correction in Harmony (default is 1.0)')

parser.add_argument('-t', '--testing', required=False, type=intstr_to_bool,default='0',
                    help='Testing mode')
# %%
# Set args for manual testing
if hasattr(sys, 'ps1'):
    args= parser.parse_args(args=[
        '-pa','/om2/user/khrovati/data/cross_system_integration/adipose_sc_sn_updated/adiposeHsSAT_sc_sn.h5ad',
        '-ps','/home/moinfar/tmp/',
        '-sk','system',
        '-gk','cell_type',
        '-bk','donor_id',
        
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
# Make folder for saving
path_save=args.path_save+'harmonypy'+\
    '_'+''.join(np.random.permutation(list(string.ascii_letters)+list(string.digits))[:8])+\
    ('-TEST' if TESTING else '')+\
    os.sep

pathlib.Path(path_save).mkdir(parents=True, exist_ok=False)
print("PATH_SAVE=",path_save)

# %%
# Set seed for eval
# Set only here below as need randomness for generation of out directory name (above)
if args.seed is not None:
    np.random.seed(args.seed)
    random.seed(args.seed)

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

# %%
if TESTING:
    # Make data smaller if testing the script
    random_idx=np.random.permutation(adata.obs_names)[:5000]
    adata=adata[random_idx,:].copy()
    print(adata.shape)

# %% [markdown]
# ### Training

# %%
print('Train')

# %%
# PCA calculation
sc.pp.pca(adata)

# %%
sce.pp.harmony_integrate(adata, args.batch_key)

# %%
adata

# %% [markdown]
# ### Eval

# %% [markdown]
# #### Embedding

# %%
print('Get embedding')

# %%
embed_full = adata.obsm['X_pca_harmony']
embed_full=sc.AnnData(embed_full,obs=adata.obs)
# Make system categorical for eval as below
embed_full.obs[args.system_key]=embed_full.obs[args.system_key].astype(str)
# Save full embed
embed_full.write(path_save+'embed_full.h5ad')

# %%
# Compute embedding
cells_eval=adata.obs_names if args.n_cells_eval==-1 else \
    np.random.RandomState(seed=0).permutation(adata.obs_names)[:args.n_cells_eval]
print('N cells for eval:',cells_eval.shape[0])
embed = embed_full[cells_eval,:].copy()
# Make system categorical for metrics and plotting
embed.obs[args.system_key]=embed.obs[args.system_key].astype(str)
# Save embed
embed.write(path_save+'embed.h5ad')

# %%
del adata
del embed
del embed_full

# %% [markdown]
# # End

# %%
print('Finished integration!')

# %%
