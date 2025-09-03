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
#     display_name: scmgca3
#     language: python
#     name: scmgca3
# ---

# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'

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
import pathlib
import string
import subprocess

from matplotlib.pyplot import rcParams
import matplotlib.pyplot as plt
import seaborn as sb

# %%
import tensorflow as tf
from scMGCA.preprocess import *
from scMGCA.utils import *
from scMGCA.scmgca import SCMGCAL
from scMGCA.losses import *
from scMGCA.graph_function import *
from spektral.layers import GraphConv

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
parser.add_argument('-nk', '--n_latent', required=False, type=int, default=15,
                    help='n_latent of module')
parser.add_argument('-me', '--max_epochs', required=False, type=int,default=50,
                    help='max_epochs for training')
parser.add_argument('-nce', '--n_cells_eval', required=False, type=int, default=-1,  
                    help='Max cells to be used for eval, if -1 use all cells. '+\
                   'For cell subsetting seed 0 is always used to be reproducible accros '+\
                   'runs with different seeds.')

parser.add_argument('--scmgca_w_a', required=False, type=float, default=0.3,  # 0.1, 0.3, 0.6, 0.9
                    help='Model regularizer')

parser.add_argument('-t', '--testing', required=False, type=intstr_to_bool,default='0',
                    help='Testing mode')
# %%
# Set args for manual testing
if hasattr(sys, 'ps1'):
    args= parser.parse_args(args=[
        '-pa','/om2/user/khrovati/data/cross_system_integration/retina_adult_organoid/combined_HVG.h5ad',
        '-ps','/home/moinfar/tmp/',
        '-sk','system',
        '-gk','cell_type',
        '-bk','sample_id',
        
        '-s','1',
                
        '-nce','1000',
        
        '-t','1'
    ])
    args, args_unknown = parser.parse_known_args("--conda_env scmgca3 --params_opt scmgca_w_a --path_adata /home/moinfar/data/human_retina_atlas/human_retina_atlas_sc_sn_hvg --fn_moransi /home/moinfar/data/human_retina_atlas/human_retina_atlas_sc_sn_hvg_moransiGenes.pkl --path_save /home/moinfar/io/csi/eval/retina_atlas_sc_sn/integration/ --system_key system --group_key cell_type --batch_key donor_id --pca_key X_pca_system --system_decoders 0 --max_epochs 3 --epochs_detail_plot 10 --n_cells_eval 100000 --scmgca_w_a 0.6 --seed 3".split(" "))
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
path_save=args.path_save+'scmgca'+\
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
    tf.random.set_seed(args.seed)

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
adata.X = adata.layers['counts']
adata

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
# Adapted from: 
# 1. https://github.com/Philyzh8/scMGCA/blob/c3a1e87c000fa1097f74a66385ef4ab0e553a13b/tutorial/demo_batch.ipynb
# 2. https://github.com/Philyzh8/scMGCA/blob/c3a1e87c000fa1097f74a66385ef4ab0e553a13b/tutorial/demo_scale.py

# %%
obs_original = adata.obs.copy()

# %%
print("Cell number:", adata.X.shape[0])
print("Gene number",adata.X.shape[1])
cluster_number = adata.obs[args.group_key].nunique()
print("Cluster number:", cluster_number)
count = normalize_batch(adata, batch_key = args.batch_key, n_high_var = adata.n_vars)
print(count.shape)

# %%
import logging
import warnings
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.CRITICAL)

# %%
# Build graph in a batched manner (otherwise will face memory issue)
batch_size = 2000
adj, adj_n = get_adj_batch(count, batch_size)


# Pre-training
n_sample=count.shape[0]
in_dim=count.shape[1]

model = SCMGCAL(n_sample, in_dim, batch_size, hidden_dim=128, latent_dim=args.n_latent, dec_dim=[128])
model.pre_train(count, adj, adj_n, epochs=args.max_epochs, W_a=float(args.scmgca_w_a))
latent_pre = model.embeddingBatch(count, adj_n)

# Intermediate processing
adata_latent = sc.AnnData(latent_pre)
sc.pp.neighbors(adata_latent, n_neighbors = 15, use_rep="X")
if adata.n_obs < 600_000:
    resolution = find_resolution(adata_latent, 8, 0)
    adata_latent = sc.tl.leiden(adata_latent, resolution = resolution, random_state = 0, copy = True)
else:
    sc.tl.leiden(adata_latent, resolution = 0.2)
Y_pred_init = adata_latent.obs['leiden']
labels = np.asarray(Y_pred_init, dtype=int)
centers = computeCentroids(latent_pre, labels)

# %%
# Clustering training
Cluster_predicted = model.train(count, adj, adj_n, epochs=args.max_epochs, W_a=float(args.scmgca_w_a), centers=centers)

# latent representation
latent = Cluster_predicted.latent
latent.shape


# %% [markdown]
# ### Eval

# %% [markdown]
# #### Embedding

# %%
print('Get embedding')

# %%
embed_full = latent
embed_full = sc.AnnData(embed_full, obs=obs_original)
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
