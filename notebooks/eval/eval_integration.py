# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: csp
#     language: python
#     name: csp
# ---

# %% [markdown]
# # Evaluate integration accross species

# %% [markdown]
# # TODO
# - Save losses numbers
# - Save latent embedding incl UMAPs

# %%
import scanpy as sc
import pickle as pkl
import pandas as pd
import numpy as np
# Make random number for seed before scvi import sets seed to 0
seed=np.random.randint(0,1000000)
import argparse
import os
import string

from matplotlib.pyplot import rcParams
import matplotlib.pyplot as plt
import seaborn as sb

import scib_metrics as sm

from constraint_pancreas_example.model._xxjointmodel import XXJointModel

# Otherwise the seed remains constant
from scvi._settings import ScviConfig
config=ScviConfig()
config.seed=seed

# %%
parser = argparse.ArgumentParser()
def intstr_to_bool(x):
    return bool(int(x))
def str_to_float_zeronone(x):
    if x is None or x=="0":
        return None
    else:
        return float(x)
parser.add_argument('-n', '--name', required=False, type=str, default='',
                    help='name')
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
parser.add_argument('-ma', '--mixup_alpha', required=False, 
                    type=str_to_float_zeronone,default='0',
                    help='mixup_alpha for model. If unspecified or 0 dont use mixup_alpha, '+
                   'else use float for mixup_alpha')
parser.add_argument('-sd', '--system_decoders', required=False, 
                    type=intstr_to_bool,default='0',
                    help='system_decodersfor model. Converts 0/1 to bool')
parser.add_argument('-me', '--max_epochs', required=False, type=int,default=50,
                    help='max_epochs for training')
parser.add_argument('-edp', '--epochs_detail_plot', required=False, type=int, default=20,
                    help='Loss subplot from this epoch on')
parser.add_argument('-kw', '--kl_weight', required=False, type=float,default=1,
                    help='kl_weight for training')
parser.add_argument('-kcw', '--kl_cycle_weight', required=False, type=float,default=0,
                    help='kl_cycle_weight for training')
parser.add_argument('-rw', '--reconstruction_weight', required=False, type=float,default=1,
                    help='reconstruction_weight for training')
parser.add_argument('-rmw', '--reconstruction_mixup_weight', required=False, 
                    type=float,default=0,
                    help='kl_weight for training')
parser.add_argument('-rcw', '--reconstruction_cycle_weight', required=False, 
                    type=float,default=0,
                    help='reconstruction_cycle_weight for training')
parser.add_argument('-zdcw', '--z_distance_cycle_weight', required=False, type=float,default=0,
                    help='z_distance_cycle_weight for training')
parser.add_argument('-tcw', '--translation_corr_weight', required=False, type=float,default=0,
                    help='translation_corr_weight for training')
parser.add_argument('-zcw', '--z_contrastive_weight', required=False, type=float,default=0,
                    help='z_contrastive_weight for training')

parser.add_argument('-t', '--testing', required=False, type=intstr_to_bool,default='0',
                    help='Testing mode')
# %%
if False:
    args= parser.parse_args(args=[
        '-pa','/lustre/groups/ml01/workspace/karin.hrovatin/data/cross_species_prediction/pancreas_example/combined_orthologues.h5ad',
        '-ps','/lustre/groups/ml01/workspace/karin.hrovatin/data/cross_species_prediction/eval/pancreas_example_v0/integration/',
        '-sk','system',
        '-gk','cell_type_final',
        '-bk','study_sample',
        '-me','2',
        '-t','1'
    ])
# Read command line args
else:
    args = parser.parse_args()
    
print(args)

TESTING=args.testing

# %%
# Make folder for saving
path_save=args.path_save+\
    'MA'+str(args.mixup_alpha)+\
    'SD'+str(args.system_decoders)+\
    'KLW'+str(args.kl_weight)+\
    'KLCW'+str(args.kl_cycle_weight)+\
    'RW'+str(args.reconstruction_weight)+\
    'RMW'+str(args.reconstruction_mixup_weight)+\
    'RCW'+str(args.reconstruction_cycle_weight)+\
    'ZDCW'+str(args.z_distance_cycle_weight)+\
    'TCW'+str(args.translation_corr_weight)+\
    'ZCW'+str(args.z_contrastive_weight)+\
    '_'+''.join(np.random.permutation(list(string.ascii_letters)+list(string.digits))[:8])+\
    ('-TEST' if TESTING else '')+\
    os.sep

os.mkdir(path_save)
print(path_save)

# %%
# Save args
pkl.dump(args,open(path_save+'args.pkl','wb'))

# %% [markdown]
# ## Integration

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
# Setup model
adata_training = XXJointModel.setup_anndata(
    adata=adata,
    system_key=args.system_key,
    group_key=args.group_key,
    categorical_covariate_keys=[args.batch_key],
)
model = XXJointModel(
    adata=adata_training,
    mixup_alpha=args.mixup_alpha,
    system_decoders=args.system_decoders
)

# %%
# Train
model.train(max_epochs=args.max_epochs if not TESTING else 2,
           plan_kwargs={'loss_weights':{
               'kl_weight':args.kl_weight,
               'kl_cycle_weight':args.kl_cycle_weight,
               'reconstruction_weight':args.reconstruction_weight,
               'reconstruction_mixup_weight':args.reconstruction_mixup_weight,
               'reconstruction_cycle_weight':args.reconstruction_cycle_weight,
               'z_distance_cycle_weight':args.z_distance_cycle_weight,
               'translation_corr_weight':args.translation_corr_weight,
               'z_contrastive_weight':args.z_contrastive_weight,
               
           }})

# %% [markdown]
# ### Eval

# %% [markdown]
# #### Losses

# %%
print('Plot losses')

# %%
# Save losses
pkl.dump(model.trainer.logger.history,open(path_save+'losses.pkl','wb'))

# %%
# Plot all loses
losses=[k for k in model.trainer.logger.history.keys() 
        if '_step' not in k and '_epoch' not in k]
fig,axs=plt.subplots(2,len(losses),figsize=(len(losses)*3,4))
for ax_i,l in enumerate(losses):
    axs[0,ax_i].plot(
        model.trainer.logger.history[l].index,
        model.trainer.logger.history[l][l])
    axs[0,ax_i].set_title(l)
    axs[1,ax_i].plot(
        model.trainer.logger.history[l].index[args.epochs_detail_plot:],
        model.trainer.logger.history[l][l][args.epochs_detail_plot:])
fig.tight_layout()
plt.savefig(path_save+'losses.png',dpi=300,bbox_inches='tight')

# %% [markdown]
# #### Embedding

# %%
print('Plot embedding')

# %%
# Compute embedding
embed = model.embed(
        adata=adata_training,
        indices=None,
        batch_size=None,
        as_numpy=True)

embed=sc.AnnData(embed,obs=adata_training.obs)

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
print('Compute scib metrics')

# %%
# Dict for saving metrics
metrics={}

# %%
# System and group lisi
metrics['ilisi_system']=sm.ilisi_knn(X=embed.obsp['distances'],
                              batches=embed.obs[args.system_key], scale=True)
metrics['clisi']=sm.clisi_knn(X=embed.obsp['distances'],
                              labels=embed.obs[args.group_key],scale=True)
# System and group asw
metrics['asw_system']=sm.silhouette_batch(
    X=embed.X, labels=embed.obs[args.group_key], batch=embed.obs[args.system_key], 
    rescale = True) 
metrics['asw_group']= sm.silhouette_label(
    X=embed.X, labels=embed.obs[args.group_key], rescale = True)

# %%
# Compute batch lisi metrics per system as else it would be confounded by system
# Same for asw batch
for system in sorted(embed.obs[args.system_key].unique()):
    embed_sub=embed[embed.obs[args.system_key]==system,:].copy()
    sc.pp.neighbors(embed_sub, use_rep='X', n_neighbors=90)
    # Made system a str above
    metrics['ilisi_batch_system-'+system]=sm.ilisi_knn(
        X=embed_sub.obsp['distances'],
        batches=embed_sub.obs[args.batch_key], scale=True)
    metrics['asw_batch_system-'+system]= sm.silhouette_batch(
    X=embed_sub.X, labels=embed_sub.obs[args.group_key], batch=embed_sub.obs[args.batch_key],
        rescale = True)

# %%
print(metrics)

# %%
pkl.dump(metrics,open(path_save+'scib_metrics.pkl','wb'))

# %%
print('Finished!')

# %%
