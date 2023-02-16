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

import scvi
import pytorch_lightning as pl

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
parser.add_argument('-me', '--max_epochs', required=False, type=int,default=50,
                    help='max_epochs for training')
parser.add_argument('-edp', '--epochs_detail_plot', required=False, type=int, default=20,
                    help='Loss subplot from this epoch on')

parser.add_argument('-nce', '--n_cells_eval', required=False, type=int, default=-1,  
                    help='Max cells to be used for eval, if -1 use all cells. '+\
                   'For cell subsetting seed 0 is always used to be reproducible accros '+\
                   'runs with different seeds.')

parser.add_argument('-t', '--testing', required=False, type=intstr_to_bool,default='0',
                    help='Testing mode')
# %%
if False:
    args= parser.parse_args(args=[
        '-pa','/lustre/groups/ml01/workspace/karin.hrovatin/data/cross_species_prediction/pancreas_example/combined_orthologues.h5ad',
        '-ps','/lustre/groups/ml01/workspace/karin.hrovatin/data/cross_species_prediction/eval/test/integration/',
        '-sk','system',
        '-gk','cell_type_final',
        '-bk','study_sample',
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
# Make folder for saving
path_save=args.path_save+'scvi'+\
    '_'+''.join(np.random.permutation(list(string.ascii_letters)+list(string.digits))[:8])+\
    ('-TEST' if TESTING else '')+\
    os.sep

os.mkdir(path_save)
print(path_save)

# %%
# Set seed for eval
# Set only here below as need randomness for generation of out directory name (above)
if args.seed is not None:
    config.seed=args.seed

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
    # Set some groups to nan for testing if this works
    adata.obs[args.group_key]=[np.nan]*10+list(adata.obs[args.group_key].iloc[10:])

# %% [markdown]
# ### Training

# %%
print('Train')

# %%
# Setup adata
adata_training = adata
scvi.model.SCVI.setup_anndata(
    adata_training, 
    layer="counts", 
    batch_key=args.system_key,
    categorical_covariate_keys=[args.batch_key])

# %%
model = scvi.model.SCVI(adata_training, 
                        n_layers=2, n_hidden=256, n_latent=15, 
                        gene_likelihood="nb")

max_epochs=args.max_epochs if not TESTING else 3
model.train(
    max_epochs = max_epochs,
    log_every_n_steps=1,
    check_val_every_n_epoch=1,
    callbacks=[pl.callbacks.LearningRateMonitor(logging_interval='epoch')] 
)

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
steps_detail_plot = args.epochs_detail_plot*int(
    model.trainer.logger.history['validation_loss'].shape[0]/max_epochs)
detail_plot= steps_detail_plot
losses=[k for k in model.trainer.logger.history.keys() 
        if #'_step' not in k and '_epoch' not in k and 
        ('validation' not in k or 'eval' in k)]
fig,axs=plt.subplots(2,len(losses),figsize=(len(losses)*3,4))
for ax_i,l_train in enumerate(losses):
    if 'lr-' not in l_train and '_eval' not in l_train and 'train_' not in l_train:
        l_val=l_train.replace('_train','_validation')
        l_name=l_train.replace('_train','')
        # Change idx of epochs to start with 1 so that below adjustment when 
        # train on step which only works for wal leads to appropriate multiplication
        l_val_values=model.trainer.logger.history[l_val].copy()
        l_val_values.index=l_val_values.index+1
        l_train_values=model.trainer.logger.history[l_train].copy()
        l_train_values.index=l_train_values.index+1
        # This happens if log on step as currently tyhis works only for val loss
        if l_train_values.shape[0]<l_val_values.shape[0]:
            l_train_values.index=\
                l_train_values.index*int(l_val_values.shape[0]/l_train_values.shape[0])
        for l_values,c,alpha,dp in [
            # train loss logged on epoch in either case now
            (l_train_values,'tab:blue',1,args.epochs_detail_plot),
            (l_val_values,'tab:orange',0.5, detail_plot)]:
            axs[0,ax_i].plot( l_values.index,l_values.values.ravel(),c=c,alpha=alpha)
            axs[0,ax_i].set_title(l_name)
            axs[1,ax_i].plot(l_values.index[dp:],
                             l_values.values.ravel()[dp:],c=c,alpha=alpha)
    else:
        l_values=model.trainer.logger.history[l_train].copy()
        l_values.index=l_values.index+1
        axs[0,ax_i].plot(l_values.index,l_values.values.ravel())
        axs[0,ax_i].set_title(l_train.replace('_validation_eval',''))
        # Lr has index of steps but logged number of epochs
        dp= args.epochs_detail_plot if 'lr-' in l_train else detail_plot
        axs[1,ax_i].plot(l_values.index[dp:],l_values.values.ravel()[dp:])
        if 'lr' in l_train:
            axs[0,ax_i].set_yscale('log')
            axs[1,ax_i].set_yscale('log')
fig.tight_layout()
plt.savefig(path_save+'losses.png',dpi=300,bbox_inches='tight')

# %% [markdown]
# #### Embedding

# %%
print('Plot embedding')

# %%
# Compute embedding
cells_eval=adata_training.obs_names if args.n_cells_eval==-1 else \
    np.random.RandomState(seed=0).permutation(adata_training.obs_names)[:args.n_cells_eval]
print('N cells for eval:',cells_eval.shape[0])
embed = model.get_latent_representation(
    adata=adata_training[cells_eval,:],
    indices=None,
    batch_size=None, )
embed=sc.AnnData(embed,obs=adata_training[cells_eval,:].obs)

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
# Which cells have group info - if nan dont use them for metrics computation
if embed.obs[args.group_key].isna().any():
    embed_group=embed[~embed.obs[args.group_key].isna(),:].copy()
    sc.pp.neighbors(embed_group, use_rep='X', n_neighbors=90)
else:
    embed_group=embed

# System and group lisi
metrics['ilisi_system']=sm.ilisi_knn(X=embed_group.obsp['distances'],
                              batches=embed_group.obs[args.system_key], scale=True)
metrics['clisi']=sm.clisi_knn(X=embed_group.obsp['distances'],
                              labels=embed_group.obs[args.group_key],scale=True)
# System and group asw
metrics['asw_system']=sm.silhouette_batch(
    X=embed_group.X, labels=embed_group.obs[args.group_key], 
    batch=embed_group.obs[args.system_key], rescale = True) 
metrics['asw_group']= sm.silhouette_label(
    X=embed_group.X, labels=embed_group.obs[args.group_key], rescale = True)

# %%
# Compute batch lisi metrics per system as else it would be confounded by system
# Same for asw batch
for system in sorted(embed_group.obs[args.system_key].unique()):
    embed_sub=embed_group[embed_group.obs[args.system_key]==system,:].copy()
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

# %% [markdown]
# # End

# %%
print('Finished!')

# %%
