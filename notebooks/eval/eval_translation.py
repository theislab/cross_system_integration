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
# # Evaluate translation accross species

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
parser.add_argument('-st', '--system_translate', required=False, type=int,default='1',
                    help='System to translate')
parser.add_argument('-gk', '--group_key', required=True, type=str,
                    help='obs col with group info')
parser.add_argument('-gt', '--group_translate', required=True, type=str,
                    help='Group to translate')
parser.add_argument('-bk', '--batch_key', required=True, type=str,
                    help='obs col with batch info')
parser.add_argument('-ce', '--cells_eval', required=False, type=str,default=None,
                    help='obs col with info which cells to use for eval.'+\
                    'If unspecified uses all cells.')
parser.add_argument('-ge', '--genes_eval', required=False, type=str,default=None,
                    help='var col with info which genes to use for eval.'+\
                    'If unspecified uses all genes.')
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
        '-pa','/lustre/groups/ml01/workspace/karin.hrovatin/data/cross_species_prediction/pancreas_example/combined_tabula_orthologues.h5ad',
        '-ps','/lustre/groups/ml01/workspace/karin.hrovatin/data/cross_species_prediction/eval/pancreas_example_v0/translation/',
        '-sk','system',
        '-gk','cell_type',
        '-gt','type B pancreatic cell',
        '-bk','batch',
        '-ce','eval_cells',
        '-ge','eval_genes',
        '-me','2',
        '-edp','0',
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
    'ST'+str(args.system_translate)+\
    'GT'+str(args.group_translate)+\
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
# ## Translation

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
print('Train and eval')

# %%
train_filter=(adata.obs[args.group_key]!=args.group_translate).values |\
             (adata.obs[args.system_key]!=args.system_translate).values
print('N cells for training:',train_filter.sum())

# %%
# Setup model
adata_training = XXJointModel.setup_anndata(
    adata=adata[train_filter,:],
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
# #### Translation

# %%
# Cell used for translation in ref system and ground truth in query system
eval_cells_query='' if args.cells_eval is None else f'& {args.cells_eval}==True'
cells_ref=adata.obs.query(
    f'{args.group_key}=="{args.group_translate}" &'+ 
    f'{args.system_key}!={args.system_translate}'+eval_cells_query).index
cells_query=adata.obs.query(
    f'{args.group_key}=="{args.group_translate}" &'+ 
    f'{args.system_key}=={args.system_translate}'+eval_cells_query).index
print(f'N cells ref: {len(cells_ref)} and query: {len(cells_query)}')

# %%
# Translate
pred=[]
for translation_type, switch in {'ref':False,
                                 'query':True}.items():
    pred.append(
        sc.AnnData(
            model.translate(
                adata= adata_training[cells_ref,:],
                switch_system = switch,
                covariates = None),
            obs=pd.DataFrame({
                #'translation':[translation_type]*cells_ref.shape[0],
                'cells_ref':cells_ref,
                'meta':['pred_'+str(translation_type)]*cells_ref.shape[0]
            }),
            var=adata_training.var
        )
    )
# Concat to single adata
pred=sc.concat(pred)

# %%
if args.genes_eval is not None:
    pred=pred[:,adata.var[args.genes_eval]]

# %% [markdown]
# #### Correlation

# %%
# Mean expression per prediction group
x=pred.to_df()
x['meta']=pred.obs['meta']
x=x.groupby('meta').mean()

# Add unpredicted expression
x.loc['ref',:]=adata[cells_ref,:].to_df().mean()
x.loc['query',:]=adata[cells_query,:].to_df().mean()

# %%
# Correlation between cell groups
cor=pd.DataFrame(np.corrcoef(x),index=x.index,columns=x.index)

# %%
# Plot correlation
sb.clustermap(cor,figsize=(3,3))
plt.tight_layout()
plt.savefig(path_save+'translation_correlation.png',dpi=300,bbox_inches='tight')

# %%
# Save correlation
cor.to_csv(path_save+'translation_correlation.tsv',sep='\t')

# %%
print('Finished!')

# %%
