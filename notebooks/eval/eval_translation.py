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
parser.add_argument('-n', '--name', required=False, type=str, default='',
                    help='name')
parser.add_argument('-po', '--params_opt', required=False, type=str, default='',
                    help='name of optimized params/test purpose')
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
parser.add_argument('-ts', '--train_size', required=False, type=float,default=0.9,
                    help='train_size for training')
parser.add_argument('-ma', '--mixup_alpha', required=False, 
                    type=str_to_float_zeronone,default='0',
                    help='mixup_alpha for model. If unspecified or 0 dont use mixup_alpha, '+
                   'else use float for mixup_alpha')
parser.add_argument('-sd', '--system_decoders', required=False, 
                    type=intstr_to_bool,default='0',
                    help='system_decodersfor model. Converts 0/1 to bool')
parser.add_argument('-p', '--prior', required=False, type=str, default='standard_normal',
                    help='VAE prior')
parser.add_argument('-npc', '--n_prior_components', required=False, type=int, default=100,
                    help='n_prior_components used for vamp prior')
parser.add_argument('-nl', '--n_layers', required=False, type=int, default=2,
                    help='n_layers of module')
parser.add_argument('-nh', '--n_hidden', required=False, type=int, default=256,
                    help='n_hidden of module')
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

parser.add_argument('-o', '--optimizer', required=False, type=str,default="Adam",
                    help='optimizer for training plan')
parser.add_argument('-lr', '--lr', required=False, type=float,default=0.001,
                    help='learning rate for training plan')
parser.add_argument('-rlrp', '--reduce_lr_on_plateau', required=False, 
                    type=intstr_to_bool, default="0",
                    help='reduce_lr_on_plateau for training plan')
parser.add_argument('-lrsm', '--lr_scheduler_metric', required=False, 
                    type=str, default='train_loss',
                    help='lr_scheduler_metric for training plan reduce_lr_on_plateau')
parser.add_argument('-lrp', '--lr_patience', required=False, type=int,default=5,
                    help='lr_patience for training plan reduce_lr_on_plateau')
parser.add_argument('-lrf', '--lr_factor', required=False, type=float,default=0.1,
                    help='lr_factor for training plan reduce_lr_on_plateau')
parser.add_argument('-lrm', '--lr_min', required=False, type=float,default=1e-7,
                    help='lr_min for training plan reduce_lr_on_plateau')
parser.add_argument('-lrtm', '--lr_threshold_mode', required=False, type=str,default='rel',
                    help='lr_threshold_mode for training plan reduce_lr_on_plateau')
parser.add_argument('-lrt', '--lr_threshold', required=False, type=float,default=-0.05,
                    help='lr_threshold for training plan reduce_lr_on_plateau')

parser.add_argument('-t', '--testing', required=False, type=intstr_to_bool,default='0',
                    help='Testing mode')
# %%
if False:
    args= parser.parse_args(args=[
        #'-pa','/lustre/groups/ml01/workspace/karin.hrovatin/data/cross_species_prediction/pancreas_example/combined_tabula_orthologues.h5ad',
        '-pa','/lustre/groups/ml01/workspace/karin.hrovatin/data/cross_species_prediction/pancreas_example/combined_orthologues.h5ad',
        '-ps','/lustre/groups/ml01/workspace/karin.hrovatin/data/cross_species_prediction/eval/test/translation/',
        '-sk','system',
        #'-gk','cell_type',
        '-gk','cell_type_final',
        #'-gt','type B pancreatic cell',
        '-gt','beta',
        #'-bk','batch',
        '-bk','study_sample',
        #'-ce','eval_cells',
        #'-ge','eval_genes',
        '-me','2',
        '-edp','0',
        
        # Lr testing
        '-o','AdamW',
        '-lr','0.01',
        '-rlrp','1',
        '-lrt','-0.05',
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
    'P'+str(''.join([i[0] for  i in args.prior.split('_')]))+\
    'NPC'+str(args.n_prior_components)+\
    'NL'+str(args.n_layers)+\
    'NH'+str(args.n_hidden)+\
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
# Setup adata
adata_training = XXJointModel.setup_anndata(
    adata=adata[train_filter,:],
    system_key=args.system_key,
    group_key=args.group_key,
    categorical_covariate_keys=[args.batch_key],
)

# %%
# Model
model = XXJointModel(
    adata=adata_training,
    mixup_alpha=args.mixup_alpha,
    system_decoders=args.system_decoders,
    prior=args.prior, 
    n_prior_components=args.n_prior_components,
    n_layers=args.n_layers,
    n_hidden=args.n_hidden,
)
# Train
model.train(max_epochs=args.max_epochs if not TESTING else 2,
            check_val_every_n_epoch=1,
            train_size=args.train_size,
            callbacks=[pl.callbacks.LearningRateMonitor(logging_interval='epoch')],
            plan_kwargs={
                'optimizer':args.optimizer,
                'lr':args.lr,
                'reduce_lr_on_plateau':args.reduce_lr_on_plateau,
                'lr_scheduler_metric':args.lr_scheduler_metric,
                'lr_patience':args.lr_patience,
                'lr_factor':args.lr_factor,
                'lr_min':args.lr_min,
                'lr_threshold_mode':args.lr_threshold_mode,
                'lr_threshold':args.lr_threshold,
                'loss_weights':{
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
        if '_step' not in k and '_epoch' not in k and 'validation' not in k]
fig,axs=plt.subplots(2,len(losses),figsize=(len(losses)*3,4))
for ax_i,l_train in enumerate(losses):
    l_val=l_train.replace('_train','_validation')
    l_name=l_train.replace('_train','')
    for l,c in [(l_train,'tab:blue'),(l_val,'tab:orange')]:
        axs[0,ax_i].plot(
            model.trainer.logger.history[l].index if 'lr' not in l 
                else list(range(model.trainer.logger.history[l].index.shape[0])),
            model.trainer.logger.history[l][l],c=c)
        axs[0,ax_i].set_title(l_name)
        axs[1,ax_i].plot(
            model.trainer.logger.history[l].index[args.epochs_detail_plot:]
            if 'lr' not in l else list(range(args.epochs_detail_plot,
                                model.trainer.logger.history[l].index.shape[0])),
            model.trainer.logger.history[l][l][args.epochs_detail_plot:],c=c)
        if 'lr' in l:
            axs[0,ax_i].set_yscale('log')
            axs[1,ax_i].set_yscale('log')
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
train_set=set(adata_training.obs_names[model.train_indices])
val_set=set(adata_training.obs_names[model.validation_indices])
cells_ref_train=[c for c in cells_ref if c in train_set]
cells_ref_val=[c for c in cells_ref if c in val_set]
cells_query=adata.obs.query(
    f'{args.group_key}=="{args.group_translate}" &'+ 
    f'{args.system_key}=={args.system_translate}'+eval_cells_query).index
print(f'N cells ref: {len(cells_ref)} and query: {len(cells_query)}')
print(f'N cells ref train: {len(cells_ref_train)} and val: {len(cells_ref_val)}')

# %% [markdown]
# ##### Correlation accross systems (prediction, translation)

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

# %% [markdown]
# ##### Correlation between samples in reference train and val data

# %%
# Translate ref cells with matching covariates
cells=cells_ref_train+cells_ref_val
pred=sc.AnnData(
        model.translate(
            adata= adata_training[cells,:],
            switch_system = switch,
            covariates = adata[cells,:].obs),
        obs=adata_training[cells,:].obs,
        var=adata_training.var
    )

# %%
# Subset genes if needed for eval
if args.genes_eval is not None:
    pred=pred[:,adata.var[args.genes_eval]]


# %%
# Correlation between predicted and true sampels, per sample
def corr_samples(x,y):
    corr=[]
    for i in range(x.shape[0]):
        corr.append(np.corrcoef(x[i,:],y[i,:])[0,1])
    return corr
corrs=corr_samples(pred.X,np.array(adata[pred.obs_names,pred.var_names].X.todense()))
corrs=pd.DataFrame({'corr':corrs,
                    'set':['train']*len(cells_ref_train)+['val']*len(cells_ref_val)},
                  index=cells)

# %%
# Save correlation
corrs.to_csv(path_save+'prediction_correlation_trainval.tsv',sep='\t')

# %% [markdown]
# # End

# %%
print('Finished!')
