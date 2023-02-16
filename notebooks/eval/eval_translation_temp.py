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
from scipy.stats import norm

from matplotlib.pyplot import rcParams
import matplotlib.pyplot as plt
import seaborn as sb

from constraint_pancreas_example.model._xxjointmodel import XXJointModel
import pytorch_lightning as pl

# Otherwise the seed remains constant
# Below it is reset based on params if specified
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
parser.add_argument('-loe', '--log_on_epoch', required=False, 
                    type=intstr_to_bool,default='1',
                    help='if true log on epoch and if false on step. Converts 0/1 to bool')
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
parser.add_argument('-pk', '--pretrain_key', required=False, type=str,default=None,
                    help='obs col with pretrain info. If unspecified dont use pre-training')
parser.add_argument('-pv', '--pretrain_value', required=False, type=str,default=None,
                    help='value in pretrain_key column to use for pretraining')
parser.add_argument('-ts', '--train_size', required=False, type=float,default=0.9,
                    help='train_size for training')
parser.add_argument('-ma', '--mixup_alpha', required=False, 
                    type=str_to_float_zeronone,default='0',
                    help='mixup_alpha for model. If unspecified or 0 dont use mixup_alpha, '+
                   'else use float for mixup_alpha')
parser.add_argument('-sd', '--system_decoders', required=False, 
                    type=intstr_to_bool,default='0',
                    help='system_decodersfor model. Converts 0/1 to bool')
parser.add_argument('-ovm', '--out_var_mode', required=False, type=str, default='feature',
                    help='out_var_mode')
parser.add_argument('-p', '--prior', required=False, type=str, default='standard_normal',
                    help='VAE prior')
parser.add_argument('-npc', '--n_prior_components', required=False, type=int, default=100,
                    help='n_prior_components used for vamp prior')
# N prior components system is int as system itself is 0/1
parser.add_argument('-pcs', '--prior_components_system', required=False, type=int, default=None,
                    help='system to sample prior components from for vamp prior.'+\
                   'If empty defaults to None and samples from both systems')
parser.add_argument('-zdm', '--z_dist_metric', required=False, type=str, default='MSE',
                    help='z_dist_metric for module')
parser.add_argument('-nl', '--n_layers', required=False, type=int, default=2,
                    help='n_layers of module')
parser.add_argument('-nh', '--n_hidden', required=False, type=int, default=256,
                    help='n_hidden of module')
parser.add_argument('-me', '--max_epochs', required=False, type=int,default=50,
                    help='max_epochs for training;'+\
                    'if using pretraining this is for post-training')
parser.add_argument('-mep', '--max_epochs_pretrain', required=False, type=int,default=20,
                    help='max_epochs_pretrain for pre-training (if used)')
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
                    type=str, default='loss_train',
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

parser.add_argument('-swa', '--swa', required=False, 
                    type=intstr_to_bool, default="0", help='use SWA')
parser.add_argument('-swalr', '--swa_lr', required=False, type=float, default=0.001, 
                    help='final SWA lr')
parser.add_argument('-swaes', '--swa_epoch_start', required=False, type=int, default=10, 
                    help='start SWA on epoch')
parser.add_argument('-swaae', '--swa_annealing_epochs', required=False, type=int, 
                    default=10,  help='SWA annealing epochs')

parser.add_argument('-t', '--testing', required=False, type=intstr_to_bool,default='0',
                    help='Testing mode')
# %%
if False:
    args= parser.parse_args(args=[
        '-pa','/lustre/groups/ml01/workspace/karin.hrovatin/data/cross_species_prediction/pancreas_example/combined_tabula_orthologues.h5ad',
        #'-pa','/lustre/groups/ml01/workspace/karin.hrovatin/data/cross_species_prediction/pancreas_example/combined_orthologues.h5ad',
        '-ps','/lustre/groups/ml01/workspace/karin.hrovatin/data/cross_species_prediction/eval/test/translation/',
        '-sk','system',
        '-gk','cell_type',
        #'-gk','cell_type_final',
        '-gt','type B pancreatic cell',
        #'-gt','beta',
        '-bk','batch',
        #'-bk','study_sample',
        '-ce','eval_cells',
        '-ge','eval_genes',
        '-me','2',
        '-edp','0',
        
        #'-pk','dataset',
        #'-pv','tabula',
        
        '-s','1',
        
        # Lr testing
        '-o','AdamW',
        '-lr','0.001',
        #'-rlrp','1',
        #'-lrt','-0.05',
        # SWA testing
        #'-swa','1',
        #'-swalr','0.0001',
        
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
path_save=args.path_save+\
    'ST'+str(args.system_translate)+\
    'GT'+str(args.group_translate)+\
    'MA'+str(args.mixup_alpha)+\
    'SD'+str(args.system_decoders)+\
    'OVM'+str(args.out_var_mode)+\
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

path_save=path_save.replace(' ','')

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
# ## Translation

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

# %%
# Cell used for translation in ref system and ground truth in query system
eval_cells_query='' if args.cells_eval is None else f'& {args.cells_eval}==True'
cells_ref=list(adata.obs.query(
    f'{args.group_key}=="{args.group_translate}" &'+ 
    f'{args.system_key}!={args.system_translate}'+eval_cells_query).index)
cells_query=list(adata.obs.query(
    f'{args.group_key}=="{args.group_translate}" &'+ 
    f'{args.system_key}=={args.system_translate}'+eval_cells_query).index)
print(f'N cells ref: {len(cells_ref)} and query: {len(cells_query)}')

# %%
adata.uns['eval_info']={
    'ref-pred_ref':{
        'cells_in':cells_ref, 
        'switch_system':False, 
        'cells_target':cells_ref, 
        'genes':set(adata.var_names if args.genes_eval is None 
                    else adata.var_names[adata.var[args.genes_eval]])
    },
    'query-pred_query':{
        'cells_in':cells_ref, 
        'switch_system':True, 
        'cells_target':cells_query, 
        'genes':set(adata.var_names if args.genes_eval is None 
                    else adata.var_names[adata.var[args.genes_eval]])
    },
}

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
adata_eval=XXJointModel.setup_anndata(
    adata=adata,
    system_key=args.system_key,
    group_key=args.group_key,
    categorical_covariate_keys=[args.batch_key],
)

# %%
# Model
if args.prior_components_system is not None:
    pdi=np.random.permutation(np.argwhere((
            adata_training.obs[args.system_key]==args.prior_components_system
                    ).ravel()).ravel())[:args.n_prior_components]
else:
    pdi=None
model = XXJointModel(
    adata=adata_training,
    out_var_mode=args.out_var_mode,
    mixup_alpha=args.mixup_alpha,
    system_decoders=args.system_decoders,
    prior=args.prior, 
    n_prior_components=args.n_prior_components,
    pseudoinputs_data_indices=pdi,
    z_dist_metric = args.z_dist_metric,
    n_layers=args.n_layers,
    n_hidden=args.n_hidden,
    adata_eval=adata_eval,
)
# Train
def train(model,indices, epochs):
    print('N training indices:',indices.shape[0])
    model.train(max_epochs=epochs,
                log_every_n_steps=1,
                check_val_every_n_epoch=1,
                val_check_interval=1.0 if args.log_on_epoch else 1, # Used irregardles of logging
                train_size=args.train_size,
                callbacks=[pl.callbacks.LearningRateMonitor(logging_interval='epoch')] +\
                    [pl.callbacks.StochasticWeightAveraging(
                        swa_lrs=args.swa_lr, 
                        swa_epoch_start=args.swa_epoch_start, 
                        annealing_epochs=args.swa_annealing_epochs)] 
                    if args.swa else [],
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
                    'log_on_epoch':args.log_on_epoch,
                    'log_on_step':not args.log_on_epoch,
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
# Epochs
max_epochs=args.max_epochs if not TESTING else 3
max_epochs_pretrain = 0 if args.pretrain_key is None else \
                      args.max_epochs_pretrain if not TESTING else 4
# Run training
if args.pretrain_key is not None:
    train(model=model, indices=np.argwhere(
          adata_training.obs[args.pretrain_key].values.ravel()==args.pretrain_value).ravel(),
          epochs=max_epochs_pretrain)
    train(model=model,indices=np.argwhere(
          adata_training.obs[args.pretrain_key].values.ravel()!=args.pretrain_value).ravel(),
          epochs=max_epochs)
else:
    train(model=model,indices=np.array(range(adata_training.n_obs)),
          epochs=max_epochs)

# %% [markdown]
# ### Eval

# %% [markdown]
# #### Losses

# %%
print('Plot losses')

# %%
# Save losses
pkl.dump(model.history,open(path_save+'losses.pkl','wb'))

# %%
# Plot all loses
steps_detail_plot = args.epochs_detail_plot*int(
    model.history['loss_validation'].shape[0]/max_epochs)
detail_plot=args.epochs_detail_plot if args.log_on_epoch else steps_detail_plot
losses=[k for k in model.history.keys() 
        if #'_step' not in k and '_epoch' not in k and 
        ('validation' not in k or 'eval' in k)]
fig,axs=plt.subplots(2,len(losses),figsize=(len(losses)*3,4))
for ax_i,l_train in enumerate(losses):
    if 'lr-' not in l_train and '_eval' not in l_train:
        l_val=l_train.replace('_train','_validation')
        l_name=l_train.replace('_train','')
        # Change idx of epochs to start with 1 so that below adjustment when 
        # train on step which only works for wal leads to appropriate multiplication
        l_val_values=model.history[l_val].copy()
        l_val_values.index=l_val_values.index+1
        l_train_values=model.history[l_train].copy()
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
        l_values=model.history[l_train].copy()
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
# #### Translation

# %% [markdown]
# ##### Correlation accross systems (prediction, translation)

# %% [markdown]
# TODO: Besides query-->pred query vs query and query-->pred ref vs ref also add eval with ref-->pred ref vs ref. Orn maybe not as key for translation decoder and even if E does not work well for ref it may work for query and then the decoder needs to translate well.

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
                #'translation':[translation_type]*len(cells_ref),
                'cells_ref':cells_ref,
                'meta':['pred_'+str(translation_type)]*len(cells_ref)
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
mean_pred=pred.to_df()
mean_pred['meta']=pred.obs['meta']
mean_pred=mean_pred.groupby('meta').mean()

# True expression summary
x_ref=adata[cells_ref,pred.var_names].to_df()
mean_ref=x_ref.mean()
mean_ref.name='ref'
std_ref=x_ref.std()
std_ref.name='ref'
del x_ref
x_query=adata[cells_query,pred.var_names].to_df()
mean_query=x_query.mean()
mean_query.name='query'
std_query=x_query.std()
std_query.name='query'
del x_query

# %%
# Correlation between cell groups
cor=pd.concat([mean_pred.T,mean_ref,mean_query],axis=1).T
cor=pd.DataFrame(np.corrcoef(cor),index=cor.index,columns=cor.index)
print(cor)

# %%
# Plot correlation
# Decided not to plot
if False:
    sb.clustermap(cor,figsize=(3,3))
    plt.tight_layout()
    plt.savefig(path_save+'translation_correlation.png',dpi=300,bbox_inches='tight')

# %%
# Save correlation
cor.to_csv(path_save+'translation_correlation.tsv',sep='\t')

# %%
# Gaussian ll of predicted mean vs distn on real cells
ll={}
for pred_group in sorted(pred.obs['meta'].unique()):
    for group,mean,std in [('ref',mean_ref,std_ref),('query',mean_query,std_query)]:
        # Nanmean as some genes have 0 var in data - problem in pdf 
        ll[group+'-'+pred_group]=np.nanmean(norm.logpdf(
            pred[pred.obs.meta==pred_group,:].to_df(),
            loc=pd.DataFrame(mean).T, scale=pd.DataFrame(std).T),axis=1).mean()
ll=pd.DataFrame([ll])
print(ll)

# %%
# Save correlation
ll.to_csv(path_save+'translation_normalLL.tsv',sep='\t', index=False)

# %% [markdown]
# ##### Correlation between samples in reference train and val data

# %%
train_set=set(adata_training.obs_names[model.train_indices])
val_set=set(adata_training.obs_names[model.validation_indices])
cells_ref_train=[c for c in cells_ref if c in train_set]
cells_ref_val=[c for c in cells_ref if c in val_set]
print(f'N cells ref train: {len(cells_ref_train)} and val: {len(cells_ref_val)}')

# %%
# Translate ref cells with matching covariates
cells=cells_ref_train+cells_ref_val
pred=sc.AnnData(
        model.translate(
            adata= adata_training[cells,:],
            switch_system = False,
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
