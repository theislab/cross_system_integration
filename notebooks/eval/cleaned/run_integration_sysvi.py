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
#     display_name: sysvi
#     language: python
#     name: sysvi
# ---

# %% [markdown]
# # Evaluate integration accross species

# %%
import scanpy as sc
import pickle as pkl
import pandas as pd
import numpy as np
# Make random number for seed before scvi import sets seed to 0
seed=np.random.randint(0,1000000)
import argparse
import os
import sys
import pathlib
import string
import subprocess

from matplotlib.pyplot import rcParams
import matplotlib.pyplot as plt
import seaborn as sb

import scvi
from scvi.external import SysVI

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
def str_to_weight(x):
    # Format: wMIN_MAX_START_END (starts with w and separated by _ )
    # Quick seml fix to pass str and not int/list - add w at start and change sep
    x=[float(i) for i in x.replace('w','').split('_')]
    if len(x)==1:
        x=x[0]
    else:
        x={'weight_start':x[0], 'weight_end':x[1],
           'point_start':x[2], 'point_end':x[3], 
           'update_on':'step'}
    return x

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
parser.add_argument('-gk', '--group_key', required=True, type=str,
                    help='obs col with group info - used only for eval, not integration')
parser.add_argument('-bk', '--batch_key', required=True, type=str,
                    help='obs col with batch info')
parser.add_argument('-ts', '--train_size', required=False, type=float,default=0.9,
                    help='train_size for training')
parser.add_argument('-p', '--prior', required=False, type=str, default='standard_normal',
                    help='VAE prior')
parser.add_argument('-npc', '--n_prior_components', required=False, type=int, default=100,
                    help='n_prior_components used for vamp prior.'+ 
                    'if -1 use as many prior components as there are cell groups'+
                    '(ignoring nan groups)')
# N prior components system is int as system itself is 0/1
parser.add_argument('-pcs', '--prior_components_system', required=False, type=int, default=None,
                    help='system to sample prior components from.'+
                    'If -1 samples balanced from both systems'+
                    'If unsepcified samples randomly'+
                    'Either this of prior_components_group must be None')
parser.add_argument('-pcg', '--prior_components_group', required=False, type=str, default=None,
                    help='group to sample prior components from.'+
                   'If BALANCED sample balanced across groups (ignores nan groups)'+
                   'If unsepcified samples randomly'+
                   'Either this of prior_components_system must be None')
parser.add_argument('-tp', '--trainable_priors', required=False, 
                    type=intstr_to_bool,default='1',
                    help='trainable_priors for module. Converts 0/1 to bool')
parser.add_argument('-nl', '--n_layers', required=False, type=int, default=2,
                    help='n_layers of module')
parser.add_argument('-nh', '--n_hidden', required=False, type=int, default=256,
                    help='n_hidden of module')
parser.add_argument('-me', '--max_epochs', required=False, type=int,default=50,
                    help='max_epochs for training')
parser.add_argument('-edp', '--epochs_detail_plot', required=False, type=int, default=20,
                    help='Loss subplot from this epoch on')

parser.add_argument('-kw', '--kl_weight', required=False, 
                    type=str_to_weight,default=1,
                    help='kl_weight for training')
parser.add_argument('-rw', '--reconstruction_weight', required=False, 
                    type=str_to_weight,default=1,
                    help='reconstruction_weight for training')
parser.add_argument('-zdcw', '--z_distance_cycle_weight', required=False, 
                    type=str_to_weight,default=0,
                    help='z_distance_cycle_weight for training')

parser.add_argument('-o', '--optimizer', required=False, type=str,default="Adam",
                    help='optimizer for training plan')
parser.add_argument('-lr', '--lr', required=False, type=float,default=0.001,
                    help='learning rate for training plan')

parser.add_argument('-swa', '--swa', required=False, 
                    type=intstr_to_bool, default="0", help='use SWA')
parser.add_argument('-swalr', '--swa_lr', required=False, type=float, default=0.001, 
                    help='final SWA lr')
parser.add_argument('-swaes', '--swa_epoch_start', required=False, type=int, default=10, 
                    help='start SWA on epoch')
parser.add_argument('-swaae', '--swa_annealing_epochs', required=False, type=int, 
                    default=10,  help='SWA annealing epochs')

parser.add_argument('-nce', '--n_cells_eval', required=False, type=int, default=-1,  
                    help='Max cells to be used for eval, if -1 use all cells. '+\
                   'For cell subsetting seed 0 is always used to be reproducible accros '+\
                   'runs with different seeds.')

parser.add_argument('-t', '--testing', required=False, type=intstr_to_bool,default='0',
                    help='Testing mode')
# %%
# Set args for manual testing
if "ipykernel" in sys.modules and "IPython" in sys.modules:
    args= parser.parse_args(args=[
        #'-pa','/lustre/groups/ml01/workspace/karin.hrovatin/data/cross_species_prediction/pancreas_example/combined_tabula_orthologues.h5ad',
        '-pa','/om2/user/khrovati/data/cross_species_prediction/pancreas_healthy/combined_orthologuesHVG2000.h5ad',
        #'-pa','/om2/user/khrovati/data/cross_system_integration/pancreas_conditions_MIA_HPAP2/combined_orthologuesHVG.h5ad',
        #'-fmi','/om2/user/khrovati/data/cross_system_integration/eval/test/integration/example/moransiGenes_mock.pkl',
        '-ps','/home/moinfar/io/csi/eval/test',
        '-sk','system',
        '-gk','cell_type',
        #'-gk','cell_type_eval',
        '-bk','sample',
        #'-bk','batch',
        '-me','2',
        '-edp','0',
        '-p','vamp',
        
        '-npc','2',
        '-pcs','-1',
        
        '-epe','1',
        '-tp','0',
        
        '-kw','w0_1_1_2',
        '-rw','1',
        
        '-s','1',
        
        # Lr testing
        '-o','AdamW',
        '-lr','0.001',
        '-rlrp','1',
        '-lrt','-0.05',
        # SWA testing
        '-swa','1',
        
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
if args.prior_components_group is not None and args.prior_components_system is not None:
    raise ValueError('Either prior_components_group or prior_components_system must be None')


# %%
# Make folder for saving
def weight_to_str(x):
    if isinstance(x,dict):
        x='-'.join([str(x) for x in x.values() if not isinstance(x,str)])
    else:
        x=str(x)
    return x

path_save=args.path_save+\
    'KLW'+weight_to_str(args.kl_weight)+\
    'RW'+weight_to_str(args.reconstruction_weight)+\
    'ZDCW'+weight_to_str(args.z_distance_cycle_weight)+\
    'P'+str(''.join([i[0] for  i in args.prior.split('_')]))+\
    'NPC'+str(args.n_prior_components)+\
    'NL'+str(args.n_layers)+\
    'NH'+str(args.n_hidden)+\
    '_'+''.join(np.random.permutation(list(string.ascii_letters)+list(string.digits))[:8])+\
    ('-TEST' if TESTING else '')+\
    os.sep

pathlib.Path(path_save).mkdir(parents=True, exist_ok=False)
print("PATH_SAVE=",path_save)

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
adata_training = adata.copy()
SysVI.setup_anndata(
    adata=adata_training,
    batch_key=args.system_key,
    categorical_covariate_keys=[args.batch_key],
)

# %%
# Define pseudoinputs
if args.n_prior_components==-1:
    n_prior_components=adata_training.obs[args.group_key].dropna().nunique()
else:
    n_prior_components=args.n_prior_components

if args.prior_components_group is not None:
    if args.prior_components_group!='BALANCED':
        prior_locations=[(args.prior_components_group,n_prior_components)]
    else:
        groups=list(np.random.permutation(adata_training.obs[args.group_key].dropna().unique()))
        multiple, remainder = divmod(n_prior_components, len(groups))
        prior_locations=[(k,v) for k,v in 
             pd.Series(groups * multiple + groups[:remainder]).value_counts().iteritems()]
        
    pdi=[]
    for (group,npc) in prior_locations:
        pdi.extend(np.random.permutation(np.argwhere((
                adata_training.obs[args.group_key]==group
                        ).ravel()).ravel())[:npc])
    print('Prior components group:')
    print(adata_training.obs.iloc[pdi,:][args.group_key].value_counts())
        
elif args.prior_components_system is not None:
    if args.prior_components_system==-1:
        npc_half=int(n_prior_components/2)
        npc_half2=n_prior_components-npc_half
        prior_locations=[(0,npc_half),(1,npc_half2)]
    else:
        prior_locations=[(args.prior_components_system,n_prior_components)]
    pdi=[]
    for (system,npc) in prior_locations:
        pdi.extend(np.random.permutation(np.argwhere((
                adata_training.obs[args.system_key]==system
                        ).ravel()).ravel())[:npc])
    print('Prior components system:')
    print(adata_training.obs.iloc[pdi,:][args.system_key].value_counts())
        
else:
    pdi=None
    
if pdi is not None and len(pdi) != n_prior_components:
    raise ValueError('Not sufficent number of prior components could be sampled')

if pdi is not None:
    pdi = np.asarray(pdi)

# %%
# Train
model = SysVI(
    adata=adata_training,
    prior=args.prior, 
    n_prior_components=n_prior_components,
    pseudoinputs_data_indices=pdi,
    trainable_priors=args.trainable_priors,
    n_layers=args.n_layers,
    n_hidden=args.n_hidden,
)
max_epochs=args.max_epochs if not TESTING else 3
model.train(max_epochs=max_epochs,
            log_every_n_steps=1,
            check_val_every_n_epoch=1,
            val_check_interval=1.0 if args.log_on_epoch else 1,
            train_size=args.train_size,            
            plan_kwargs={
                'optimizer':args.optimizer,
                'lr':args.lr,
                'kl_weight':args.kl_weight,
                'reconstruction_weight':args.reconstruction_weight,
                'z_distance_cycle_weight':args.z_distance_cycle_weight,
           })

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
detail_plot=args.epochs_detail_plot if args.log_on_epoch else steps_detail_plot
losses=[k for k in model.trainer.logger.history.keys() 
        if #'_step' not in k and '_epoch' not in k and 
        ('validation' not in k or 'eval' in k)]
fig,axs=plt.subplots(2,len(losses),figsize=(len(losses)*3,4))
for ax_i,l_train in enumerate(losses):
    if 'lr-' not in l_train and '_eval' not in l_train:
        l_val=l_train.replace('_train','_validation')
        l_name=l_train.replace('_train','')
        # Change idx of epochs to start with 1 so that below adjustment when 
        # train on step which only works for val leads to appropriate multiplication
        l_val_values=model.trainer.logger.history[l_val].copy()
        l_val_values.index=l_val_values.index+1
        l_train_values=model.trainer.logger.history[l_train].copy()
        l_train_values.index=l_train_values.index+1
        # This happens if log on step as currently this works only for val loss
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

# %%

# %% [markdown]
# #### Embedding

# %%
print('Get embedding')

# %%
# Compute and save whole embedding
if args.n_cells_eval!=-1:
    embed_full = model.get_latent_representation(
        adata=adata_training,
        indices=None,
        batch_size=None)
    embed_full=sc.AnnData(embed_full,obs=adata_training.obs)
    # Make system categorical for eval as below
    embed_full.obs[args.system_key]=embed_full.obs[args.system_key].astype(str)
    # Save full embed
    embed_full.write(path_save+'embed_full.h5ad')
    del embed_full

# %%
# Compute embedding for eval
cells_eval=adata_training.obs_names if args.n_cells_eval==-1 else \
    np.random.RandomState(seed=0).permutation(adata_training.obs_names)[:args.n_cells_eval]
print('N cells for eval:',cells_eval.shape[0])
embed = model.get_latent_representation(
        adata=adata_training[cells_eval,:],
        indices=None,
        batch_size=None)

embed=sc.AnnData(embed,obs=adata_training[cells_eval,:].obs)
# Make system categorical for metrics and plotting
embed.obs[args.system_key]=embed.obs[args.system_key].astype(str)
# Save embed
embed.write(path_save+'embed.h5ad')
del embed

# %%
del adata
del adata_training

# %% [markdown]
# # End

# %%
print('Finished integration!')

# %%
