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
from pathlib import Path
import glob

from matplotlib.pyplot import rcParams
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import random
import scglue
import networkx as nx

# %%
SATURN_EMB_PATH = os.path.expanduser("~/Downloads/protein_embeddings_export/ESM2/")
SATURN_GIT_LOCATION = os.path.expanduser("~/projects/clones/SATURN/")
SATURN_CONDA_ENV = "cs_integration_saturn"

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
parser.add_argument('-ps', '--path_save', required=True, type=str,
                    help='directory path for saving, creates subdir within it')
parser.add_argument('-sk', '--system_key', required=True, type=str,
                    help='obs col with system info')
parser.add_argument('-sv', '--system_value', required=True, type=str,
                    help='real titles corresponding to system indicators')
parser.add_argument('-gk', '--group_key', required=True, type=str,
                    help='obs col with group info')
parser.add_argument('-bk', '--batch_key', required=True, type=str,
                    help='obs col with batch info')
parser.add_argument('-me', '--max_epochs', required=False, type=int,default=200,
                    help='max_epochs for training')
parser.add_argument('-edp', '--epochs_detail_plot', required=False, type=int, default=60,
                    help='Loss subplot from this epoch on')

parser.add_argument('-nce', '--n_cells_eval', required=False, type=int, default=-1,  
                    help='Max cells to be used for eval, if -1 use all cells. '+\
                   'For cell subsetting seed 0 is always used to be reproducible accros '+\
                   'runs with different seeds.')

parser.add_argument('-t', '--testing', required=False, type=intstr_to_bool,default='0',
                    help='Testing mode')



parser.add_argument('--num_macrogenes', required=False, type=int, default=1500,
                    help='Number of macrogenes')
parser.add_argument('--hv_genes', required=False, type=int, default=6000,
                    help='Number of HVG genes to be limited to in SATURN')
parser.add_argument('--hv_span', required=False, type=float, default=0.3,
                    help='hv_span parameter for seurat_v3 HVG selection.')
parser.add_argument('--latent_dim', required=False, type=int, default=32,
                    help='Latent dim in SATURN')
parser.add_argument('--hidden_dim', required=False, type=int, default=256,
                    help='Dim of hidden layers in SATURN')
parser.add_argument('--pretrain_epochs', required=False, type=int, default=50,
                    help='Pretrain Epochs')
# %%
if True:
    args= parser.parse_args(args=[
        # With one data with common vars
        # '-pa','/Users/amirali.moinfar/Downloads/pancreas_conditions_MIA_HPAP2/combined_orthologuesHVG.h5ad',
        # With two data with gene graph
        '-pa','/Users/amirali.moinfar/Downloads/pancreas_conditions_MIA_HPAP2/combined-mmPart_nonortholHVG.h5ad',
        '-pa2','/Users/amirali.moinfar/Downloads/pancreas_conditions_MIA_HPAP2/combined-hsPart_nonortholHVG.h5ad',
        '-ps','/Users/amirali.moinfar/tmp/cross_species_prediction/eval/test/integration/',
        # '-pa','/om2/user/khrovati/data/cross_species_prediction/pancreas_healthy/combined_orthologuesHVG2000.h5ad',
        # '-ps','/om2/user/khrovati/data/cross_system_integration/eval/test/integration/',
        '-sk','system',
        '-sv','mouse,human',
        '-gk','cell_type_eval',
        '-bk','batch',
        '-me','2',
        '-edp','0',
        
        '-s','1',
                
        '-nce','1000',
        
        '-t','1',

        '--pretrain_epochs', '1',
        '--hv_genes', '1000',
        '--num_macrogenes', '500',
        '--hv_span', '1.',  # Otherwise we get error on sc.pp.highly_variable_genes in SATURN because some batches have a few cells
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
    SINGLE_ADATA = False

# %%
# saturn params
num_macrogenes = args.num_macrogenes
hv_genes = args.hv_genes
latent_dim = args.latent_dim
hidden_dim = args.hidden_dim

pretrain_epochs = args.pretrain_epochs
epochs = args.max_epochs - pretrain_epochs
assert epochs > 0

# %%
# Make folder for saving
path_save=args.path_save+'saturn'+\
    '_'+''.join(np.random.permutation(list(string.ascii_letters)+list(string.digits))[:8])+\
    ('-TEST' if TESTING else '')+\
    os.sep

Path(path_save).mkdir(parents=True, exist_ok=False)
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
adata=sc.read(args.path_adata)
# Since SATURN interface is csv file and it only accepts str as species, we have to change this column.
adata.obs[args.system_key] = "spc_" + adata.obs[args.system_key].astype("str")
if SINGLE_ADATA:
    adata_2 = None
    system_val_dict = dict(zip(sorted(adata.obs[args.system_key].unique()), args.system_value.split(",")))
else:
    adata_2 = sc.read(args.path_adata_2)
    adata_2.obs[args.system_key] = "spc_" + adata_2.obs[args.system_key].astype("str")
    system_val_dict = {
        adata.obs[args.system_key].unique()[0]: args.system_value.split(",")[0],
        adata_2.obs[args.system_key].unique()[0]: args.system_value.split(",")[1],
    }
adata, adata_2, system_val_dict

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

total_mods


# %%
def prepare_adata(adata_mod, normalize_again=False, leiden_resolution=1., n_neighbors=15):
    if normalize_again:
        adata_mod.X = adata_mod.layers['counts'].copy()
        sc.pp.normalize_total(adata_mod)
        sc.pp.log1p(adata_mod)
    sc.pp.pca(adata_mod)
    sc.pp.neighbors(adata_mod, n_neighbors=n_neighbors, use_rep='X_pca')
    sc.tl.leiden(adata_mod, resolution=leiden_resolution)
    adata_mod.obs['leiden'] = adata_mod.obs['leiden'].astype(str).astype('category')
    adata_mod.X = adata_mod.layers['counts'].copy()
    # Attention: we assume var.index of dataframes have correct gene symbols
    # assert adata_mod.var['gs_mm'].is_unique
    # adata_mod.var.index = adata_mod.var['gs_mm']


# %%
mods_adata = {}
if SINGLE_ADATA:
    for mod in total_mods:
        mods_adata[mod] = adata[adata.obs[args.system_key] == mod]
        print(f"mod: {mod}\n", mods_adata[mod])
else:
    for adata_current in [adata, adata_2]:
        mod = adata_current.obs[args.system_key].unique()[0]
        mods_adata[mod] = adata_current
        print(f"mod: {mod}\n", mods_adata[mod])

# %%
species = []
paths = []
prot_embs = []
for mod in total_mods:
    adata_mod = mods_adata[mod]
    prepare_adata(adata_mod)
    print(f"mod: {mod}\n", adata_mod)
    save_path = os.path.join(path_save, f"mod_{mod}.h5ad")
    adata_mod.write(save_path)
    mods_adata[mod] = adata_mod
    species.append(mod)
    paths.append(save_path)
    prot_embs.append(os.path.join(SATURN_EMB_PATH, f"{system_val_dict[mod]}_embedding.torch"))

df = pd.DataFrame(columns=["path", "species", "embedding_path"])
df["species"] = species
df["path"] = paths
df["embedding_path"] = prot_embs
df

# %%
run_info_path = os.path.join(path_save, "run_info_for_saturn.csv")
df.to_csv(run_info_path, index=False)

# %% [markdown]
# ### Training

# %%
# The following lines are changes in SATURN:
# 1. https://github.com/snap-stanford/SATURN/blame/main/model/saturn_model.py#L215
# >> idx1 = torch.randint(low=0, high=embeddings.shape[1], size=(x1.shape[0],)) 
# << idx1 = torch.randint(low=0, high=x1.shape[0], size=(x1.shape[0],)) 

# %%
print('Train')

# %%
saturn_label_key = "leiden"
saturn_wcd = os.path.join(path_save, "saturn_wcd")
Path(saturn_wcd).mkdir(parents=True, exist_ok=True)
command = (
    f"conda run -n {SATURN_CONDA_ENV} --live-stream".split(" ") +
    ["python", "train-saturn.py", 
     "--in_data", run_info_path, "--in_label_col", saturn_label_key, "--ref_label_col", saturn_label_key, 
     "--non_species_batch_col", args.batch_key,
     "--work_dir", saturn_wcd,
     "--centroids_init_path", os.path.join(saturn_wcd, "centroids_init_path.pkl"),
     "--num_macrogenes", str(num_macrogenes), "--hv_genes", str(hv_genes), "--hv_span", str(args.hv_span), 
     "--model_dim", str(latent_dim), "--hidden_dim", str(hidden_dim),
     "--pretrain_epochs", str(pretrain_epochs), "--epochs", str(epochs),
     *(["--pretrain"] if pretrain_epochs > 0 else []), '--seed', str(args.seed)]
)

# %%
" ".join(command)

# %%
subprocess.run(command,
               cwd=SATURN_GIT_LOCATION,
               capture_output=True)

# %%
h5ad_output_filename = glob.glob(os.path.join(saturn_wcd, "saturn_results", f"*_seed_{args.seed}.h5ad"))
assert len(h5ad_output_filename) == 1
h5ad_output_filename = h5ad_output_filename[0]
run_name = h5ad_output_filename.split("/")[-1].split(".h5ad")[0]
run_name

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
latent = ad.read(h5ad_output_filename)
latent

# %%
# Compute embedding
if SINGLE_ADATA:
    all_obs_names = list(adata.obs_names)
    obs = adata.obs
else:
    all_obs_names = list(adata.obs_names) + list(adata_2.obs_names)
    obs = pd.concat([adata.obs, adata_2.obs], axis=0)

cells_eval = all_obs_names if args.n_cells_eval==-1 else \
    np.random.RandomState(seed=0).permutation(all_obs_names)[:args.n_cells_eval]
print('N cells for eval:',cells_eval.shape[0])
embed = sc.AnnData(latent[cells_eval].X, obs=obs.loc[cells_eval,:].copy())
embed.obs[args.system_key] = embed.obs[args.system_key].str.split("spc_").str[1]

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

# %%

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
