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
import numpy as np
import scanpy as sc
import argparse
import pathlib

from matplotlib.pyplot import rcParams
import matplotlib.pyplot as plt

# %%
parser = argparse.ArgumentParser()
def intstr_to_bool(x):
    return bool(int(x))
parser.add_argument('-p', '--path', required=True, type=str,
                    help='directory path for reading embed from and saving results')
parser.add_argument('-sk', '--system_key', required=True, type=str,
                    help='obs col with system info')
parser.add_argument('-gk', '--group_key', required=True, type=str,
                    help='obs col with group info')
parser.add_argument('-bk', '--batch_key', required=True, type=str,
                    help='obs col with batch info')
parser.add_argument('-t', '--testing', required=False, type=intstr_to_bool,default='0',
                    help='Testing mode')

# %%
# Set args for manual testing
if False:
    args= parser.parse_args(args="--path /home/moinfar/io/csi/eval/retina_adult_organoid/integration/seurat_r3_hPWuSmEE --system_key system --group_key cell_type --batch_key sample_id --testing 0".split(" "))
# Read command line args
else:
    args = parser.parse_args()
TESTING=args.testing  
print(args)

# %%
# only save if something was changed
save=False
exp_path = pathlib.Path(args.path)

# %%
# Load embed
embed=sc.read(exp_path / 'embed.h5ad')

# %%

# %%
# Compute neighbors
if 'neighbors' not in embed.uns:
    save=True
    print('Computing embedding')
    # Copy data
    embed_cp=embed.copy()
    # Use 90 neighbours so that this can be also used for lisi metrics
    sc.pp.neighbors(embed_cp, use_rep='X', n_neighbors=90)

    ################# Correct neighborhood if overlapping cells exist ########################
    X = embed_cp.obsp['distances']
    n_neighbors = np.unique(X.nonzero()[0], return_counts=True)[1]
    if len(np.unique(n_neighbors)) > 1:
        "There are zero distances. Recomputing with some noise ..."
        embed_cp.X = embed_cp.X + np.random.randn(*embed_cp.X.shape).clip(-1, 1) * 1e-3
        sc.pp.neighbors(embed_cp, use_rep='X', n_neighbors=90)
    X = embed_cp.obsp['distances']
    n_neighbors = np.unique(X.nonzero()[0], return_counts=True)[1]
    if len(np.unique(n_neighbors)) > 1:
        raise ValueError("Failed to Corrent overlapping cells")
    ################# End correction ########################
    
    sc.tl.umap(embed_cp)
    
    # Add back to embed
    embed.uns['neighbors']=embed_cp.uns['neighbors']
    embed.obsp['connectivities']=embed_cp.obsp['connectivities']
    embed.obsp['distances']=embed_cp.obsp['distances']
    embed.obsm['X_umap']=embed_cp.obsm['X_umap']
    del embed_cp
    
    # Plot embedding
    rcParams['figure.figsize']=(8,8)
    cols=[args.system_key,args.group_key,args.batch_key]
    fig,axs=plt.subplots(len(cols),1,figsize=(8,8*len(cols)))
    for col,ax in zip(cols,axs):
        sc.pl.embedding(embed,'X_umap',color=col,s=10,ax=ax,show=False,sort_order=False)
    plt.savefig(exp_path / 'umap.png',dpi=300,bbox_inches='tight')

# %%
# Compute neighbors on scaled data
# Prepare scaled data if not yet present in adata
if 'scaled_neighbors' not in embed.uns:
    save=True
    print('Computing scaled embedding')
    
    # Scaled embed
    embed_scl=embed.copy()
    sc.pp.scale(embed_scl)
    # Use 90 neighbours so that this can be also used for lisi metrics
    sc.pp.neighbors(embed_scl, use_rep='X', n_neighbors=90,key_added='scaled')

    ################# Correct neighborhood if overlapping cells exist ########################
    X = embed_scl.obsp['distances']
    n_neighbors = np.unique(X.nonzero()[0], return_counts=True)[1]
    if len(np.unique(n_neighbors)) > 1:
        "There are zero distances. Recomputing with some noise ..."
        embed_scl.X = embed_scl.X + np.random.randn(*embed_scl.X.shape).clip(-1, 1) * 1e-3
        sc.pp.neighbors(embed_scl, use_rep='X', n_neighbors=90)
    X = embed_scl.obsp['distances']
    n_neighbors = np.unique(X.nonzero()[0], return_counts=True)[1]
    if len(np.unique(n_neighbors)) > 1:
        raise ValueError("Failed to Corrent overlapping cells")
    ################# End correction ########################
    
    sc.tl.umap(embed_scl,neighbors_key='scaled')
    # Add back to embed
    embed.uns['scaled_neighbors']=embed_scl.uns['scaled']
    embed.obsp['scaled_connectivities']=embed_scl.obsp['scaled_connectivities']
    embed.obsp['scaled_distances']=embed_scl.obsp['scaled_distances']
    embed.obsm['X_umap_scaled']=embed_scl.obsm['X_umap']
    del embed_scl
    
    # Plot scaled embedding
    rcParams['figure.figsize']=(8,8)
    cols=[args.system_key,args.group_key,args.batch_key]
    fig,axs=plt.subplots(len(cols),1,figsize=(8,8*len(cols)))
    for col,ax in zip(cols,axs):
        sc.pl.embedding(embed,'X_umap_scaled',color=col,s=10,ax=ax,show=False,sort_order=False)
    plt.savefig(exp_path / 'umap_scaled.png',dpi=300,bbox_inches='tight')

# %%
# Compute clusters
if 'leiden' not in embed.obs.columns:
    save=True
    print('Computing leiden')
    sc.tl.leiden(embed, resolution=2, key_added='leiden', neighbors_key=None)
if 'scaled_leiden' not in embed.obs.columns:
    save=True
    print('Computing scaled leiden')
    sc.tl.leiden(embed, resolution=2, key_added='scaled_leiden', neighbors_key='scaled_neighbors')

# %%
# Save embed
if save:
    print('Saving')
    embed.write(exp_path / 'embed.h5ad')

# %%
print('Finished!')

# %%
