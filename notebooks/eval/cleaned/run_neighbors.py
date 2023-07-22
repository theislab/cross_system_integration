# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: csi
#     language: python
#     name: csi
# ---

# %%
import scanpy as sc
import argparse

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
if False:
    args= parser.parse_args(args=[
        '-p','/om2/user/khrovati/data/cross_system_integration/eval/test/integration/example/',
        '-sk','system',
        '-gk','cell_type',
        '-bk','sample',
        '-t','1',
    ])
# Read command line args
else:
    args = parser.parse_args()
TESTING=args.testing  
print(args)

# %%
# only save if something was changed
save=False

# %%
# Load embed
embed=sc.read(args.path+'embed.h5ad')

# %%
if 'neighbors' not in embed.uns:
    save=True
    print('Computing embedding')
    # Use 90 neighbours so that this can be also used for lisi metrics
    # Scanpy umap and neihgbors have random seed set in params 
    #  so they shouldn't be affected by the global one
    sc.pp.neighbors(embed, use_rep='X', n_neighbors=90)
    sc.tl.umap(embed)
    
    # Plot embedding
    rcParams['figure.figsize']=(8,8)
    cols=[args.system_key,args.group_key,args.batch_key]
    fig,axs=plt.subplots(len(cols),1,figsize=(8,8*len(cols)))
    for col,ax in zip(cols,axs):
        sc.pl.embedding(embed,'X_umap',color=col,s=10,ax=ax,show=False,sort_order=False)
    plt.savefig(args.path+'umap.png',dpi=300,bbox_inches='tight')

# %%
# Prepare scaled data if not yet present in adata
if 'scaled_neighbors' not in embed.uns:
    save=True
    print('Computing scaled embedding')
    
    # Scaled embed
    embed_scl=embed.copy()
    sc.pp.scale(embed_scl)
    # Use 90 neighbours so that this can be also used for lisi metrics
    sc.pp.neighbors(embed_scl, use_rep='X', n_neighbors=90,key_added='scaled')
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
    plt.savefig(args.path+'umap_scaled.png',dpi=300,bbox_inches='tight')


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
    embed.write(args.path+'embed.h5ad')

# %%
print('Finished!')

# %%
