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

# %% [markdown]
# # Integration metrics

# %%
import scanpy as sc
import pickle as pkl
import argparse

import scib_metrics as sm

# %%
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', required=True, type=str,
                    help='directory path for reading embed from and saving results')
parser.add_argument('-sk', '--system_key', required=True, type=str,
                    help='obs col with system info')
parser.add_argument('-gk', '--group_key', required=True, type=str,
                    help='obs col with group info')
parser.add_argument('-bk', '--batch_key', required=True, type=str,
                    help='obs col with batch info')

# %%
if False:
    args= parser.parse_args(args=[
        '-p','/om2/user/khrovati/data/cross_system_integration/eval/test/integration/example/',
        '-sk','system',
        '-gk','cell_type',
        '-bk','sample'
    ])
# Read command line args
else:
    args = parser.parse_args()
    
print(args)

# %%
# Load embedding
embed = sc.read(args.path+'embed.h5ad')

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
#metrics['asw_system']=sm.silhouette_batch(
#    X=embed_group.X, labels=embed_group.obs[args.group_key], 
#    batch=embed_group.obs[args.system_key], rescale = True) 
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
    #metrics['asw_batch_system-'+system]= sm.silhouette_batch(
    #    X=embed_sub.X, labels=embed_sub.obs[args.group_key], batch=embed_sub.obs[args.batch_key],
    #    rescale = True)

# %%
print(metrics)

# %%
pkl.dump(metrics,open(args.path+'scib_metrics.pkl','wb'))

# %% [markdown]
# # End

# %%
print('Finished metrics!')

# %%
