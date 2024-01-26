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
import pickle as pkl
import argparse
import pandas as pd
import numpy as np
import os
from collections import defaultdict

from metrics import ilisi, clisi, asw_label, cluster_classification,\
cluster_classification_optimized, knn_purity
import scib_metrics as sm

# %%
parser = argparse.ArgumentParser()
def intstr_to_bool(x):
    return bool(int(x))
parser.add_argument('-p', '--path', required=True, type=str,
                    help='directory path for reading embed from and saving results')
parser.add_argument('-fe', '--fn_expr', required=True, type=str,
                    help='file name for reading adata with expression information')
parser.add_argument('-fmi', '--fn_moransi', required=True, type=str,
                    help='file name for reading Morans I information')
parser.add_argument('-sk', '--system_key', required=True, type=str,
                    help='obs col with system info')
parser.add_argument('-gk', '--group_key', required=True, type=str,
                    help='obs col with group info')
parser.add_argument('-bk', '--batch_key', required=True, type=str,
                    help='obs col with batch info')
parser.add_argument('-s', '--scaled', required=False, type=intstr_to_bool, default='0',
                    help='Should scaled X be used. Assumes X in embed data is still unscaled. '+
                    'Assumes there are neighbors with prefix '+
                    'scaled_ (for dist, conn) in the embedding.')
parser.add_argument('-co', '--cluster_optimized', required=False, type=intstr_to_bool, default='1',
                    help='Should clustering metrics on optimized clustering resolution be computed')
parser.add_argument('-t', '--testing', required=False, type=intstr_to_bool,default='0',
                    help='Testing mode')

# %%
# Set args for manual testing
if False:
    args= parser.parse_args(args=[
        '-p','/om2/user/khrovati/data/cross_system_integration/eval/test/integration/example/',
        '-fe','/om2/user/khrovati/data/cross_species_prediction/pancreas_healthy/combined_orthologuesHVG2000.h5ad',
        '-fmi','/om2/user/khrovati/data/cross_system_integration/eval/test/integration/example/moransiGenes_mock.pkl',
        '-sk','system',
        '-gk','cell_type',
        '-bk','sample',
        '-s','1',
        '-co','1',
        '-t','1',
    ])
# Read command line args
else:
    args = parser.parse_args()
TESTING=args.testing    
print(args)

# %%
# Load embedding (embed - subset for eval, 
# embed_full is loaded below - all cells from integration data)
embed = sc.read(args.path+'embed.h5ad')

# %%
# Prepare for scaled/unscaled setting
if args.scaled:
    sc.pp.scale(embed)
    neigh_prefix='scaled_'
else:
    neigh_prefix=''

# %%
# Which cells have group info - if nan dont use them for metrics computation 
# Used only for testing - otherwise this would require recomputing neighbors here which 
# would be inefficient
if embed.obs[args.group_key].isna().any():
    if TESTING:
        embed_group=embed
        embed_group.obs[args.group_key]=embed_group.obs[args.group_key].astype(str).fillna('NA')
    else:
        raise ValueError('nan group cells in adata')
else:
    embed_group=embed

# %%
# Dict for saving metrics
fn_scaled='_scaled' if args.scaled else ''
fn_metrics=args.path+f'scib_metrics{fn_scaled}.pkl'
if os.path.exists(fn_metrics):
    metrics=pkl.load(open(fn_metrics,'rb'))
else:
    metrics={}

# Dict for saving extra metric data
fn_metrics_data=args.path+f'scib_metrics_data{fn_scaled}.pkl'
if os.path.exists(fn_metrics_data):
    metrics_data=pkl.load(open(fn_metrics_data,'rb'))
else:
    metrics_data={}

# %%
# Which metrics to compute
# For now only computes metrics that are not yet computed
if 'asw_group' in metrics and 'asw_group_label' in metrics_data:
    ASW_GROUP=False
else:
    ASW_GROUP=True

if 'clisi' in metrics and 'clisi_label' in metrics_data:
    CLISI=False
else:
    CLISI=True  

if 'ilisi_system' in metrics and 'ilisi_system_label' in metrics_data:
    ILISI_SYSTEM=False
else:
    ILISI_SYSTEM=True  

if 'nmi' in metrics and 'ari' in metrics and\
    'jaccard' in metrics and 'jaccard_label' in metrics_data:
    CLUSTER_CLASSIFICATION=False
else:
    CLUSTER_CLASSIFICATION=True  

if not(('nmi_opt' not in metrics or 'ari_opt' not in metrics) and args.cluster_optimized):
    CLUSTER_OPTIMIZED=False
else:
    CLUSTER_OPTIMIZED=True
    
if 'knn_purity_macro' in metrics and 'knn_purity' in metrics_data:
    KNN_PURITY=False
else:
    KNN_PURITY=True  

if 'moransi' in metrics and 'moransi_label' in metrics_data and 'moransi_data' in metrics_data:
    MORANSI=False
else:
    MORANSI=True   

if all(['ilisi_batch_system-'+system in metrics and 
    'ilisi_batch_label_system-'+system in metrics_data
    for system in embed_group.obs[args.system_key].unique() ]):
    ILISI_BATCH_SYSTEM=False
else:
    ILISI_BATCH_SYSTEM=True

# %%
# System and group lisi
if ILISI_SYSTEM or TESTING:
    print('ilisi system')
    metrics['ilisi_system'], metrics['ilisi_system_macro'], metrics_data[
        'ilisi_system_label']=ilisi(
        X=embed_group.obsp[neigh_prefix+'distances'],
        batches=embed_group.obs[args.system_key], 
        labels=embed_group.obs[args.group_key])
if CLISI or TESTING:
    print('clisi')
    metrics['clisi'], metrics['clisi_macro'], metrics_data['clisi_label']=clisi(
        X=embed_group.obsp[neigh_prefix+'distances'],
        labels=embed_group.obs[args.group_key])
# Group asw
if ASW_GROUP or TESTING:
    print('asw_group')
    metrics['asw_group'], metrics['asw_group_macro'], metrics_data['asw_group_label']= asw_label(
        X=embed_group.X, 
        labels=embed_group.obs[args.group_key])

# Cluster classification
if CLUSTER_CLASSIFICATION or TESTING:
    print('cluster_classification')
    metrics['nmi'], metrics['ari'], \
    metrics['jaccard'], metrics['jaccard_macro'], metrics_data['jaccard_label']=\
    cluster_classification(
        labels=embed_group.obs[args.group_key],
        clusters=embed_group.obs[neigh_prefix+'leiden'])

if CLUSTER_OPTIMIZED or TESTING:
    print('cluster_optmimized')
    metrics['nmi_opt'], metrics['ari_opt'] =\
    cluster_classification_optimized(
        X=embed_group.obsp[neigh_prefix+'connectivities'], 
        labels=embed_group.obs[args.group_key])

if KNN_PURITY or TESTING:
    print('knn_purity')
    metrics['knn_purity_macro'],metrics_data['knn_purity']=\
    knn_purity(
        distances=embed_group.obsp[neigh_prefix+'distances'],
        labels=embed_group.obs[args.group_key])

# %%
# Moran's I
if MORANSI or  TESTING:
    # Load adata with expression and Moran's I base values and full embedding
    adata_expr=sc.read(args.fn_expr)
    moransi_base=pkl.load(open(args.fn_moransi,'rb'))
    if os.path.exists(args.path+'embed_full.h5ad'):
        embed_full=sc.read(args.path+'embed_full.h5ad')
    else:
        embed_full=embed.copy()
        print('Full embedding equals eval embedding')
    # Prepare for scaled/unscaled setting
    if args.scaled:
        # This could be done only in the case where embed_full is loaded anew as
        # embed is already scaled. But it doesnt change much if normal-scaling 2x
        sc.pp.scale(embed_full)
    
    # Compute Moran's I per celltype-sample group and obtain difference with base level
    if MORANSI or TESTING:
        print('moransi')
        moransi_data=[]
        for group_mi in moransi_base:
            res=dict(group=group_mi['group'],
                     system=group_mi['system'],
                     batch=group_mi['batch'])
            embed_sub=embed_full[
                (embed_full.obs[args.group_key]==group_mi['group']).values&
                (embed_full.obs[args.system_key]==str(group_mi['system'])).values&
                (embed_full.obs[args.batch_key]==group_mi['batch']).values,:].copy()
            # Check that there are enough cells for testing
            if not TESTING or embed_sub.shape[0]>50:
                sc.pp.neighbors(embed_sub, use_rep='X')
                genes=group_mi['genes'].index
                res['moransi_genes']=pd.Series(
                    (sc.metrics._morans_i._morans_i(
                        g=embed_sub.obsp['connectivities'],
                        vals=adata_expr[embed_sub.obs_names,genes].X.T)+1)/2,
                    index=genes)
                res['moransi_diff']=(res['moransi_genes']/group_mi['genes']).mean()
                moransi_data.append(res)
        metrics_data['moransi_data']=moransi_data

        # Average MI diffs across samples per cell type
        metrics_data['moransi_label']=pd.DataFrame([
            {'label':i['group'],'moransi':i['moransi_diff']} for i in moransi_data
        ]).groupby('label').mean()
        # Average MI diffs accross cell types
        metrics['moransi']=metrics_data['moransi_label'].mean()[0]

# %%
# Compute batch lisi metrics per system as else it would be confounded by system
if ILISI_BATCH_SYSTEM or TESTING:
    print('ilisi_batch_system')
    for system in sorted(embed_group.obs[args.system_key].unique()):
        embed_sub=embed_group[embed_group.obs[args.system_key]==system,:].copy()
        sc.pp.neighbors(embed_sub, use_rep='X', n_neighbors=90)
        metrics['ilisi_batch_system-'+system], metrics[
            'ilisi_batch_macro_system-'+system], metrics_data[
            'ilisi_batch_label_system-'+system]=ilisi(
            X=embed_sub.obsp['distances'], 
            batches=embed_sub.obs[args.batch_key],
            labels=embed_sub.obs[args.group_key])

# %%
print(metrics)

# %%
# Save metricsß
pkl.dump(metrics,open(fn_metrics,'wb'))
pkl.dump(metrics_data,open(fn_metrics_data,'wb'))

# %% [markdown]
# # End

# %%
print('Finished metrics!')

# %%
