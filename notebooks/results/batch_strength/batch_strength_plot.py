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
import pandas as pd
import scanpy as sc
import pickle as pkl

from sklearn.metrics import roc_auc_score

import seaborn as sb
from matplotlib import rcParams
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# %%
path_data='/home/moinfar/data/'
path_fig='/home/moinfar/io/csi/figures/'
path_names='/home/moinfar/io/csi/names_parsed/'

# %%
# Load distances 
dataset_map=pkl.load(open(path_names+'datasets.pkl','rb'))
system_map=pkl.load(open(path_names+'systems.pkl','rb'))
dataset_path=pkl.load(open(path_names+'dataset_path.pkl','rb'))
dataset_h5ad_path=pkl.load(open(path_names+'dataset_h5ad_path.pkl','rb'))

# %%
distances={
    dataset:pkl.load(open(dataset_path[dataset]+'/'+dataset_h5ad_path[dataset][:-5]+'_PcaSysBatchDist.pkl','rb'))
    for dataset in dataset_path.keys()}

# %% [markdown]
# ## Plot one cell type in all sample pair groups (delta cells of pancreas)

# %%
# Prepare df for plotting (focus on delta cells)
plot=[]
ct='delta'
dat=distances['pancreas_conditions_MIA_HPAP2'][ct]
y_col='Compared samples'
for comparison,dist in dat.items():
    dist=pd.DataFrame(dist,columns=['dist'])
    dist['group']=ct
    dist[y_col]=comparison
    plot.append(dist)
plot=pd.concat(plot)

# %%
# Make data names prettier
plot.rename({'dist':'Distance'},axis=1,inplace=True)
plot.replace({'s0_within':'Mouse\n(within datasets)',
              's0_between':'Mouse\n(between datasets)',
              's1':'Human',
              's0s1':'Mouse vs human'},inplace=True)

# %%
# Plot
fig,ax=plt.subplots(figsize=(1.5,2))
sb.violinplot(y=y_col,x='Distance',data=plot,inner=None,linewidth=0.5,ax=ax)
fig.set(facecolor = (0,0,0,0))
ax.set(facecolor = (0,0,0,0))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig(path_fig+f'batch_strength-pancreas_{ct}-violin.pdf',dpi=300,bbox_inches='tight')
plt.savefig(path_fig+f'batch_strength-pancreas_{ct}-violin.png',dpi=300,bbox_inches='tight')

# %%
# N comparisons (samples) per group
plot.groupby(y_col).size()

# %% [markdown]
# ## Plot distance of between vs within comparisons

# %%
# Prepare plotting df
scores=[]
for dataset, distances_ds in distances.items():
    for cell_type,distances_ct in distances_ds.items():
        for group,distances_g in distances_ct.items():
            if len(distances_g)>0:
                scores.append({
                    'dataset':dataset,'cell_type':cell_type,'compared':group,
                    'score':distances_g.mean()
                })
scores=pd.DataFrame(scores)
# Plot  distances

# Use gridspec as different datasets have different number of categories
heights=[scores.query('dataset==@dataset').compared.nunique() for dataset in dataset_map]
nrows=len(heights)
fig = plt.figure(figsize=(1.4, 1.4 * len(heights)))
fig.set(facecolor = (0,0,0,0))
gs = GridSpec(nrows, 1, height_ratios=heights)
# Set one ax as ref so that ax can be shared
ax0=None
for idx,(dataset,dataset_name) in enumerate(dataset_map.items()):
    ax = fig.add_subplot(gs[idx, 0], sharex=ax0)
    if ax0 is None:
        ax0=ax
    data_plot=scores.query('dataset==@dataset').copy()
    # Parse names of comparison groups
    compared_map={
        x: (system_map[dataset][x.replace('s','').split('_')[0]
                              ]+(f'\n({x.split("_")[1]} datasets)' if '_' in x else ''))
        if x!='s0s1' else 'Between systems'
        for x in data_plot.compared.unique()}
    data_plot['compared_name']=data_plot.compared.map(compared_map)
    sb.swarmplot(x='score',y='compared_name',data=data_plot,ax=ax, c='k',s=2)
    # ax.set_ylabel(dataset_name.replace('-','-\n'))
    ax.set_ylabel("\n".join(dataset_name.rsplit(" ", 1)))
    if idx==nrows-1:
        ax.set_xlabel('Distances between samples\n(mean per cell type)')
        ax.xaxis.set_label_coords(-0.11, -0.3)
        plt.locator_params(axis='x', nbins=3)
    else:
        ax.xaxis.set_visible(False)
    ax.yaxis.set_label_coords(-1.4, 0.5)
    ax.set(facecolor = (0,0,0,0))
    ax.spines['right'].set_visible(False)
    if idx==0:
        ax.spines['top'].set_visible(False)
plt.subplots_adjust(hspace=0)
plt.savefig(path_fig+f'batch_strength-overview_absolute-swarm.pdf',dpi=300,bbox_inches='tight')
plt.savefig(path_fig+f'batch_strength-overview_absolute-swarm.png',dpi=300,bbox_inches='tight')

# %%

# %%
