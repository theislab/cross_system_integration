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

from scipy.stats import mannwhitneyu

import seaborn as sb
from matplotlib import rcParams
import matplotlib.pyplot as plt

# %%
path_data='/home/moinfar/data/'
path_fig='/home/moinfar/io/csi/figures/'
path_names='/home/moinfar/io/csi/names_parsed/'

# %%
# Names
model_map=pkl.load(open(path_names+'models.pkl','rb'))
param_map=pkl.load(open(path_names+'params.pkl','rb'))
metric_map=pkl.load(open(path_names+'metrics.pkl','rb'))
metric_map=dict([(k,v) if k!='nmi_opt' else ('nmi','NMI fixed') for k,v in metric_map.items() ])
dataset_map=pkl.load(open(path_names+'datasets.pkl','rb'))
metric_meaning_map=pkl.load(open(path_names+'metric_meanings.pkl','rb'))
metric_meaning_map['nmi']=metric_meaning_map['nmi_opt']
metric_map_rev=dict(zip(metric_map.values(),metric_map.keys()))
dataset_map_rev=dict(zip(dataset_map.values(),dataset_map.keys()))
system_map=pkl.load(open(path_names+'systems.pkl','rb'))
params_opt_map=pkl.load(open(path_names+'params_opt_model.pkl','rb'))
params_opt_gene_map=pkl.load(open(path_names+'params_opt_genes.pkl','rb'))
param_opt_vals=pkl.load(open(path_names+'optimized_parameter_values.pkl','rb'))
cell_type_map=pkl.load(open(path_names+'cell_types.pkl','rb'))

# cmap
model_cmap=pkl.load(open(path_names+'model_cmap.pkl','rb'))
obs_col_cmap=pkl.load(open(path_names+'obs_col_cmap.pkl','rb'))
metric_background_cmap=pkl.load(open(path_names+'metric_background_cmap.pkl','rb'))
metric_background_cmap['nmi']=metric_background_cmap['nmi_opt']

# %%
# Load results
dataset_metric_fns={
    'pancreas_conditions_MIA_HPAP2':('pancreas_conditions_MIA_HPAP2','combined_orthologuesHVG'),
    'retina_adult_organoid':('retina_adult_organoid', 'combined_HVG'),
    'adipose_sc_sn_updated':('adipose_sc_sn_updated', 'adiposeHsSAT_sc_sn'),
    'retina_atlas_sc_sn':('human_retina_atlas', 'human_retina_atlas_sc_sn_hvg'),
    'skin_mm_hs':('skin_mouse_human/processed', 'skin_mm_hs_hvg'),
    'skin_mm_hs_limited':('skin_mouse_human/processed', 'limited_data_skin_mm_hs_hvg'),
}
res={}
for dataset,(dataset_path,fn_part) in dataset_metric_fns.items():
    res[dataset]=pkl.load(open(f'{path_data}{dataset_path}/{fn_part}_embed_integrationMetrics.pkl','rb'
                              ))['asw_batch']['asw_data_label'].values

# %%
# Make DF from results for plotting
score_name='ASW system'
ds_name='Dataset'
res=pd.Series(res).explode().rename(score_name).reset_index().rename({'index':'dataset'},axis=1)
res[ds_name]=res['dataset'].map(dataset_map)

# %%
# Plot
fig,ax=plt.subplots(figsize=(2,3))
sb.swarmplot(x=score_name,y=ds_name,data=res,s=3,c='k')
ax.set(facecolor = (0,0,0,0))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)           
fig.set(facecolor = (0,0,0,0))
plt.subplots_adjust( wspace=0.1)

plt.savefig(path_fig+'batch_strength_datasets-asw_label-swarm.pdf',
            dpi=300,bbox_inches='tight')
plt.savefig(path_fig+'batch_strength_datasets-asw_label-swarm.png',
            dpi=300,bbox_inches='tight')

# %%
# Significance of batch effect differences across data asettings
dss=sorted(res['dataset'].unique())
for i in range(len(dss)-1):
    for j in range(i+1,len(dss)):
        ds_i=dss[i]
        ds_j=dss[j]
        u,p=mannwhitneyu(
            res.query('dataset==@ds_i')[score_name].astype(float), 
            res.query('dataset==@ds_j')[score_name].astype(float))
        print(ds_i,'VS',ds_j)
        print(f'p-value = {p:.1e} u={u}')

# %%
