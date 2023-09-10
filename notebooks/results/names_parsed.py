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
# # Set names and cmaps for plotting

# %%
import pickle as pkl

# %%
path_data='/om2/user/khrovati/data/cross_system_integration/'
path_save=path_data+'names_parsed/'


# %% [markdown]
# ## Names

# %%
# Models
model_map={
    'non-integrated':'non-integrated',
    'vamp': 'vamp',
    'cycle': 'cycle',
    'vamp_cycle':'vamp+cycle',
    'cVAE': 'cVAE',
    'scvi': 'scVI',
    'scglue': 'GLUE',
    'saturn': 'SATURN',
    'saturn_super': 'SATURN-CT'
}
pkl.dump(model_map,open(path_save+'models.pkl','wb'))

# %%
# Models
model_map_additional={
    'vamp_fixed': 'vamp - FP',
    'gmm':'GMM',
    'gmm_ri':'GMM - RPI',
    'gmm_fixed':'GMM - FP',
}
pkl.dump(model_map_additional,open(path_save+'models_additional.pkl','wb'))

# %%
# Metrics
metric_map={
    'nmi_opt':'NMI',
    'moransi':"Moran's I",
    'ilisi_system':'iLISI',
}
pkl.dump(metric_map,open(path_save+'metrics.pkl','wb'))

# %%
# Metrics meaning
metric_meaning_map={
    'nmi_opt':'bio (coarse)',
    'moransi':"bio (fine)",
    'ilisi_system':'batch',
}
pkl.dump(metric_meaning_map,open(path_save+'metric_meanings.pkl','wb'))

# %%
# Datasets
dataset_map={
    'pancreas_conditions_MIA_HPAP2':'mouse-human',
    'retina_adult_organoid':'organoid-tissue',
    'adipose_sc_sn_updated':'cell-nuclei',
}
pkl.dump(dataset_map,open(path_save+'datasets.pkl','wb'))

# %%
# Optimized parameter values to be used alongside relevant models
param_vals=[
    (['cVAE'],[
        ('kl_weight',[1.0, 1.5, 2.0, 5.0]) ]),
    (['scvi'],[
        ('kl_weight',[0.5,1.0, 1.5, 2.0]) ]),
    (['scglue'],[
        ('lam_align',[0.0005, 0.005, 0.05, 0.5]),
        ('lam_graph',[0.005, 0.1, 0.5, 2.0]),
        ('rel_gene_weight',[0.4, 0.6, 0.8, 1.0]) ]),
    (['vamp'],[
        #('n_prior_components',[10.0, 100.0, 1000.0, 5000.0]),
        ('kl_weight',[1.0, 1.5, 2.0, 5]) 
    ]),
    (['saturn','saturn_super'],[
        ('pe_sim_penalty',[0.01, 0.1, 1.0, 10.0]) ]),
    (['cycle', 'vamp_cycle'],[
        ('z_distance_cycle_weight',[2.0, 5.0, 10.0, 50.0]) ]),
] 

pkl.dump(param_vals,open(path_save+'optimized_parameter_values.pkl','wb'))

# %%
# Optimized parameter values to be used alongside relevant models
param_vals_additional=[
    (['vamp','vamp_fixed','gmm','gmm_fixed','gmm_ri'],[
        ('n_prior_components',[1.0,2.0,5.0,10.0, 100.0, 5000.0]),
    ])
] 

pkl.dump(param_vals_additional,open(path_save+'optimized_parameter_values_additional.pkl','wb'))

# %%
# Map params_opt to model name
params_opt_map={
     'kl_weight':'cVAE',
     'saturn_pe_sim_penalty':'saturn',
     'saturn_pe_sim_penalty_no':'saturn',
     'saturn_pe_sim_penalty_super':'saturn_super',
     'saturn_pe_sim_penalty_super_no':'saturn_super',
     'scglue_lam_align':'scglue',
     'scglue_lam_align_no':'scglue',
     'scglue_lam_graph':'scglue',
     'scglue_lam_graph_no':'scglue',
     'scglue_rel_gene_weight':'scglue',
     'scglue_rel_gene_weight_no':'scglue',
     'scvi_kl_anneal':'scvi',
     'vamp_z_distance_cycle_weight_std_eval':'vamp_cycle',
     'vamp_kl_weight_eval':'vamp',
     'z_distance_cycle_weight_std':'cycle',
    }
pkl.dump(params_opt_map,open(path_save+'params_opt_model.pkl','wb'))

# %%
# Params opt used in main plots
params_opt_map_additional={
     'vamp_eval':'vamp',
     'vamp_eval_fixed':'vamp_fixed',
     'gmm_eval':'gmm',
     'gmm_eval_fixed':'gmm_fixed',
     'gmm_eval_ri':'gmm_ri',
    }
pkl.dump(params_opt_map_additional,open(path_save+'params_opt_model_additional.pkl','wb'))

# %%
# Map param names to pretty names
param_names={
 'n_prior_components':'N priors',
 'z_distance_cycle_weight':'cycle LW',
 'kl_weight':'KL LW',
 'lam_graph':'graph LW',
 'lam_align':'alignment LW',
 'rel_gene_weight':'graph W',
 'pe_sim_penalty':'protein sim. LW',
  None:'None',  
}
pkl.dump(param_names,open(path_save+'params.pkl','wb'))

# %%
# Map params_opt to gene relationship type
params_opt_gene_map={
     'kl_weight':'OTO',
     'saturn_pe_sim_penalty':'OTO',
     'saturn_pe_sim_penalty_no':'FO',
     'saturn_pe_sim_penalty_super':'OTO',
     'saturn_pe_sim_penalty_super_no':'FO',
     'scglue_lam_align':'OTO',
     'scglue_lam_align_no':'FO',
     'scglue_lam_graph':'OTO',
     'scglue_lam_graph_no':'FO',
     'scglue_rel_gene_weight':'OTO',
     'scglue_rel_gene_weight_no':'FO',
     'scvi_kl_anneal':'OTO',
     'vamp_kl_weight_eval':'OTO',
     'vamp_z_distance_cycle_weight_std_eval':'OTO',
     'z_distance_cycle_weight_std':'OTO',
    }
pkl.dump(params_opt_gene_map,open(path_save+'params_opt_genes.pkl','wb'))

# %%
# System 0-1 to name mapping
system_map={
    'pancreas_conditions_MIA_HPAP2':{'0':'mouse','1':'human'},
    'retina_adult_organoid':{'0':'organoid','1':'tissue'},
    'adipose_sc_sn_updated':{'0':'cell','1':'nuclei'},
}
pkl.dump(system_map,open(path_save+'systems.pkl','wb'))

# %% [markdown]
# ## Cmaps

# %%
import scanpy as sc
import pandas as pd
from collections import defaultdict
import matplotlib.colors as mcolors
import seaborn as sb
import numpy as np

# %%
# Model colors
models=[m for m in model_map.values() if m!='non-integrated']
colors =[mcolors.to_hex(color) for color in sb.color_palette("colorblind")]
palette=dict(zip(models,colors[:len(models)]))
pkl.dump(palette,open(path_save+'model_cmap.pkl','wb'))

# %%
# UMAP col colloring for datasets
cmaps=defaultdict(dict)
for fn,cols in [
    ('adipose_sc_sn_updated/adiposeHsSAT_sc_sn.h5ad',
     ['cluster','system','donor_id']),
    ('pancreas_conditions_MIA_HPAP2/combined_orthologuesHVG.h5ad',
     ['cell_type_eval','system','batch','leiden_system']),
    ('retina_adult_organoid/combined_HVG.h5ad',
     ['cell_type','system','sample_id'])
]:
    dataset=fn.split('/')[0]
    adata=sc.read(path_data+fn,backed='r')
    for col in cols:
        adata.obs[col]=adata.obs[col].astype(str)
        adata.obs[col]=pd.Categorical(values=adata.obs[col],
                                      categories=sorted(adata.obs[col].unique()),
                                      ordered=True)
        cmap=sc.pl._tools.scatterplots._get_palette(adata, col)
        if len(set(cmap.values()))<len(cmap):
            np.random.seed(0)
            colors = ['#%06X' % np.random.randint(0, 0xFFFFFF) for i in range(len(cmap))]
            cmap=dict(zip(cmap.keys(),colors))
        cmaps[dataset][col]=cmap       
pkl.dump(cmaps,open(path_save+'obs_col_cmap.pkl','wb'))

# %%
# Metric background colors
palette={'ilisi_system':'#F9F3F3',
             'moransi':'#EAF0F5',
             'nmi_opt':'#EFF9FB'}
pkl.dump(palette,open(path_save+'metric_background_cmap.pkl','wb'))

# %%
# Dataset colors
palette={
    'pancreas_conditions_MIA_HPAP2':'#8a9e59',
    'retina_adult_organoid':'#c97fac',
    'adipose_sc_sn_updated':'#92C2D0',
}
pkl.dump(palette,open(path_save+'dataset_cmap.pkl','wb'))

# %%
