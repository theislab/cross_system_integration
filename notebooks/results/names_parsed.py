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
    'non-integrated':'Non-integrated',
    'vamp': 'VAMP',
    'cycle': 'CYC',
    'vamp_cycle':'VAMP+CYC',
    'cVAE': 'cVAE',
    'scvi': 'scVI',
    'scglue': 'GLUE',
    'saturn': 'SATURN',
    'saturn_super': 'SATURN-CT'
}
pkl.dump(model_map,open(path_save+'models.pkl','wb'))

# %%
# Models extended
model_map_additional={
    'vamp_fixed': 'VAMP - FP',
    'gmm':'GMM',
    'gmm_ri':'GMM - RPI',
    'gmm_fixed':'GMM - FP',
    'scgen_sample':'scGEN - sample',
    'scgen_system':'scGEN - system',
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
    'nmi_opt':'Bio (coarse)',
    'moransi':"Bio (fine)",
    'ilisi_system':'Batch',
}
pkl.dump(metric_meaning_map,open(path_save+'metric_meanings.pkl','wb'))

# %%
# Datasets
dataset_map={
    'pancreas_conditions_MIA_HPAP2':'Mouse-Human',
    'retina_adult_organoid':'Organoid-Tissue',
    'adipose_sc_sn_updated':'Cell-Nuclei',
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
        ('kl_weight',[1.0, 1.5, 2.0, 5]) 
    ]),
    (['saturn','saturn_super'],[
        ('pe_sim_penalty',[0.01, 0.1, 1.0, 10.0]) ]),
    (['cycle', 'vamp_cycle'],[
        ('z_distance_cycle_weight',[2.0, 5.0, 10.0, 50.0]) ]),
] 

pkl.dump(param_vals,open(path_save+'optimized_parameter_values.pkl','wb'))

# %%
# Optimized parameter values to be used alongside relevant models - extended
param_vals_additional=[
    (['vamp','vamp_fixed','gmm','gmm_fixed','gmm_ri'],[
        ('n_prior_components',[1.0,2.0,5.0,10.0, 100.0, 5000.0]),
    ]),
    (['vamp'],[
        ('prior_components_group', [ 'BALANCED','beta', 'alpha', 'acinar','schwann', 'immune']),
        ('prior_components_system',[-1, 0, 1]),
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
# Map params_opt to model name - extended 
params_opt_map_additional={
     'vamp_eval':'vamp',
     'vamp_eval_fixed':'vamp_fixed',
     'gmm_eval':'gmm',
     'gmm_eval_fixed':'gmm_fixed',
     'gmm_eval_ri':'gmm_ri',
     'prior_group':'vamp', 
     'prior_system':'vamp', 
    }
pkl.dump(params_opt_map_additional,open(path_save+'params_opt_model_additional.pkl','wb'))

# %%
# Map param names to pretty names
param_names={
 'n_prior_components':'N priors',
 'z_distance_cycle_weight':'Cycle LW',
 'kl_weight':'KL LW',
 'lam_graph':'Graph LW',
 'lam_align':'Alignment LW',
 'rel_gene_weight':'Graph W',
 'pe_sim_penalty':'Protein sim. LW',
  None:'None',  
}
pkl.dump(param_names,open(path_save+'params.pkl','wb'))

# %%
# Map param names to pretty names - extended
param_names_additional={
    'prior_components_group':'Prior init. cell type',
    'prior_components_system':'Prior init. system',
}
pkl.dump(param_names_additional,open(path_save+'params_additional.pkl','wb'))

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
    'pancreas_conditions_MIA_HPAP2':{'0':'Mouse','1':'Human'},
    'retina_adult_organoid':{'0':'Organoid','1':'Tissue'},
    'adipose_sc_sn_updated':{'0':'Cell','1':'Nuclei'},
}
pkl.dump(system_map,open(path_save+'systems.pkl','wb'))

# %%
# Cell type labels name mapping
cell_type_map={
    'pancreas_conditions_MIA_HPAP2':{
        'acinar': 'Acinar', 
        'alpha': 'Alpha', 
        'alpha+beta': 'Alpha+Beta', 
        'alpha+delta': 'Alpha+Delta', 
        'beta': 'Beta', 
        'beta+delta': 'Beta+Delta', 
        'beta+gamma': 'Beta+Gamma', 
        'delta': 'Delta', 
        'delta+gamma': 'Delta+Gamma', 
        'ductal': 'Ductal', 
        'endo. prolif.': 'Endo. prolif.', 
        'endothelial': 'Endothelial', 
        'gamma': 'Gamma', 
        'immune': 'Immune', 
        'schwann': 'Schwann', 
        'stellate a.': 'Stellate a.', 
        'stellate q.': 'Stellate q.'
    },
    'retina_adult_organoid':{
        'B cell': 'B cell', 
        'Mueller cell': 'Mueller cell', 
        'OFF-bipolar cell': 'OFF-bipolar cell', 
        'ON-bipolar cell': 'ON-bipolar cell', 
        'T cell': 'T cell', 
        'amacrine cell': 'Amacrine cell', 
        'astrocyte': 'Astrocyte', 
        'endothelial cell of vascular tree': 'Endothelial cell of vascular tree', 
        'fibroblast': 'Fibroblast', 
        'mast cell': 'Mast cell', 
        'melanocyte': 'Melanocyte', 
        'microglial cell': 'Microglial cell', 
        'monocyte': 'Monocyte', 
        'natural killer cell': 'Natural killer cell', 
        'pericyte': 'Pericyte', 
        'retina horizontal cell': 'Retina horizontal cell', 
        'retinal cone cell': 'Retinal cone cell', 
        'retinal ganglion cell': 'Retinal ganglion cell', 
        'retinal pigment epithelial cell': 'Retinal pigment epithelial cell', 
        'retinal rod cell': 'Retinal rod cell', 
        'rod bipolar cell': 'Rod bipolar cell'
    },
    'adipose_sc_sn_updated':{
        'ASPC': 'ASPC', 
        'LEC': 'LEC', 
        'SMC': 'SMC', 
        'adipocyte': 'Adipocyte', 
        'b_cell': 'B cell', 
        'dendritic_cell': 'Dendritic cell', 
        'endometrium': 'Endometrium', 
        'endothelial': 'Endothelial', 
        'macrophage': 'Macrophage', 
        'mast_cell': 'Mast cell', 
        'monocyte': 'Monocyte', 
        'neutrophil': 'Neutrophil', 
        'nk_cell': 'NK cell', 
        'pericyte': 'Pericyte', 
        't_cell': 'T cell' 
    }
}
pkl.dump(cell_type_map,open(path_save+'cell_types.pkl','wb'))

# %%
# Prior init name mapping
prior_init_map={
    'prior_components_group':{'BALANCED':'Balanced'},
    'prior_components_system':{'-1':'Balanced'},
}
pkl.dump(prior_init_map,open(path_save+'prior_init.pkl','wb'))

# %% [markdown]
# ## Color maps

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
# UMAP metadata colloring for individual datasets
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
