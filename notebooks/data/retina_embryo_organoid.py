# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: csp
#     language: python
#     name: csp
# ---

# %%
import scanpy as sc
import pandas as pd
import numpy as np
from collections import defaultdict
import glob
from scipy.io import mmread
from scipy.sparse import csr_matrix

import gc

from matplotlib import rcParams
import matplotlib.pyplot as plt

# %%
path_data='/lustre/groups/ml01/workspace/karin.hrovatin/data/'
path_eo=path_data+'datasets/d10_1016_j_celrep_2020_01_007/'
path_save=path_data+'cross_species_prediction/retina_organoid_embryo/'

# %%
adatas=[]
samples=[]
for f in glob.glob(path_eo+'*_filtered*/'):
    sample=f.split('/')[-2].split('_fil')[0]
    samples.append(sample)
    # Some samples have extra subdir
    if f+'barcodes.tsv.gz' not in glob.glob(f+'*') \
        and f+'barcodes.tsv.tsv' not in glob.glob(f+'*')\
        and f+'barcodes.tsv' not in glob.glob(f+'*') :
        f=glob.glob(f+'*')[0]+'/'
    
    print(sample,f)
    files=glob.glob(f+'*')
    # Different file endings shared set of base names
    mtx_f=[f for f in files if 'matrix.mtx' in f][0]
    cells_f=[f for f in files if 'barcodes.tsv' in f][0]
    genes_f=[f for f in files if 'features.tsv' in f or 'genes.tsv' in f][0]
    #print(mtx_f,cells_f,genes_f)
    mtx=mmread(mtx_f).T
    cells=pd.read_table(cells_f,index_col=0,header=None)
    genes=pd.DataFrame(pd.read_table(genes_f,index_col=0,header=None).iloc[:,0])
    cells.index.name=None
    genes.index.name=None
    genes.columns=['gene_symbol']
    #display(cells.head(1))
    #display(genes.head(1))
    adata=sc.AnnData(mtx,obs=cells,var=genes)
    adatas.append(adata)
adata=sc.concat(adatas,label='sample',keys =samples,index_unique='-',merge='same')

# %%
adata

# %%
# Load obs
obs=[]
for fig_f in glob.glob(path_eo+'GSE142526_metadata/*/'):
    fig=fig_f.split('/')[-2]
    #print(fig)
    for f in glob.glob(fig_f+'*'):
        obs_part=pd.read_table(f,sep=',')
        fig_sub=f.split('/')[-1].split('.csv')[0]
        obs_part['fig']=fig
        obs_part['fig_sub']=fig_sub
        obs_part.rename({'cluster.ident':'cell_type','type':'cell_type'},axis=1,inplace=True)
        # Find only droplet id parts without prefixes etc
        obs_part['Unnamed: 0']=obs_part['Unnamed: 0'].apply(
            lambda x:
            [i for i in x.split('_') if not i.startswith('D')][0])
        obs.append(obs_part)
obs=pd.concat(obs).reset_index(drop=True).rename(
    {'Unnamed: 0':'droplet','orig.ident':'sample'},
    axis=1)[['droplet','sample','cell_type','fig','fig_sub']]


# %% [markdown]
# Match obs samples to adata

# %%
obs.groupby(['fig','fig_sub'])['sample'].value_counts()

# %%
adata.obs['sample'].value_counts()

# %%
map_samples={
    # Some mappings were determined based on droplet name matching
 '10X_D125C':'D125Cfetal',
 '10X_D60_2':'ORG_D60b',
 '10X_PBMC':'D59_fetal',
 'D104':'ORG_D104',
 'D110':'ORG_D110',
 'D125C':'D125Cfetal',
 'D125P':'D125Pfetal',
 'D205':'ORG_D205',
 'D59':'D59_fetal',
 'D82C':'D82Cfetal',
 'D82P':'D82Pfetal',
 'D82P_old':'D80P_fetal',
 'D82P_old2':'D80P_fetal',
 'D90':'ORG_D90',
 'H7_D104_run1':'ORG_D104',
 'H7_D110_new2':'ORG_D110',
 'RS78_35':'RSD78pl35',
 'RS_D76_D109':'RSD76pl33',
 'RS_D91fovea_D109':'RSD91pl18',
 'd45_org':'ORG_D45',
 'd59_fetal':'D59_fetal',
 'd60_org':'ORG_D60b',
 'wt':'D82Cfetal'
}
# Check that droplet ids from obs are in corresponding samples from adata
for i,j in map_samples.items():
    cells=[c.split('-')[0] for c in adata.obs.query('sample==@j').index]
    if len(
        [c  for c  in obs.query('sample==@i').droplet if c not in cells])>0:
        print(i,j)

# %%
obs['sample']=obs['sample'].map(map_samples)

# %%
# Match obs index to adata
obs.index=obs.apply(
lambda x: x['droplet']+'-1-'+x['sample'],axis=1)
# check that all obs indices are in adata
cells=set(adata.obs_names)
[o for o in obs.index if o not in cells]

# %%
# Save obs
obs.to_csv(path_eo+'obs.tsv',sep='\t')

# %%
# Save adata
adata.write(path_eo+'adata.h5ad')

# %%
#adata=sc.read(path_eo+'adata.h5ad')

# %%
(adata.X>0).sum(axis=1)

# %%
adata.obs['n_genes']=np.array((adata.X>0).sum(axis=1)).ravel()

# %%
_=plt.hist(adata.obs['n_genes'],bins=100)

# %%
_=plt.hist(adata[adata.obs['n_genes']<300,:].obs['n_genes'],bins=100)
plt.axvline(240,c='k')

# %% [markdown]
# C: The data was not pp  - need to remove low count cells later (also some annotated cells have low counts (not shown).

# %% [markdown]
# ## Prepare integration data - embryo organoid with per sample anno
# Only use cells that have annotation per sample (not integrated embryo+organoid) from embryo and organoid

# %%
min_genes=240

# %%
# PP embryo
# annotated cells
obs_e=[('F1','D59_metadata'),
         ('F2','D82C_metadata'),
         ('F3','D125C_metadata')]
obs_e=pd.concat([obs.query('fig==@fig & fig_sub==@fig_sub') for fig,fig_sub in obs_e ])
adata_e=adata[obs_e.index,:]
# Remove low n genes cells 
adata_e=adata_e[adata_e.obs['n_genes']>=min_genes,:]
adata_e.obs['cell_type']=obs_e['cell_type']
adata_e.obs['system']=1

# Subset to expr genes and normalise
adata_e=adata_e[:,np.array((adata_e.X>0).sum(axis=0)>20).ravel()].copy()
adata_e.layers['counts']=adata_e.X.copy()
sc.pp.normalize_total(adata_e)
sc.pp.log1p(adata_e)
# HVG
sc.pp.highly_variable_genes(
     adata=adata_e, n_top_genes=3000, flavor='cell_ranger', batch_key='sample', subset=True)


# %%
adata_e.shape

# %%
# PP organoid
# annotated cells
obs_o=[('F4','d60_org_metadata'),
       ('F5','cca_d100org_metadata')]
obs_o=pd.concat([obs.query('fig==@fig & fig_sub==@fig_sub') for fig,fig_sub in obs_o ])
adata_o=adata[obs_o.index,:]
# Remove low n genes cells 
adata_o=adata_o[adata_o.obs['n_genes']>=min_genes,:]
adata_o.obs['cell_type']=obs_o['cell_type']
adata_o.obs['system']=0

# Subset to expr genes and normalise
adata_o=adata_o[:,np.array((adata_o.X>0).sum(axis=0)>20).ravel()].copy()
adata_o.layers['counts']=adata_o.X.copy()
sc.pp.normalize_total(adata_o)
sc.pp.log1p(adata_o)
# HVG
sc.pp.highly_variable_genes(
     adata=adata_o, n_top_genes=3000, flavor='cell_ranger', batch_key='sample', subset=True)

# %%
adata_o.shape

# %%
# Shared HVGs
shared_hvgs=list(set(adata_e.var_names) & set(adata_o.var_names))
len(shared_hvgs)

# %%
sorted(adata_o.obs.cell_type.unique())

# %%
sorted(adata_e.obs.cell_type.unique())

# %%
adata_o.obs.cell_type=adata_o.obs.cell_type.replace({
 'Amacrine':'AC',
 'Forebrain':"FB",
 'Midbrain':"MB",
 'Photoreceptors':'PR',
 'Progenitors':'Prog',
 'T3/BC':'T3',
 'iMG':'imGlia'
})

# %%
adata_e.obs.cell_type=adata_e.obs.cell_type.replace({
 'Amacrine':'AC',
 'Bipolar':'BC',
 'Cones':'PR',
 'Horizontal':'HC',
 'Microglia':'mGlia',
 'Photoreceptors':'PR',
 'Progenitors':'Prog',
 'Rods':'PR',
})

# %%
sorted(adata_o.obs.cell_type.unique())

# %%
sorted(adata_e.obs.cell_type.unique())

# %%
adata_o.obs.cell_type.value_counts()

# %%
adata_e.obs.cell_type.value_counts()

# %%
# Subset to shraed HVGs and concat
adata=sc.concat([adata_o[:,shared_hvgs], adata_e[:,shared_hvgs]],
                join='outer',
                index_unique='_', keys=['organoid','embryo'])
adata

# %%
adata.write(path_save+'organoid_embryo_sampleAnno.h5ad')

# %%
