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

# %% [markdown]
# Select features used in encoder (mouse) and decoder (human) as pancreatic cell type markers. This is done justto speed up the training. Also to explore if other tissues provide valuable info on correlation in human.

# %%
import pandas as pd
import pickle as pkl
import numpy as np

# %%
path_data='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/combined/celltypes/'
path_save='/lustre/groups/ml01/workspace/karin.hrovatin/data/cross_species_prediction/pancreas_ct_example_constraint/'

# %% [markdown]
# ## Mouse endocrine markers (from atlas)

# %%
# Load de results of each cell type
res=dict()
for dt,res_sub in pd.read_excel(
    path_data+'DEedgeR_summaries_ctParsed.xlsx',sheet_name=None,index_col=0).items():
    res[dt]=res_sub

# %%
# See how many genes would be selected at given threshold per ct
# Also collect genes
genes=dict()
for dt,res_sub in res.items():
    genes_sub=set(res_sub.query('logFC>0.5 & padj<0.05').index)
    print(dt, len(genes_sub))
    genes[dt]=genes_sub
print('N total genes:',len(set([g for gs in genes.values() for g in gs])))

# %%
pkl.dump(genes,open(path_save+'endo_markers_set_mm.pkl','wb'))

# %% [markdown]
# ## Human markers (from anotehr study)

# %%
# Find position of table for each ct in the excel sheet
cts=['ALPHA','BETA','DELTA','GAMMA']
header=pd.read_excel(path_data+'10_1038_s41467-022-29588-8_source_data.xlsx',
                     sheet_name='figure 1',nrows=1)
ct_icol_start={}
ct_icol_end={}
ct_previous=None
for idx,col in enumerate(header.columns):
    if ct_previous is not None:
        ct_icol_end[ct_previous]=idx-1
    if col in cts:
        ct_icol_start[col]=idx
        ct_previous=col
print('Starts',ct_icol_start,'Ends',ct_icol_end)


# %%
# read markers
markers={}
for ct in cts:
    # read ct sheet subset
    m=pd.read_excel(path_data+'10_1038_s41467-022-29588-8_source_data.xlsx',
                     sheet_name='figure 1',usecols=range(ct_icol_start[ct],ct_icol_end[ct])
                   ).dropna(subset=[ct],axis=0)
    # Rename cols according to 2 line header
    col1_prev=None
    cols=[]
    for col1,col2 in zip(m.columns,m.iloc[0,:].values):
        if col1==ct:
            cols.append(col2)
        else:
            if not 'Unnamed' in col1:
                col1_prev=col1
            cols.append(col1_prev.replace(' ','_').replace('#','n')+'-'+col2)
    m.columns=cols
    # Subset to remove 2nd header row from data rows
    m=m.iloc[1:,:]
    # Result
    print(ct,m.shape)
    m.replace('.',np.nan,inplace=True)
    markers[ct]=m


# %%
# Select markers that are on avergae (across datasets) significant against all other endo cts
genes=dict()
for dt,res_sub in markers.items():
    genes_sub=set(res_sub[
        ((res_sub[[c for c in res_sub.columns if 'logFC' in c and '-tot' not in c]
             ]>0.5).all(axis=1)&\
        (res_sub[[c for c in res_sub.columns if 'Padj' in c and '-tot' not in c]
             ]<0.05).all(axis=1))].gene)
    print(dt,len(genes_sub))
    genes[dt.lower()]=genes_sub
print('N total genes:',len(set([g for gs in genes.values() for g in gs])))

# %%
pkl.dump(genes,open(path_save+'endo_markers_set_hs.pkl','wb'))

# %% [markdown]
# Are there any marker overlaps

# %%
cts=list(genes.keys())
for i in range(len(cts)-1):
    for j in range(i+1,len(cts)):
        ct1=cts[i]
        ct2=cts[j]
        print(ct1,len(genes[ct1]),ct2,len(genes[ct2]),
              'overlap:',len(genes[ct1]&genes[ct2]),genes[ct1]&genes[ct2])

# %% [markdown]
# C: Some markers are overlapping across cell types - issue of averaging across datasets

# %%
