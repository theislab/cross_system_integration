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
import glob

# %%
path='/om2/user/khrovati/data/cross_system_integration/pancreas_conditions_MIA_HPAP2/'

# %%
batches=['STZ_G2','VSG_MUC13632','HPAP-020','HPAP-029']

# %%
for f in glob.glob(path+'*h5ad'):
    print(f)
    #print(f.replace('HPAP2/','HPAP2/test/'))
    a=sc.read(f)
    a=a[a.obs.batch.isin(batches),:]
    print(a.shape)
    a.write(f.replace('HPAP2/','HPAP2/test/'))
