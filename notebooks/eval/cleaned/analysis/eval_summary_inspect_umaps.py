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
import pandas as pd
import numpy as np
import scanpy as sc

from matplotlib import rcParams
import matplotlib.pyplot as plt
import seaborn as sb

# %%
path_data='/om2/user/khrovati/data/cross_system_integration/eval/'

# %% [markdown]
# ## Saturn
# Check what is going on with the odd saturn cell separation

# %%
run_dir=path_data+'pancreas_conditions_MIA_HPAP2/integration/saturn_1fMcluvQ/'
embed=sc.read(run_dir+'embed.h5ad')

# %%
embed.obs['batch'].nunique()

# %%
colors = []
n = embed.obs['batch'].nunique()
for i in range(n):
    colors.append('#%06X' % np.random.randint(0, 0xFFFFFF))
embed.uns['batch_colors']=colors

# %%
sc.pl.umap(embed,color=['cell_type_eval','mm_study','leiden_system','system'])
sc.pl.umap(embed,color=['batch'])

# %% [markdown]
# C: The cell type separation in Saturn is likely due to clusters. These originate as data wasn't integrated prior clustering, so different studies have gotten different clusters.

# %%
