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
import numpy as np
import scib_metrics as sm
from collections import defaultdict
import pandas as pd
from matplotlib import rcParams
import matplotlib.pyplot as plt
import scanpy as sc

import seaborn as sb

# %% [markdown]
# # ASW batch
# Effect of adding N noise features to batch-split feature on ASW batch

# %%
n=100
x_diff=np.concatenate([np.random.normal(0,1,size=n),np.random.normal(1,1,size=n)]).reshape(-1,1)
batch=np.array(['a']*n+['b']*n)

# %%
rcParams['figure.figsize']=(2.5,2)
sb.violinplot(y=x_diff.ravel(),x=batch)
plt.xlabel('batch')
plt.ylabel('feature')

# %% [markdown]
# ### 1 batch & N-1 noise features

# %%
res=[]
for n_noise in [0,1,2,4,8]:
    for i in range(10):
        # Regenerate also the feature taht differs in batch
        x_diff=np.concatenate(
            [np.random.normal(0,1,size=n),np.random.normal(1,1,size=n)]).reshape(-1,1)
        x_noise=np.random.normal(0,1,size=(2*n,n_noise))
        x=np.concatenate([x_diff,x_noise],axis=1)
        asw=sm.silhouette_batch(
            X=x, labels=np.array(['l']*2*n), 
            batch=batch, rescale = True)
        adata=sc.AnnData(x)
        sc.pp.neighbors(adata,use_rep='X')
        ilisi=sm.ilisi_knn(X=adata.obsp['distances'], batches=batch, scale=True)
        res.append({'n_random_features':n_noise,'rep':i,'asw_batch':asw,'ilisi':ilisi})
    print(res[n_noise])
res=pd.DataFrame(res)

# %%
rcParams['figure.figsize']=(2.5,1.5)
sb.swarmplot(x='n_random_features',y='asw_batch',data=res)

# %%
rcParams['figure.figsize']=(2.5,1.5)
sb.swarmplot(x='n_random_features',y='ilisi',data=res)

# %% [markdown]
# #### n batch & N-n random features

# %%
res=[]
n_total=9
for n_noise in [0,1,2,4,8]:
    for i in range(10):
        # Regenerate also the feature taht differs in batch
        x_diff=np.concatenate(
            [np.random.normal(0,1,size=(n,n_total-n_noise)),
             np.random.normal(1,1,size=(n,n_total-n_noise))])
        x_noise=np.random.normal(0,1,size=(2*n,n_noise))
        x=np.concatenate([x_diff,x_noise],axis=1)
        asw=sm.silhouette_batch(
            X=x, labels=np.array(['l']*2*n), 
            batch=batch, rescale = True)
        adata=sc.AnnData(x)
        sc.pp.neighbors(adata,use_rep='X')
        ilisi=sm.ilisi_knn(X=adata.obsp['distances'], batches=batch, scale=True)
        res.append({'n_random_features':n_noise,'rep':i,'asw_batch':asw,'ilisi':ilisi})
    print(res[n_noise])
res=pd.DataFrame(res)

# %%
rcParams['figure.figsize']=(2.5,1.5)
sb.swarmplot(x='n_random_features',y='asw_batch',data=res)

# %%
rcParams['figure.figsize']=(2.5,1.5)
sb.swarmplot(x='n_random_features',y='ilisi',data=res)

# %% [markdown]
# #### 1 batch feature & n1 random features with large noise & N-n1-1 features with small noise

# %%
res=[]
n_total=9
for n_noise in [0,1,2,4,8]:
    for i in range(10):
        # Regenerate also the feature taht differs in batch
        x_diff=np.concatenate(
            [np.random.normal(0,1,size=(n,1)),
             np.random.normal(1,1,size=(n,1))])
        x_noise=np.random.normal(0,1,size=(2*n,n_noise))
        x_noise_small=np.random.normal(0,0.1,size=(2*n,n_total-1-n_noise))
        x=np.concatenate([x_diff,x_noise,x_noise_small],axis=1)
        asw=sm.silhouette_batch(
            X=x, labels=np.array(['l']*2*n), 
            batch=batch, rescale = True)
        adata=sc.AnnData(x)
        sc.pp.neighbors(adata,use_rep='X')
        ilisi=sm.ilisi_knn(X=adata.obsp['distances'], batches=batch, scale=True)
        res.append({'n_random_features':n_noise,'rep':i,'asw_batch':asw,'ilisi':ilisi})
    print(res[n_noise])
res=pd.DataFrame(res)

# %%
rcParams['figure.figsize']=(2.5,1.5)
sb.swarmplot(x='n_random_features',y='asw_batch',data=res)
plt.xlabel('n_random_largeSTD')

# %%
rcParams['figure.figsize']=(2.5,1.5)
sb.swarmplot(x='n_random_features',y='ilisi',data=res)
plt.xlabel('n_random_largeSTD')

# %%
