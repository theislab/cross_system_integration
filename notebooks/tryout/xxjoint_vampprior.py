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
# Data pp imports
import scanpy as sc
import pickle as pkl
import pandas as pd
from scipy.sparse import csr_matrix, find
import numpy as np
from sklearn.preprocessing import minmax_scale
from scipy.stats import entropy
from sklearn.cluster import MiniBatchKMeans

import gc

from matplotlib.pyplot import rcParams
import matplotlib.pyplot as plt
import seaborn as sb

# %%
# Modelling imports
import torch

import sys
sys.path.insert(0, '/lustre/groups/ml01/code/karin.hrovatin/cross_species_prediction/')
import constraint_pancreas_example.model._xxjointmodel as xxjm
import importlib
importlib.reload(xxjm)
from constraint_pancreas_example.model._xxjointmodel import XXJointModel

# %%
path_data='/lustre/groups/ml01/workspace/karin.hrovatin/data/cross_species_prediction/pancreas_example/'
path_genes='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/gene_lists/'
path_tabula='/lustre/groups/ml01/workspace/karin.hrovatin/data/cross_species_prediction/tabula/'

# %% [markdown]
# ## Pancreas

# %%
adata=sc.read(path_data+'combined_orthologues.h5ad')

# %%
adata_training = XXJointModel.setup_anndata(
    adata=adata,
    system_key='system',
    #class_key='cell_type_final',
    categorical_covariate_keys=['study_sample'],
)

# %% [markdown]
# ### cVAE - VampPrior (n_components=100)

# %%
model = XXJointModel(adata=adata_training, prior='vamp', n_prior_components=100,
                    pseudoinputs_data_init=False)
model.train(max_epochs=20,
           plan_kwargs={'loss_weights':{
               'reconstruction_mixup_weight':0,
               'reconstruction_cycle_weight':0,
               'kl_cycle_weight':0,
               'z_distance_cycle_weight':0,
               'translation_corr_weight':0,
               'z_contrastive_weight':0,
               
           }})

# %%
# Plot all loses
losses=[k for k in model.trainer.logger.history.keys() 
        if '_step' not in k and '_epoch' not in k]
fig,axs=plt.subplots(2,len(losses),figsize=(len(losses)*3,4))
for ax_i,l in enumerate(losses):
    axs[0,ax_i].plot(
        model.trainer.logger.history[l].index,
        model.trainer.logger.history[l][l])
    axs[0,ax_i].set_title(l)
    axs[1,ax_i].plot(
        model.trainer.logger.history[l].index[20:],
        model.trainer.logger.history[l][l][20:])
fig.tight_layout()

# %%
embed = model.embed(
        adata=adata_training,
        indices=None,
        batch_size=None,
        as_numpy=True)

# %%
embed=sc.AnnData(embed,obs=adata_training.obs)
embed.obs['species']=embed.obs.system.map({0:'mm',1:'hs'})

# %%
sc.pp.neighbors(embed, use_rep='X')
sc.tl.umap(embed)

# %%
rcParams['figure.figsize']=(8,8)
np.random.seed(0)
random_indices=np.random.permutation(list(range(embed.shape[0])))
sc.pl.umap(embed[random_indices,:],
           color=['cell_type_final', 'study_sample',  'species'],s=10, wspace = 0.5)

# %% [markdown]
# C: Interestingly, VampPriror may even help with integration. But it may loose some bio var (some cts merged).

# %%
rcParams['figure.figsize']=(5,3)
_=plt.violinplot(embed.to_df())

# %%
ct_var={}
ct_entropy={}
for group,cells in embed.obs.groupby(['species']):
    embed_sub=embed[cells.index,:]
    # Minmax scale the embed comonents so that perceived var is not affected by range
    x=pd.DataFrame(minmax_scale(embed_sub.X),index=embed_sub.obs_names)
    #var_idx=x.var().sort_values(ascending=False).index
    x['cell_type']=embed_sub.obs['cell_type_final']
    x=x.groupby('cell_type').mean()
    #x=x[var_idx]
    g=sb.clustermap(x,cmap='cividis',figsize=(7,5),
                    col_cluster=False,row_cluster=False
                   )
    name='_'.join([group] if isinstance(group,str) else group)
    g.fig.suptitle(name)
    ct_var[name]=x.var()
    ct_entropy[name]=entropy(x)
ct_var=pd.DataFrame(ct_var)
ct_entropy=pd.DataFrame(ct_entropy)
fig,ax=plt.subplots(1,2,figsize=(8,3))
ct_var.boxplot(ax=ax[0])
ax[0].set_title('var over cts')
ct_entropy.boxplot(ax=ax[1])
ax[1].set_title('entropy over cts')

# %% [markdown]
# ### cVAE - VampPrior (n_components=10)

# %%
model = XXJointModel(adata=adata_training, prior='vamp', n_prior_components=10,
                    pseudoinputs_data_init=False)
model.train(max_epochs=20,
           plan_kwargs={'loss_weights':{
               'reconstruction_mixup_weight':0,
               'reconstruction_cycle_weight':0,
               'kl_cycle_weight':0,
               'z_distance_cycle_weight':0,
               'translation_corr_weight':0,
               'z_contrastive_weight':0,
               
           }})

# %%
# Plot all loses
losses=[k for k in model.trainer.logger.history.keys() 
        if '_step' not in k and '_epoch' not in k]
fig,axs=plt.subplots(2,len(losses),figsize=(len(losses)*3,4))
for ax_i,l in enumerate(losses):
    axs[0,ax_i].plot(
        model.trainer.logger.history[l].index,
        model.trainer.logger.history[l][l])
    axs[0,ax_i].set_title(l)
    axs[1,ax_i].plot(
        model.trainer.logger.history[l].index[20:],
        model.trainer.logger.history[l][l][20:])
fig.tight_layout()

# %%
embed = model.embed(
        adata=adata_training,
        indices=None,
        batch_size=None,
        as_numpy=True)

# %%
embed=sc.AnnData(embed,obs=adata_training.obs)
embed.obs['species']=embed.obs.system.map({0:'mm',1:'hs'})

# %%
sc.pp.neighbors(embed, use_rep='X')
sc.tl.umap(embed)

# %%
rcParams['figure.figsize']=(8,8)
np.random.seed(0)
random_indices=np.random.permutation(list(range(embed.shape[0])))
sc.pl.umap(embed[random_indices,:],
           color=['cell_type_final', 'study_sample',  'species'],s=10, wspace = 0.5)

# %%
rcParams['figure.figsize']=(5,3)
_=plt.violinplot(embed.to_df())

# %%
ct_var={}
ct_entropy={}
for group,cells in embed.obs.groupby(['species']):
    embed_sub=embed[cells.index,:]
    # Minmax scale the embed comonents so that perceived var is not affected by range
    x=pd.DataFrame(minmax_scale(embed_sub.X),index=embed_sub.obs_names)
    #var_idx=x.var().sort_values(ascending=False).index
    x['cell_type']=embed_sub.obs['cell_type_final']
    x=x.groupby('cell_type').mean()
    #x=x[var_idx]
    g=sb.clustermap(x,cmap='cividis',figsize=(7,5),
                    col_cluster=False,row_cluster=False
                   )
    name='_'.join([group] if isinstance(group,str) else group)
    g.fig.suptitle(name)
    ct_var[name]=x.var()
    ct_entropy[name]=entropy(x)
ct_var=pd.DataFrame(ct_var)
ct_entropy=pd.DataFrame(ct_entropy)
fig,ax=plt.subplots(1,2,figsize=(8,3))
ct_var.boxplot(ax=ax[0])
ax[0].set_title('var over cts')
ct_entropy.boxplot(ax=ax[1])
ax[1].set_title('entropy over cts')

# %% [markdown]
# ### cVAE - VampPrior (n_components=100) with data init

# %%
model = XXJointModel(adata=adata_training, prior='vamp', 
                     n_prior_components=100,
                     pseudoinputs_data_init=True)
model.train(max_epochs=20,
            check_val_every_n_epoch=1,
           plan_kwargs={'loss_weights':{
               'reconstruction_mixup_weight':0,
               'reconstruction_cycle_weight':0,
               'kl_cycle_weight':0,
               'z_distance_cycle_weight':0,
               'translation_corr_weight':0,
               'z_contrastive_weight':0,
               
           }})

# %%
# Plot all loses
losses=[k for k in model.trainer.logger.history.keys() 
        if '_step' not in k and '_epoch' not in k and 'validation' not in k]
fig,axs=plt.subplots(2,len(losses),figsize=(len(losses)*3,4))
for ax_i,l_train in enumerate(losses):
    l_val=l_train.replace('_train','_validation')
    l_name=l_train.replace('_train','')
    for l,c in [(l_train,'tab:blue'),(l_val,'tab:orange')]:
        axs[0,ax_i].plot(
            model.trainer.logger.history[l].index,
            model.trainer.logger.history[l][l],c=c)
        axs[0,ax_i].set_title(l_name)
        axs[1,ax_i].plot(
            model.trainer.logger.history[l].index[10:],
            model.trainer.logger.history[l][l][10:],c=c)
fig.tight_layout()

# %%
embed = model.embed(
        adata=adata_training,
        indices=None,
        batch_size=None,
        as_numpy=True)

# %%
embed=sc.AnnData(embed,obs=adata_training.obs)
embed.obs['species']=embed.obs.system.map({0:'mm',1:'hs'})

# %%
rcParams['figure.figsize']=(5,3)
_=plt.violinplot(embed.to_df())

# %% [markdown]
# C: Problem of inactive latent dimensions is still not resolved

# %%
ct_var={}
ct_entropy={}
for group,cells in embed.obs.groupby(['species']):
    embed_sub=embed[cells.index,:]
    # Minmax scale the embed comonents so that perceived var is not affected by range
    x=pd.DataFrame(minmax_scale(embed_sub.X),index=embed_sub.obs_names)
    #var_idx=x.var().sort_values(ascending=False).index
    x['cell_type']=embed_sub.obs['cell_type_final']
    x=x.groupby('cell_type').mean()
    #x=x[var_idx]
    g=sb.clustermap(x,cmap='cividis',figsize=(7,5),
                    col_cluster=False,row_cluster=False
                   )
    name='_'.join([group] if isinstance(group,str) else group)
    g.fig.suptitle(name)
    ct_var[name]=x.var()
    ct_entropy[name]=entropy(x)
ct_var=pd.DataFrame(ct_var)
ct_entropy=pd.DataFrame(ct_entropy)
fig,ax=plt.subplots(1,2,figsize=(8,3))
ct_var.boxplot(ax=ax[0])
ax[0].set_title('var over cts')
ct_entropy.boxplot(ax=ax[1])
ax[1].set_title('entropy over cts')

# %% [markdown]
# Combine pseudoinput and data subset embedding to compare the two

# %%
# Encode pseudoinputs
embed_pseudoinputs=model.module.encoder(x=model.module.prior.u,cov=model.module.prior.u_cov
                                       )['y'].detach().cpu().numpy()

# %%
embed_pseudoinputs=sc.AnnData(embed_pseudoinputs)

# %%
# Pseudoinput and data sub embedding
random_indices=np.random.permutation(list(range(embed.shape[0])))[:10000]
embed_sub_pin = sc.concat([embed[random_indices,:],embed_pseudoinputs], 
                          merge='unique', join='outer')
embed_sub_pin.obs['input_type']=['expr']*len(random_indices)+\
        ['pseudo']*embed_pseudoinputs.shape[0]

# %%
sc.pp.neighbors(embed_sub_pin, use_rep='X')
sc.tl.umap(embed_sub_pin)

# %%
rcParams['figure.figsize']=(5,5)
sc.pl.umap(embed_sub_pin,color=['input_type','cell_type_final', 'study_sample',  'species'],
           s=40,wspace=0.5)

# %% [markdown]
# C: Embedding looks better after adding pseudoinputs initialised based on data

# %% [markdown]
# Embed of all reall cells

# %%
sc.pp.neighbors(embed, use_rep='X')
sc.tl.umap(embed)

# %%
rcParams['figure.figsize']=(8,8)
np.random.seed(0)
random_indices=np.random.permutation(list(range(embed.shape[0])))
sc.pl.umap(embed[random_indices,:],
           color=['cell_type_final', 'study_sample',  'species'],s=10, wspace = 0.5)


# %% [markdown]
# Make random samples from prior and embed them in UMAP

# %%
def sample_prior(prior, batch_size):
    # mu, lof_var
    means, logvars = prior.get_params()
    means=means.detach().cpu()
    logvars=logvars.detach().cpu()

    # mixing probabilities
    w = torch.nn.functional.softmax(prior.w, dim=0)
    w = w.squeeze().detach().cpu()

    # pick components
    indexes = torch.multinomial(w, batch_size, replacement=True)

    # means and logvars
    eps = torch.randn(batch_size, means.shape[1])
    for i in range(batch_size):
        indx = indexes[i]
        if i == 0:
            z = means[[indx]] + eps[[i]] * torch.exp(logvars[[indx]])
        else:
            z = torch.cat((z, means[[indx]] + eps[[i]] * torch.exp(logvars[[indx]])), 0)
    return z.numpy()


# %%
prior_samples=sc.AnnData(sample_prior(model.module.prior,1000))

# %%
sc.pp.neighbors(prior_samples, use_rep='X')
sc.tl.umap(prior_samples)

# %%
rcParams['figure.figsize']=(5,5)
sc.pl.umap(prior_samples,s=40,wspace=0.5)

# %% [markdown]
# ## Pancreas + tabula

# %%
adata=sc.read(path_data+'combined_tabula_orthologues.h5ad')

# %%
adata_training = XXJointModel.setup_anndata(
    adata=adata,
    system_key='system',
    #group_key='cell_type',
    categorical_covariate_keys=['batch'],
)

# %%
ct_tabula_matched=pd.read_table(path_tabula+'cell_type_mapping.tsv',index_col=0)

# %% [markdown]
# ### cVAE - VampPrior (n_components=100)

# %%
model = XXJointModel(adata=adata_training, prior='vamp', n_prior_components=100,
                    pseudoinputs_data_init=False)
model.train(max_epochs=20,
            check_val_every_n_epoch=1,
           plan_kwargs={'loss_weights':{
               'reconstruction_mixup_weight':0,
               'reconstruction_cycle_weight':0,
               'kl_cycle_weight':0,
               'z_distance_cycle_weight':0,
               'translation_corr_weight':0,
               'z_contrastive_weight':0,
               
           }})

# %%
# Plot all loses
losses=[k for k in model.trainer.logger.history.keys() 
        if '_step' not in k and '_epoch' not in k and 'validation' not in k]
fig,axs=plt.subplots(2,len(losses),figsize=(len(losses)*3,4))
for ax_i,l_train in enumerate(losses):
    l_val=l_train.replace('_train','_validation')
    l_name=l_train.replace('_train','')
    for l,c in [(l_train,'tab:blue'),(l_val,'tab:orange')]:
        axs[0,ax_i].plot(
            model.trainer.logger.history[l].index,
            model.trainer.logger.history[l][l],c=c)
        axs[0,ax_i].set_title(l_name)
        axs[1,ax_i].plot(
            model.trainer.logger.history[l].index[10:],
            model.trainer.logger.history[l][l][10:],c=c)
fig.tight_layout()

# %% [markdown]
# C: there seems to be a problem in training - reconstruction and KL of val are both bad and KL even unstable.

# %%
embed = model.embed(
        adata=adata_training,
        indices=None,
        batch_size=None,
        as_numpy=True)

# %%
embed=sc.AnnData(embed,obs=adata_training.obs)
embed.obs['species']=embed.obs.system.map({0:'mm',1:'hs'})

# %%
rcParams['figure.figsize']=(5,3)
_=plt.violinplot(embed.to_df())

# %%
ct_var={}
ct_entropy={}
for group,cells in embed.obs.groupby(['species']):
    embed_sub=embed[cells.index,:]
    # Minmax scale the embed comonents so that perceived var is not affected by range
    x=pd.DataFrame(minmax_scale(embed_sub.X),index=embed_sub.obs_names)
    #var_idx=x.var().sort_values(ascending=False).index
    x['cell_type']=embed_sub.obs['cell_type']
    x=x.groupby('cell_type').mean()
    #x=x[var_idx]
    g=sb.clustermap(x,cmap='cividis',figsize=(7,5),
                    col_cluster=False,row_cluster=False
                   )
    name='_'.join([group] if isinstance(group,str) else group)
    g.fig.suptitle(name)
    ct_var[name]=x.var()
    ct_entropy[name]=entropy(x)
ct_var=pd.DataFrame(ct_var)
ct_entropy=pd.DataFrame(ct_entropy)
fig,ax=plt.subplots(1,2,figsize=(8,3))
ct_var.boxplot(ax=ax[0])
ax[0].set_title('var over cts')
ct_entropy.boxplot(ax=ax[1])
ax[1].set_title('entropy over cts')

# %%
sc.pp.neighbors(embed, use_rep='X')
sc.tl.umap(embed)

# %%
rcParams['figure.figsize']=(8,8)
sc.pl.umap(embed,color=['species','cell_type','dataset','batch'],s=10,wspace=0.5)

# %%
embed_sub=embed[embed.obs.dataset=='pancreas',:].copy()
embed_sub.obs['cell_type']=embed_sub.obs['cell_type'].astype(str)
if 'cell_type_colors' in embed_sub.uns:
    del embed_sub.uns['cell_type_colors']
sc.pl.umap(embed_sub,
           color=['species','cell_type','batch'],s=10,wspace=0.5)
del embed_sub

# %%
embed_sub=embed[embed.obs.dataset=='tabula',:].copy()
sc.pl.umap(embed_sub,
           color=['system'],s=10,wspace=0.5)
del embed_sub

embed_sub=embed[(embed.obs.dataset=='tabula').values & 
                (embed.obs.system==1).values,:].copy()
embed_sub.obs['cell_type_matched']=ct_tabula_matched.loc[embed_sub.obs_names,'cell_type']
fig,ax=plt.subplots(figsize=(8,8))
sc.pl.umap(embed,ax=ax,show=False)
sc.pl.umap(embed_sub,
           color=['cell_type_matched'],s=10,wspace=0.5,ax=ax)

del embed_sub
embed_sub=embed[(embed.obs.dataset=='tabula').values & 
                (embed.obs.system==0).values,:].copy()
embed_sub.obs['cell_type_matched']=ct_tabula_matched.loc[embed_sub.obs_names,'cell_type']
fig,ax=plt.subplots(figsize=(8,8))
sc.pl.umap(embed,ax=ax,show=False)
sc.pl.umap(embed_sub,
           color=['cell_type_matched'],s=10,wspace=0.5,ax=ax)
del embed_sub

# %% [markdown]
# C: The mebedding does not look good.

# %% [markdown]
# ##### Embed of pseudoinputs

# %%
# Encode pseudoinputs
embed_pseudoinputs=model.module.encoder(x=model.module.prior.u,cov=model.module.prior.u_cov
                                       )['y'].detach().cpu().numpy()

# %%
embed_pseudoinputs=sc.AnnData(embed_pseudoinputs)

# %%
rcParams['figure.figsize']=(5,3)
_=plt.violinplot(embed_pseudoinputs.to_df())

# %% [markdown]
# Combine pseudoinput and data subset embedding to compare the two

# %%
# Pseudoinput and data sub embedding
random_indices=np.random.permutation(list(range(embed.shape[0])))[:10000]
embed_sub_pin = sc.concat([embed[random_indices,:],embed_pseudoinputs], 
                          merge='unique', join='outer')
embed_sub_pin.obs['input_type']=['expr']*len(random_indices)+\
        ['pseudo']*embed_pseudoinputs.shape[0]

# %%
sc.pp.neighbors(embed_sub_pin, use_rep='X')
sc.tl.umap(embed_sub_pin)

# %%
rcParams['figure.figsize']=(5,5)
sc.pl.umap(embed_sub_pin,color=['input_type'],s=20,wspace=0.5)

# %% [markdown]
# C: The pseudo-inputs did not work well. They are all at the same place. Maybe problem with initialisation.

# %% [markdown]
# ###### Compare random selection of samples (cells) vs per k-means cluster selection

# %%
# Kmens on adata
kmeans=MiniBatchKMeans(n_clusters=100,random_state=0)
clusters=kmeans.fit_predict(adata.X)

# %% [markdown]
# Compare kmeans clusters and cell types

# %%
sb.clustermap(pd.crosstab(adata.obs['cell_type'],clusters,normalize='columns'))

# %%
# N all cell types
adata.obs.cell_type.nunique()

# %% [markdown]
# How many cts are selected in pseudoinputs if we use random sampling or k-means based sampling

# %%
# Cells selected based on kmeans
# N cell types and n cells per ct, for 10 re-samples
fix,ax=plt.subplots()
for i in range(10):
    cells=pd.DataFrame({'cluster':clusters,'cell_type':adata.obs['cell_type']}
                      ).groupby('cluster').apply(
                        lambda x: x.sample(1)).reset_index(drop=True)
    print('N cell types:',cells.cell_type.nunique())
    ax.hist(cells.cell_type.value_counts(),alpha=0.2)

# %%
# Randomly select cells
# N cell types and n cells per ct, for 10 re-samples
fix,ax=plt.subplots()
for i in range(10):
    random_indices=np.random.permutation(list(range(embed.shape[0])))[:100]
    cells=adata[random_indices,:].obs
    print('N cell types:',cells.cell_type.nunique())
    ax.hist(cells.cell_type.value_counts(),alpha=0.2)

# %% [markdown]
# C: Using kmeans seems to enable selection of cells from more cell types as pseudoinputs.

# %% [markdown]
# ### cVAE - VampPrior (n_components=100) with data based init

# %%
model = XXJointModel(adata=adata_training, prior='vamp', n_prior_components=100,
                    pseudoinputs_data_init=True)
model.train(max_epochs=20,
            check_val_every_n_epoch=1,
           plan_kwargs={'loss_weights':{
               'reconstruction_mixup_weight':0,
               'reconstruction_cycle_weight':0,
               'kl_cycle_weight':0,
               'z_distance_cycle_weight':0,
               'translation_corr_weight':0,
               'z_contrastive_weight':0,
               
           }})

# %%
# Plot all loses
losses=[k for k in model.trainer.logger.history.keys() 
        if '_step' not in k and '_epoch' not in k and 'validation' not in k]
fig,axs=plt.subplots(2,len(losses),figsize=(len(losses)*3,4))
for ax_i,l_train in enumerate(losses):
    l_val=l_train.replace('_train','_validation')
    l_name=l_train.replace('_train','')
    for l,c in [(l_train,'tab:blue'),(l_val,'tab:orange')]:
        axs[0,ax_i].plot(
            model.trainer.logger.history[l].index,
            model.trainer.logger.history[l][l],c=c)
        axs[0,ax_i].set_title(l_name)
        axs[1,ax_i].plot(
            model.trainer.logger.history[l].index[10:],
            model.trainer.logger.history[l][l][10:],c=c)
fig.tight_layout()

# %%
embed = model.embed(
        adata=adata_training,
        indices=None,
        batch_size=None,
        as_numpy=True)

# %%
embed=sc.AnnData(embed,obs=adata_training.obs)
embed.obs['species']=embed.obs.system.map({0:'mm',1:'hs'})

# %%
rcParams['figure.figsize']=(5,3)
_=plt.violinplot(embed.to_df())

# %% [markdown]
# C: Problem of inactive latent dimensions is still not resolved

# %%
ct_var={}
ct_entropy={}
for group,cells in embed.obs.groupby(['species']):
    embed_sub=embed[cells.index,:]
    # Minmax scale the embed comonents so that perceived var is not affected by range
    x=pd.DataFrame(minmax_scale(embed_sub.X),index=embed_sub.obs_names)
    #var_idx=x.var().sort_values(ascending=False).index
    x['cell_type']=embed_sub.obs['cell_type']
    x=x.groupby('cell_type').mean()
    #x=x[var_idx]
    g=sb.clustermap(x,cmap='cividis',figsize=(7,5),
                    col_cluster=False,row_cluster=False
                   )
    name='_'.join([group] if isinstance(group,str) else group)
    g.fig.suptitle(name)
    ct_var[name]=x.var()
    ct_entropy[name]=entropy(x)
ct_var=pd.DataFrame(ct_var)
ct_entropy=pd.DataFrame(ct_entropy)
fig,ax=plt.subplots(1,2,figsize=(8,3))
ct_var.boxplot(ax=ax[0])
ax[0].set_title('var over cts')
ct_entropy.boxplot(ax=ax[1])
ax[1].set_title('entropy over cts')

# %% [markdown]
# Combine pseudoinput and data subset embedding to compare the two

# %%
# Encode pseudoinputs
embed_pseudoinputs=model.module.encoder(x=model.module.prior.u,cov=model.module.prior.u_cov
                                       )['y'].detach().cpu().numpy()

# %%
embed_pseudoinputs=sc.AnnData(embed_pseudoinputs)

# %%
# Pseudoinput and data sub embedding
random_indices=np.random.permutation(list(range(embed.shape[0])))[:10000]
embed_sub_pin = sc.concat([embed[random_indices,:],embed_pseudoinputs], 
                          merge='unique', join='outer')
embed_sub_pin.obs['input_type']=['expr']*len(random_indices)+\
        ['pseudo']*embed_pseudoinputs.shape[0]

# %%
sc.pp.neighbors(embed_sub_pin, use_rep='X')
sc.tl.umap(embed_sub_pin)

# %%
rcParams['figure.figsize']=(5,5)
sc.pl.umap(embed_sub_pin,color=['input_type'],
           s=40,wspace=0.5)

# %%
embed_sub=embed_sub_pin[embed_sub_pin.obs.dataset=='pancreas',:].copy()
embed_sub.obs['cell_type']=embed_sub.obs['cell_type'].astype(str)
if 'cell_type_colors' in embed_sub.uns:
    del embed_sub.uns['cell_type_colors']
sc.pl.umap(embed_sub,
           color=['species','cell_type','batch'],s=10,wspace=0.5)
del embed_sub

# %%
embed_sub=embed_sub_pin[embed_sub_pin.obs.dataset=='tabula',:].copy()
sc.pl.umap(embed_sub,
           color=['system'],s=10,wspace=0.5)
del embed_sub

embed_sub=embed_sub_pin[(embed_sub_pin.obs.dataset=='tabula').values & 
                (embed_sub_pin.obs.system==1).values,:].copy()
embed_sub.obs['cell_type_matched']=ct_tabula_matched.loc[embed_sub.obs_names,'cell_type']
fig,ax=plt.subplots(figsize=(4,4))
#sc.pl.umap(embed,ax=ax,show=False)
sc.pl.umap(embed_sub,
           color=['cell_type_matched'],s=10,wspace=0.5,ax=ax)

del embed_sub
embed_sub=embed_sub_pin[(embed_sub_pin.obs.dataset=='tabula').values & 
                (embed_sub_pin.obs.system==0).values,:].copy()
embed_sub.obs['cell_type_matched']=ct_tabula_matched.loc[embed_sub.obs_names,'cell_type']
fig,ax=plt.subplots(figsize=(4,4))
#sc.pl.umap(embed,ax=ax,show=False)
sc.pl.umap(embed_sub,
           color=['cell_type_matched'],s=10,wspace=0.5,ax=ax)
del embed_sub

# %% [markdown]
# ### VAMP - change of pseudoinputs over training

# %%
pis=[]
model = XXJointModel(adata=adata_training, prior='vamp', n_prior_components=100,
                    pseudoinputs_data_init=True)
pis.append([model.module.prior.u.detach().cpu().numpy(),
          model.module.prior.u_cov.detach().cpu().numpy()])
# Train for 1 epoch and see how PIs migrate
for i in range(20):
    model.train(max_epochs=1,
            check_val_every_n_epoch=1,
           plan_kwargs={'loss_weights':{
               'reconstruction_mixup_weight':0,
               'reconstruction_cycle_weight':0,
               'kl_cycle_weight':0,
               'z_distance_cycle_weight':0,
               'translation_corr_weight':0,
               'z_contrastive_weight':0,
               
           }})
    pis.append([model.module.prior.u.detach().cpu().numpy(),
          model.module.prior.u_cov.detach().cpu().numpy()])

# %%
# PIs are on level of data - thus they must be encoded to get them to latent space
pis_z=[model.module.encoder(x=torch.tensor(x,device=model.module.device), 
                            cov=torch.tensor(cov,device=model.module.device)
                           )['y_m'].detach().cpu().numpy() 
       for x,cov in pis]

# %%
embed_pis=sc.AnnData(np.concatenate(pis_z),
           obs=pd.DataFrame({
               'epoch':[ei for e,data in enumerate(pis_z) for ei in [e]*data.shape[0]]}))

# %%
np.random.seed(0)
random_indices=np.random.permutation(list(range(embed.shape[0])))[:10000]
embed_sub = model.embed(
        adata=adata_training[random_indices,:],
        indices=None,
        batch_size=None,
        as_numpy=True)
embed_sub=sc.AnnData(embed_sub,obs=adata_training[random_indices,:].obs)
embed_sub.obs['species']=embed_sub.obs.system.map({0:'mm',1:'hs'})

# %%
# Pseudoinput and data sub joint embedding
embed_sub_pin = sc.concat([embed_sub,embed_pis], 
                          merge='unique', join='outer')
embed_sub_pin.obs['input_type']=['expr']*embed_sub.shape[0]+\
        ['pseudo']*embed_pis.shape[0]

# %%
sc.pp.neighbors(embed_sub_pin, use_rep='X')
sc.tl.umap(embed_sub_pin)

# %%
rcParams['figure.figsize']=(10,10)
sc.pl.umap(embed_sub_pin,color='epoch',s=20)

# %% [markdown]
# C: It seems that pins move during training, mainly away from real cells.

# %% [markdown]
# Figure out how similar are pins to real cells accross epochs.

# %%
# Compute similarity of pins to real cells
sim_cell=pd.DataFrame({
    'mean_sim':np.array(embed_sub_pin.obsp['connectivities'
                  ][~embed_sub_pin.obs.epoch.isna(),:][:,embed_sub_pin.obs.epoch.isna()].\
                  mean(axis=1)).ravel(),
    'max_sim':np.array(embed_sub_pin.obsp['connectivities'
                  ][~embed_sub_pin.obs.epoch.isna(),:][:,embed_sub_pin.obs.epoch.isna()].\
                  max(axis=1).todense()).ravel(),
    'epoch':embed_sub_pin[~embed_sub_pin.obs.epoch.isna(),:].obs['epoch'].astype(int)})

# %%
rcParams['figure.figsize']=(7,3)
sb.boxplot(x='epoch',y='mean_sim',data=sim_cell,whis=10)

# %%
rcParams['figure.figsize']=(7,3)
sb.boxplot(x='epoch',y='max_sim',data=sim_cell,whis=10)

# %% [markdown]
# C: Indeed, it seems that pseudoinputs travel further from closest cells during learning. Maybe this helps to make them globally relevant? Unexpected still.

# %%
