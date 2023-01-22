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

# %%
adata_hs=sc.read(path_data+'human_orthologues.h5ad')
adata_mm=sc.read(path_data+'mouse_orthologues.h5ad')

# %% [markdown]
# ## PP

# %%
# Prepare adatas for concat and concat
# Human
adata_hs_sub=adata_hs.copy()
adata_hs_sub.obs=adata_hs_sub.obs[['cell_type_final','study_sample']]
adata_hs_sub.obs['system']=1
adata_hs_sub.var['EID']=adata_hs_sub.var_names
adata_hs_sub.var_names=adata_mm.var_names
del adata_hs_sub.obsm
# Mouse
adata_mm_sub=adata_mm.copy()
adata_mm_sub.obs=adata_mm_sub.obs[['cell_type_final','study_sample']]
adata_mm_sub.obs['system']=0
del adata_mm_sub.obsm
# Concat
adata=sc.concat([adata_mm_sub,adata_hs_sub])

del adata_mm_sub
del adata_hs_sub
gc.collect()

display(adata)

# %% [markdown]
# ## Train model

# %% [markdown]
# ### cVAE
# No cycle losses - normal cVAE that also has the system in the covariates.

# %%
adata_training = XXJointModel.setup_anndata(
    adata=adata,
    system_key='system',
    #class_key='cell_type_final',
    categorical_covariate_keys=['study_sample'],
)
model = XXJointModel(adata=adata_training)

# %%
model.train(max_epochs=20,
           plan_kwargs={'loss_weights':{
               'reconstruction_cycle_weight':0,
               'kl_cycle_weight':0,
               'z_distance_cycle_weight':0
           }})

# %%
losses=['reconstruction_loss_train', 'reconstruction_loss_cycle_train', 
        'kl_local_train', 'kl_local_cycle_train','z_distance_cycle_train']
fig,axs=plt.subplots(1,len(losses),figsize=(len(losses)*3,2))
for ax,l in zip(axs,losses):
    ax.plot(
        model.trainer.logger.history[l].index,
        model.trainer.logger.history[l][l])
    ax.set_title(l)

# %% [markdown]
# C: The losses are computed but the weight was set to 0

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
sc.pl.umap(embed,color=['species','cell_type_final','study_sample'],s=10,wspace=0.5)

# %% [markdown]
# ### cVAE + cycle losses (reconstruction, kl, z similarity)

# %%
adata_training = XXJointModel.setup_anndata(
    adata=adata,
    system_key='system',
    #class_key='cell_type_final',
    categorical_covariate_keys=['study_sample'],
)
model = XXJointModel(adata=adata_training)

# %%
model.train(max_epochs=20)

# %%
losses=['reconstruction_loss_train', 'reconstruction_loss_cycle_train', 
        'kl_local_train', 'kl_local_cycle_train','z_distance_cycle_train']
fig,axs=plt.subplots(1,len(losses),figsize=(len(losses)*3,2))
for ax,l in zip(axs,losses):
    ax.plot(
        model.trainer.logger.history[l].index,
        model.trainer.logger.history[l][l])
    ax.set_title(l)

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
sc.pl.umap(embed,color=['species','cell_type_final','study_sample'],s=10,wspace=0.5)

# %% [markdown]
# C: Compared to above it seems that adding the cycle weight improves integration. The main difference seems to be in the z_distance_cycle (different losss shape).

# %% [markdown]
# ### cVAE + cycle (without z distance)
# Test if z distance is the main component leading to better integration.

# %%
adata_training = XXJointModel.setup_anndata(
    adata=adata,
    system_key='system',
    #class_key='cell_type_final',
    categorical_covariate_keys=['study_sample'],
)
model = XXJointModel(adata=adata_training)

# %%
model.train(max_epochs=20,
           plan_kwargs={'loss_weights':{
               'z_distance_cycle_weight':0
           }})

# %%
losses=['reconstruction_loss_train', 'reconstruction_loss_cycle_train', 
        'kl_local_train', 'kl_local_cycle_train','z_distance_cycle_train']
fig,axs=plt.subplots(1,len(losses),figsize=(len(losses)*3,2))
for ax,l in zip(axs,losses):
    ax.plot(
        model.trainer.logger.history[l].index,
        model.trainer.logger.history[l][l])
    ax.set_title(l)

# %% [markdown]
# C: Maybe other cycle components also help with keeping down the z_distance_cycle - would need to run multiple times to check if the reduction in z_distance_cycle at the end is random or not.

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
sc.pl.umap(embed,color=['species','cell_type_final','study_sample'],s=10,wspace=0.5)

# %% [markdown]
# ### cVAE + cycle (with stronger z distance weight)
# Test if z distance is the main component leading to better integration.

# %%
adata_training = XXJointModel.setup_anndata(
    adata=adata,
    system_key='system',
    #class_key='cell_type_final',
    categorical_covariate_keys=['study_sample'],
)
model = XXJointModel(adata=adata_training)

# %%
model.train(max_epochs=20,
           plan_kwargs={'loss_weights':{
               'z_distance_cycle_weight':10
           }})

# %%
losses=['reconstruction_loss_train', 'reconstruction_loss_cycle_train', 
        'kl_local_train', 'kl_local_cycle_train','z_distance_cycle_train']
fig,axs=plt.subplots(1,len(losses),figsize=(len(losses)*3,2))
for ax,l in zip(axs,losses):
    ax.plot(
        model.trainer.logger.history[l].index,
        model.trainer.logger.history[l][l])
    ax.set_title(l)

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
sc.pl.umap(embed,color=['species','cell_type_final','study_sample'],s=10,wspace=0.5)

# %% [markdown]
# C: It seems that stronger z distance may prevent improvement of the kl loss. - Works similarly as reconstruction loss in that manner.

# %% [markdown]
# ### cVAE + cycle (without cycle kl)
# Test how ofther cycle components affect integration.

# %%
adata_training = XXJointModel.setup_anndata(
    adata=adata,
    system_key='system',
    #class_key='cell_type_final',
    categorical_covariate_keys=['study_sample'],
)
model = XXJointModel(adata=adata_training)

# %%
model.train(max_epochs=20,
           plan_kwargs={'loss_weights':{
               'kl_cycle_weight':0
           }})

# %%
losses=['reconstruction_loss_train', 'reconstruction_loss_cycle_train', 
        'kl_local_train', 'kl_local_cycle_train','z_distance_cycle_train']
fig,axs=plt.subplots(1,len(losses),figsize=(len(losses)*3,2))
for ax,l in zip(axs,losses):
    ax.plot(
        model.trainer.logger.history[l].index,
        model.trainer.logger.history[l][l])
    ax.set_title(l)

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
sc.pl.umap(embed,color=['species','cell_type_final','study_sample'],s=10,wspace=0.5)

# %% [markdown]
# C: Removing cycle kl reduces integration accross systems. But the KL in general pushes cells closer together.

# %% [markdown]
# ### cVAE + cycle (without cycle reconstruction)
# Test how ofther cycle components affect integration.

# %%
adata_training = XXJointModel.setup_anndata(
    adata=adata,
    system_key='system',
    #class_key='cell_type_final',
    categorical_covariate_keys=['study_sample'],
)
model = XXJointModel(adata=adata_training)

# %%
model.train(max_epochs=20,
           plan_kwargs={'loss_weights':{
               'reconstruction_cycle_weight':0
           }})

# %%
losses=['reconstruction_loss_train', 'reconstruction_loss_cycle_train', 
        'kl_local_train', 'kl_local_cycle_train','z_distance_cycle_train']
fig,axs=plt.subplots(1,len(losses),figsize=(len(losses)*3,2))
for ax,l in zip(axs,losses):
    ax.plot(
        model.trainer.logger.history[l].index,
        model.trainer.logger.history[l][l])
    ax.set_title(l)

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
sc.pl.umap(embed,color=['species','cell_type_final','study_sample'],s=10,wspace=0.5)

# %% [markdown]
# C: Cycle reconstruction also affects integrartion efficiency. This allows kl in cycle to reduce, though.

# %% [markdown]
# C: Losses affect each other - removing other cycle losses also increases z_distance_cycle.

# %% [markdown]
# C: If dont do reconstruction in cycle then kl drops - dont need to preserve info in z. 

# %% [markdown]
# ### cVAE with higher kl weight
# Test how just increasing kl weight (as we add extra kl from cycle) afects integration.

# %%
adata_training = XXJointModel.setup_anndata(
    adata=adata,
    system_key='system',
    #class_key='cell_type_final',
    categorical_covariate_keys=['study_sample'],
)
model = XXJointModel(adata=adata_training)

# %%
model.train(max_epochs=20,
           plan_kwargs={'loss_weights':{
               'kl_weight':2,
               'reconstruction_cycle_weight':0,
               'kl_cycle_weight':0,
               'z_distance_cycle_weight':0
           }})

# %%
losses=['reconstruction_loss_train', 'reconstruction_loss_cycle_train', 
        'kl_local_train', 'kl_local_cycle_train','z_distance_cycle_train']
fig,axs=plt.subplots(1,len(losses),figsize=(len(losses)*3,2))
for ax,l in zip(axs,losses):
    ax.plot(
        model.trainer.logger.history[l].index,
        model.trainer.logger.history[l][l])
    ax.set_title(l)

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
sc.pl.umap(embed,color=['species','cell_type_final','study_sample'],s=10,wspace=0.5)

# %% [markdown]
# C: just increasing KL seems to have similar effect of pushing the two systems together (as the cycle) so the cycle does not bring much to their integration.

# %% [markdown]
# ### cVAE with cycle reconstruction
# Test how just adding one cycle loss affects integration.

# %%
adata_training = XXJointModel.setup_anndata(
    adata=adata,
    system_key='system',
    #class_key='cell_type_final',
    categorical_covariate_keys=['study_sample'],
)
model = XXJointModel(adata=adata_training)

# %%
model.train(max_epochs=20,
           plan_kwargs={'loss_weights':{
               'reconstruction_cycle_weight':1,
               'kl_cycle_weight':0,
               'z_distance_cycle_weight':0
           }})

# %%
losses=['reconstruction_loss_train', 'reconstruction_loss_cycle_train', 
        'kl_local_train', 'kl_local_cycle_train','z_distance_cycle_train']
fig,axs=plt.subplots(1,len(losses),figsize=(len(losses)*3,2))
for ax,l in zip(axs,losses):
    ax.plot(
        model.trainer.logger.history[l].index,
        model.trainer.logger.history[l][l])
    ax.set_title(l)

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
sc.pl.umap(embed,color=['species','cell_type_final','study_sample'],s=10,wspace=0.5)

# %% [markdown]
# C: Cycle reconstruction loss does not help the alignment

# %% [markdown]
# ### cVAE with cycle kl
# Test how just adding one cycle loss affects integration.

# %%
adata_training = XXJointModel.setup_anndata(
    adata=adata,
    system_key='system',
    #class_key='cell_type_final',
    categorical_covariate_keys=['study_sample'],
)
model = XXJointModel(adata=adata_training)

# %%
model.train(max_epochs=20,
           plan_kwargs={'loss_weights':{
               'reconstruction_cycle_weight':0,
               'kl_cycle_weight':1,
               'z_distance_cycle_weight':0
           }})

# %%
losses=['reconstruction_loss_train', 'reconstruction_loss_cycle_train', 
        'kl_local_train', 'kl_local_cycle_train','z_distance_cycle_train']
fig,axs=plt.subplots(1,len(losses),figsize=(len(losses)*3,2))
for ax,l in zip(axs,losses):
    ax.plot(
        model.trainer.logger.history[l].index,
        model.trainer.logger.history[l][l])
    ax.set_title(l)

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
sc.pl.umap(embed,color=['species','cell_type_final','study_sample'],s=10,wspace=0.5)

# %% [markdown]
# C: It is odd that adding cycle kl loss alone actually reduces alignment.

# %% [markdown]
# ### cVAE with cycle z distance
# Test how just adding one cycle loss affects integration.

# %%
adata_training = XXJointModel.setup_anndata(
    adata=adata,
    system_key='system',
    #class_key='cell_type_final',
    categorical_covariate_keys=['study_sample'],
)
model = XXJointModel(adata=adata_training)

# %%
model.train(max_epochs=20,
           plan_kwargs={'loss_weights':{
               'reconstruction_cycle_weight':0,
               'kl_cycle_weight':0,
               'z_distance_cycle_weight':1
           }})

# %%
losses=['reconstruction_loss_train', 'reconstruction_loss_cycle_train', 
        'kl_local_train', 'kl_local_cycle_train','z_distance_cycle_train']
fig,axs=plt.subplots(1,len(losses),figsize=(len(losses)*3,2))
for ax,l in zip(axs,losses):
    ax.plot(
        model.trainer.logger.history[l].index,
        model.trainer.logger.history[l][l])
    ax.set_title(l)

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
sc.pl.umap(embed,color=['species','cell_type_final','study_sample'],s=10,wspace=0.5)

# %% [markdown]
# C: Cycle z dist seems to have the biggest effect on its own of the three cycle losses on integration.

# %% [markdown]
# C: Interestingly, it consistently seems that gamma cells are more distinct accross species.

# %% [markdown]
# ## Translation
#
# Two options, easier and harder:
# - Predict when all cells are seen
# - Predict when one ct is missing
#
# Evaluation options:
# - Compare beta cells from mouse, human, predicted from mouse/human etc. to see if prediction from mouse comes closer to human than mouse
# - Harder (stage 2): Look at marker genes shared between species and specific to one species (e.g. Spock2 beta marker only in mouse).

# %% [markdown]
# ### Prediction with all cells used in training

# %%
adata_training = XXJointModel.setup_anndata(
    adata=adata,
    system_key='system',
    #class_key='cell_type_final',
    categorical_covariate_keys=['study_sample'],
)
model = XXJointModel(adata=adata_training)

# %%
model.train(max_epochs=20,
           plan_kwargs={'loss_weights':{
           }})

# %%
embed = model.embed(
        adata=adata_training,
        indices=None,
        batch_size=None,
        as_numpy=True)

# %%
embed=sc.AnnData(embed,obs=adata_training.obs)
embed.obs['species']=embed.obs.system.map({0:'mm',1:'hs'})

# %% [markdown]
# Find samples that are close in the latent space to test translation with different cov and systems. - Use PAGA similarities.

# %%
embed_beta=embed[embed.obs.cell_type_final=='beta',:].copy()
sc.pp.neighbors(embed_beta, use_rep='X')
sc.tl.umap(embed_beta)

# %%
rcParams['figure.figsize']=(8,8)
sc.pl.umap(embed_beta,color=['species','study_sample'],s=30,wspace=0.5)

# %%
# PAGA
sc.tl.paga(embed_beta,groups='study_sample')

# %%
# Get connectivities
groups=embed_beta.obs[embed_beta.uns['paga']['groups']].cat.categories
data=pd.DataFrame(embed_beta.uns['paga']['connectivities'].todense(),
                  index=groups,columns=groups)

# %%
sb.clustermap(data)

# %% [markdown]
# C: At least within mouse samples we see that they group together as expected.

# %%
rcParams['figure.figsize']=(5,5)
groups=['STZ_G1','human_UPR_Donor_10']
sc.pl.umap(embed_beta,color=['study_sample'],groups=groups,
           palette={
               **dict(zip(embed_beta.obs['study_sample'].cat.categories,embed_beta.uns['study_sample_colors'])),
               **dict(zip(groups,['tab:red','tab:blue']))},
           s=30,wspace=0.5)

# %% [markdown]
# C: Could also look at sample similarity with mean z distance

# %% [markdown]
# Make the following examples of cell translation to see similarity between them (on beta cells from STZ_G1 and human_UPR_Donor_10):
# - System translation: all 4 combinations between mm and hs
# - Covariates: mm, hs, all-0

# %%
# Cells to use
sample_mm="STZ_G1"
sample_hs="human_UPR_Donor_10"
cells_mm=adata_training.obs.query(
    'cell_type_final=="beta" & study_sample==@sample_mm').index
cells_hs=adata_training.obs.query(
    'cell_type_final=="beta" & study_sample==@sample_hs').index
print('N mouse cells:',len(cells_mm),'N human cells:',len(cells_hs))

# %%
pred=[]
for system,cells in {'mm':cells_mm,'hs':cells_hs}.items():
    for translation_type, switch in {'self':False,'switch':True}.items():
        for cov_type, cov in {'mm':pd.Series({'study_sample':sample_mm}),
                              'hs':pd.Series({'study_sample':sample_hs}),
                              'zeros': None}.items():
            print(system, translation_type,cov_type)
            n=cells.shape[0]
            pred.append(
                sc.AnnData(
                    model.translate(
                        adata= adata_training[cells,:],
                        switch_system = switch,
                        covariates = cov),
                    obs=pd.DataFrame({
                        'system':[system]*n,
                        'translation':[translation_type]*n,
                        'cov':[cov_type]*n
                    }),
                    var=adata_training.var
                )
            )
# Concat to single adata
pred=sc.concat(pred)

# %%
# Combine metadata to single string
# Which system was predicted
system_map={'mm':False,'hs':True}
pred.obs['system_pred']=pred.obs.system.map(system_map)
pred.obs['system_pred']=pred.obs.apply(
    lambda x: x.system_pred if x.translation=='self' else not x.system_pred, axis=1
).map({v: k for k, v in system_map.items()})
pred.obs['meta']=pred.obs.apply(
    lambda x: '_'.join([x['system'],x['system_pred'],x['cov']]),axis=1)

# %%
# Mean expression per prediction group
x=pred.to_df()
x['meta']=pred.obs['meta']
x=x.groupby('meta').mean()

# Add unpredicted expression
x.loc['mm',:]=adata_training[cells_mm,:].to_df().mean()
x.loc['hs',:]=adata_training[cells_hs,:].to_df().mean()

# %%
# Correlation between cell groups
cor=pd.DataFrame(np.corrcoef(x),index=x.index,columns=x.index)

# %%
cmap={'hs':'tab:pink','mm':'tab:olive','zeros':'gray'}
colors=pd.DataFrame(index=cor.index)
colors['system']=[cmap[i.split('_')[0]] for i in cor.index ]
colors['pred']=[cmap[i.split('_')[1]] if len(i.split('_'))>1 else cmap[i] for i in cor.index]
colors['cov']=[cmap[i.split('_')[2]] if len(i.split('_'))>1 else cmap[i]  for i in cor.index]

# %%
sb.clustermap(cor, row_colors=colors,col_colors=colors)

# %% [markdown]
# C: As expected, one problem with covariates that are confounded with the system is that system can be learned by covariates (mm pred with hs cov is more similar to hs). 
#
# C: It seems that prediction to one system is also similar to original data of that system, especially when using the right set of covariates.
#
# C: Interestingly, mm and hs have correlation of only 0.5. This may be partially due to different covar as predicting without systemn switch with other cov increases similarity.

# %% [markdown]
# Check expression correlation between real-world samples to see how similar they are to each other within and accross species.

# %%
# Mean expression in beta cells per sample
x=adata[adata.obs.cell_type_final=='beta',:].to_df()
x['study_sample']=adata.obs.loc[x.index,'study_sample']
x=x.groupby('study_sample').mean()

# %%
# Correlation between samples
cor=pd.DataFrame(np.corrcoef(x),index=x.index,columns=x.index)

# %%
sb.clustermap(cor)

# %%
display(cor.loc[adata.obs.query('system==1').study_sample.unique(),
        adata.obs.query('system==0').study_sample.unique()])
rcParams['figure.figsize']=(3,3)
plt.boxplot(cor.loc[adata.obs.query('system==1').study_sample.unique(),
        adata.obs.query('system==0').study_sample.unique()].values.ravel())

# %% [markdown]
# C: Correlation between mouse and human samples is ~0.55, but goes as high as 0.65.

# %% [markdown]
# ### Prediction of beta cells with hs beta cells removed from training

# %%
adata_training = XXJointModel.setup_anndata(
    adata=adata[
        (~adata.obs.system.astype(bool)).values | 
        (adata.obs.cell_type_final!='beta').values,
        :],
    system_key='system',
    #class_key='cell_type_final',
    categorical_covariate_keys=['study_sample'],
)
model = XXJointModel(adata=adata_training)

# %%
model.train(max_epochs=20,
           plan_kwargs={'loss_weights':{
           }})

# %% [markdown]
# Make the following examples of cell translation to see similarity between them (on beta cells from STZ_G1 and human_UPR_Donor_10):
# - System translation: mm to mm and to hs
# - Covariates: mm, hs, all-0

# %%
# Cells to use
sample_mm="STZ_G1"
sample_hs="human_UPR_Donor_10"
cells_mm=adata.obs.query(
    'cell_type_final=="beta" & study_sample==@sample_mm').index
cells_hs=adata.obs.query(
    'cell_type_final=="beta" & study_sample==@sample_hs').index
print('N mouse cells:',len(cells_mm),'N human cells:',len(cells_hs))

# %%
pred=[]
for system,cells in {'mm':cells_mm}.items():
    for translation_type, switch in {'self':False,'switch':True}.items():
        for cov_type, cov in {'mm':pd.Series({'study_sample':sample_mm}),
                              'hs':pd.Series({'study_sample':sample_hs}),
                              'zeros': None}.items():
            print(system, translation_type,cov_type)
            n=cells.shape[0]
            pred.append(
                sc.AnnData(
                    model.translate(
                        adata= adata_training[cells,:],
                        switch_system = switch,
                        covariates = cov),
                    obs=pd.DataFrame({
                        'system':[system]*n,
                        'translation':[translation_type]*n,
                        'cov':[cov_type]*n
                    }),
                    var=adata_training.var
                )
            )
# Concat to single adata
pred=sc.concat(pred)

# %%
# Combine metadata to single string
# Which system was predicted
system_map={'mm':False,'hs':True}
pred.obs['system_pred']=pred.obs.system.map(system_map)
pred.obs['system_pred']=pred.obs.apply(
    lambda x: x.system_pred if x.translation=='self' else not x.system_pred, axis=1
).map({v: k for k, v in system_map.items()})
pred.obs['meta']=pred.obs.apply(
    lambda x: '_'.join([x['system'],x['system_pred'],x['cov']]),axis=1)

# %%
# Mean expression per prediction group
x=pred.to_df()
x['meta']=pred.obs['meta']
x=x.groupby('meta').mean()

# Add unpredicted expression
x.loc['mm',:]=adata[cells_mm,:].to_df().mean()
x.loc['hs',:]=adata[cells_hs,:].to_df().mean()

# %%
# Correlation between cell groups
cor=pd.DataFrame(np.corrcoef(x),index=x.index,columns=x.index)

# %%
cmap={'hs':'tab:pink','mm':'tab:olive','zeros':'gray'}
colors=pd.DataFrame(index=cor.index)
colors['system']=[cmap[i.split('_')[0]] for i in cor.index ]
colors['pred']=[cmap[i.split('_')[1]] if len(i.split('_'))>1 else cmap[i] for i in cor.index]
colors['cov']=[cmap[i.split('_')[2]] if len(i.split('_'))>1 else cmap[i]  for i in cor.index]

# %%
sb.clustermap(cor, row_colors=colors,col_colors=colors)

# %% [markdown]
# C: After removing beta cells from human the prediction isnt that good anymore. It seems that using zero or hs cov doesnt matter in this case. Probably the most realistic comparison would be mm->hs_cov0 since we likely wont have in reality the sample with the missing data anyways. 
#
# C: mm->hs_cov0 is still more similar to mouse predictions (except the one with mouse cov) than to human ground truth. Maybe having two separate decoders would actually be useful as the deconder cant learn to cheat and just predict mouse (may help in forcing to learn human gene expression relationships from other human data). The downside is that maybe then decoder doesnt know how the missing ct should look like in the first place.

# %%
print('Corr hs & mm->hs_cov0:',cor.at['hs','mm_hs_zeros'], 
      '\nCor hs & mm:',cor.at['hs','mm'])

# %% [markdown]
# ### Prediction of beta cells with hs beta cells removed from training, without cycle losses

# %%
adata_training = XXJointModel.setup_anndata(
    adata=adata[
        (~adata.obs.system.astype(bool)).values | 
        (adata.obs.cell_type_final!='beta').values,
        :],
    system_key='system',
    #class_key='cell_type_final',
    categorical_covariate_keys=['study_sample'],
)
model = XXJointModel(adata=adata_training)

# %%
model.train(max_epochs=20,
           plan_kwargs={'loss_weights':{
               'reconstruction_cycle_weight':0,
               'kl_cycle_weight':0,
               'z_distance_cycle_weight':0
           }})

# %% [markdown]
# Make the following examples of cell translation to see similarity between them (on beta cells from STZ_G1 and human_UPR_Donor_10):
# - System translation: mm to mm and to hs
# - Covariates: mm, hs, all-0

# %%
# Cells to use
sample_mm="STZ_G1"
sample_hs="human_UPR_Donor_10"
cells_mm=adata.obs.query(
    'cell_type_final=="beta" & study_sample==@sample_mm').index
cells_hs=adata.obs.query(
    'cell_type_final=="beta" & study_sample==@sample_hs').index
print('N mouse cells:',len(cells_mm),'N human cells:',len(cells_hs))

# %%
pred=[]
for system,cells in {'mm':cells_mm}.items():
    for translation_type, switch in {'self':False,'switch':True}.items():
        for cov_type, cov in {'mm':pd.Series({'study_sample':sample_mm}),
                              'hs':pd.Series({'study_sample':sample_hs}),
                              'zeros': None}.items():
            print(system, translation_type,cov_type)
            n=cells.shape[0]
            pred.append(
                sc.AnnData(
                    model.translate(
                        adata= adata_training[cells,:],
                        switch_system = switch,
                        covariates = cov),
                    obs=pd.DataFrame({
                        'system':[system]*n,
                        'translation':[translation_type]*n,
                        'cov':[cov_type]*n
                    }),
                    var=adata_training.var
                )
            )
# Concat to single adata
pred=sc.concat(pred)

# %%
# Combine metadata to single string
# Which system was predicted
system_map={'mm':False,'hs':True}
pred.obs['system_pred']=pred.obs.system.map(system_map)
pred.obs['system_pred']=pred.obs.apply(
    lambda x: x.system_pred if x.translation=='self' else not x.system_pred, axis=1
).map({v: k for k, v in system_map.items()})
pred.obs['meta']=pred.obs.apply(
    lambda x: '_'.join([x['system'],x['system_pred'],x['cov']]),axis=1)

# %%
# Mean expression per prediction group
x=pred.to_df()
x['meta']=pred.obs['meta']
x=x.groupby('meta').mean()

# Add unpredicted expression
x.loc['mm',:]=adata[cells_mm,:].to_df().mean()
x.loc['hs',:]=adata[cells_hs,:].to_df().mean()

# %%
# Correlation between cell groups
cor=pd.DataFrame(np.corrcoef(x),index=x.index,columns=x.index)

# %%
cmap={'hs':'tab:pink','mm':'tab:olive','zeros':'gray'}
colors=pd.DataFrame(index=cor.index)
colors['system']=[cmap[i.split('_')[0]] for i in cor.index ]
colors['pred']=[cmap[i.split('_')[1]] if len(i.split('_'))>1 else cmap[i] for i in cor.index]
colors['cov']=[cmap[i.split('_')[2]] if len(i.split('_'))>1 else cmap[i]  for i in cor.index]

# %%
sb.clustermap(cor, row_colors=colors,col_colors=colors)

# %%
print('Corr hs & mm->hs_cov0:',cor.at['hs','mm_hs_zeros'], 
      '\nCor hs & mm:',cor.at['hs','mm'])

# %% [markdown]
# C: Just blindly adding the cycle does not help. Would need to check how much of the correlation difference is at reandom due to different runs.

# %% [markdown]
# ## Train after adding new functionality
# (Mixup, translation correlation).

# %% [markdown]
# ### cVAE with mixup loss 
# Test how adding mixup loss on z&reconstruction affects the result.

# %%
adata_training = XXJointModel.setup_anndata(
    adata=adata,
    system_key='system',
    #class_key='cell_type_final',
    categorical_covariate_keys=['study_sample'],
)
model = XXJointModel(adata=adata_training, mixup_alpha=0.4)

# %%
model.train(max_epochs=20,
           plan_kwargs={'loss_weights':{
               'reconstruction_mixup_weight':1,
               'reconstruction_cycle_weight':0,
               'kl_cycle_weight':0,
               'z_distance_cycle_weight':0,
               'translation_corr_weight':0,
               
           }})

# %%
losses=['reconstruction_loss_train', 
        'reconstruction_loss_mixup_train', 
        'reconstruction_loss_cycle_train', 
        'kl_local_train', 
        'kl_local_cycle_train',
        'z_distance_cycle_train',
        'translation_corr_train']
fig,axs=plt.subplots(1,len(losses),figsize=(len(losses)*3,2))
for ax,l in zip(axs,losses):
    ax.plot(
        model.trainer.logger.history[l].index,
        model.trainer.logger.history[l][l])
    ax.set_title(l)

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
sc.pl.umap(embed,color=['species','cell_type_final','study_sample'],s=10,wspace=0.5)

# %% [markdown]
# C: Similarly as in normal cVAE the systems are not integrated, but the mixup does not seem to corrupt the representation.

# %% [markdown]
# ### cVAE with translation corr loss
# Test how adding correlation loss on reconstruction of x&y affects the result.

# %%
adata_training = XXJointModel.setup_anndata(
    adata=adata,
    system_key='system',
    #class_key='cell_type_final',
    categorical_covariate_keys=['study_sample'],
)
model = XXJointModel(adata=adata_training)

# %%
model.train(max_epochs=40,
           plan_kwargs={'loss_weights':{
               'reconstruction_mixup_weight':0,
               'reconstruction_cycle_weight':0,
               'kl_cycle_weight':0,
               'z_distance_cycle_weight':0,
               'translation_corr_weight':10,
               
           }})

# %%
losses=['reconstruction_loss_train', 
        'reconstruction_loss_mixup_train', 
        'reconstruction_loss_cycle_train', 
        'kl_local_train', 
        'kl_local_cycle_train',
        'z_distance_cycle_train',
        'translation_corr_train']
fig,axs=plt.subplots(1,len(losses),figsize=(len(losses)*3,2))
for ax,l in zip(axs,losses):
    ax.plot(
        model.trainer.logger.history[l].index,
        model.trainer.logger.history[l][l])
    ax.set_title(l)

# %% [markdown]
# C: May need to train for longer/higher weight to get the effect of correlation loss.

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
sc.pl.umap(embed,color=['species','cell_type_final','study_sample'],s=10,wspace=0.5)

# %% [markdown]
# C: Unsure how this affects representation. Potentially leads to less integration (would need to be tested).

# %% [markdown]
# ### cVAE with two decoders
# Two decoders, for a start without additional losses beyond cVAE.

# %%
adata_training = XXJointModel.setup_anndata(
    adata=adata,
    system_key='system',
    #class_key='cell_type_final',
    categorical_covariate_keys=['study_sample'],
)
model = XXJointModel(adata=adata_training, system_decoders=True)

# %%
model.train(max_epochs=20,
           plan_kwargs={'loss_weights':{
               'reconstruction_mixup_weight':0,
               'reconstruction_cycle_weight':0,
               'kl_cycle_weight':0,
               'z_distance_cycle_weight':0,
               'translation_corr_weight':0,
               
           }})

# %%
losses=['reconstruction_loss_train', 
        'reconstruction_loss_mixup_train', 
        'reconstruction_loss_cycle_train', 
        'kl_local_train', 
        'kl_local_cycle_train',
        'z_distance_cycle_train',
        'translation_corr_train']
fig,axs=plt.subplots(1,len(losses),figsize=(len(losses)*3,2))
for ax,l in zip(axs,losses):
    ax.plot(
        model.trainer.logger.history[l].index,
        model.trainer.logger.history[l][l])
    ax.set_title(l)

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
sc.pl.umap(embed,color=['species','cell_type_final','study_sample'],s=10,wspace=0.5)

# %% [markdown]
# C: Having two decoders completely separates batches in the latent space.

# %% [markdown]
# ### cVAE with two decoders and cycle z distance loss

# %%
adata_training = XXJointModel.setup_anndata(
    adata=adata,
    system_key='system',
    #class_key='cell_type_final',
    categorical_covariate_keys=['study_sample'],
)
model = XXJointModel(adata=adata_training, system_decoders=True)

# %%
model.train(max_epochs=20,
           plan_kwargs={'loss_weights':{
               'reconstruction_mixup_weight':0,
               'reconstruction_cycle_weight':0,
               'kl_cycle_weight':0,
               'z_distance_cycle_weight':1,
               'translation_corr_weight':0,
               
           }})

# %%
losses=['reconstruction_loss_train', 
        'reconstruction_loss_mixup_train', 
        'reconstruction_loss_cycle_train', 
        'kl_local_train', 
        'kl_local_cycle_train',
        'z_distance_cycle_train',
        'translation_corr_train']
fig,axs=plt.subplots(1,len(losses),figsize=(len(losses)*3,2))
for ax,l in zip(axs,losses):
    ax.plot(
        model.trainer.logger.history[l].index,
        model.trainer.logger.history[l][l])
    ax.set_title(l)

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
sc.pl.umap(embed,color=['species','cell_type_final','study_sample'],s=10,wspace=0.5)

# %% [markdown]
# C: Adding z distance loss seems to help to bring together most cell types, but not all - some are still falsely mapped.

# %% [markdown]
# ### cVAE with two decoders and cycle z distance loss and translation corr loss

# %%
adata_training = XXJointModel.setup_anndata(
    adata=adata,
    system_key='system',
    #class_key='cell_type_final',
    categorical_covariate_keys=['study_sample'],
)
model = XXJointModel(adata=adata_training, system_decoders=True)

# %%
model.train(max_epochs=20,
           plan_kwargs={'loss_weights':{
               'reconstruction_mixup_weight':0,
               'reconstruction_cycle_weight':0,
               'kl_cycle_weight':0,
               'z_distance_cycle_weight':1,
               'translation_corr_weight':1,
               
           }})

# %%
losses=['reconstruction_loss_train', 
        'reconstruction_loss_mixup_train', 
        'reconstruction_loss_cycle_train', 
        'kl_local_train', 
        'kl_local_cycle_train',
        'z_distance_cycle_train',
        'translation_corr_train']
fig,axs=plt.subplots(1,len(losses),figsize=(len(losses)*3,2))
for ax,l in zip(axs,losses):
    ax.plot(
        model.trainer.logger.history[l].index,
        model.trainer.logger.history[l][l])
    ax.set_title(l)

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
sc.pl.umap(embed,color=['species','cell_type_final','study_sample'],s=10,wspace=0.5)

# %% [markdown]
# C: Adding z distance loss and translation correlation loss creates shared latent space despite two decoders, however, the cells are still a bit more separated by species.

# %% [markdown]
# ### cVAE with contrastive loss

# %%
adata_training = XXJointModel.setup_anndata(
    adata=adata,
    system_key='system',
    group_key='cell_type_final',
    categorical_covariate_keys=['study_sample'],
)
model = XXJointModel(adata=adata_training)

# %%
model.train(max_epochs=20,
           plan_kwargs={'loss_weights':{
               'reconstruction_mixup_weight':0,
               'reconstruction_cycle_weight':0,
               'kl_cycle_weight':0,
               'z_distance_cycle_weight':0,
               'translation_corr_weight':0,
               'z_contrastive_weight':100,
               
           }})

# %%
losses=['reconstruction_loss_train', 
        'reconstruction_loss_mixup_train', 
        'reconstruction_loss_cycle_train', 
        'kl_local_train', 
        'kl_local_cycle_train',
        'z_distance_cycle_train',
        'translation_corr_train',
        'z_contrastive_train',
        'z_contrastive_pos_train',
        'z_contrastive_neg_train',
       ]
fig,axs=plt.subplots(1,len(losses),figsize=(len(losses)*3,2))
for ax,l in zip(axs,losses):
    ax.plot(
        model.trainer.logger.history[l].index,
        model.trainer.logger.history[l][l])
    ax.set_title(l)

# %% [markdown]
# C: Contrastive loss makes training unstable. Tried to increase weight as first though that contrastive loss is unstable due to small size. But after increasing weight it also made other losses unstable.

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
sc.pl.umap(embed,color=['species','cell_type_final','study_sample'],s=10,wspace=0.5)

# %% [markdown]
# C: Interestingly, the contrastive loss may even push cell types accross systems away. Seems to be most evident in large cts.
#

# %%

# %%
