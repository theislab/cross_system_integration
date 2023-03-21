# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.7.1
#   kernelspec:
#     display_name: csp
#     language: python
#     name: csp
# ---

# %%
import pandas as pd
import numpy as np
import glob
import pickle as pkl
import os
import itertools

from sklearn.preprocessing import minmax_scale

from matplotlib import rcParams
import matplotlib.pyplot as plt
import seaborn as sb
import matplotlib.image as mpimg
import matplotlib.colors as mcolors
import plotly.express as px

# %%
path_eval='/lustre/groups/ml01/workspace/karin.hrovatin/data/cross_species_prediction/eval/'

# %% [markdown]
# ## Integration pancreas_conditions

# %%
path_integration=path_eval+'pancreas_conditions/integration/'

# %%
# Load integration results - params and metrics
res=[]
for run in glob.glob(path_integration+'*/'):
    if os.path.exists(run+'args.pkl') and os.path.exists(run+'scib_metrics.pkl'):
        args=pd.Series(vars(pkl.load(open(run+'args.pkl','rb'))))
        metrics=pd.Series(pkl.load(open(run+'scib_metrics.pkl','rb')))
        data=pd.concat([args,metrics])
        data.name=run.split('/')[-2]
        res.append(data)
res=pd.concat(res,axis=1).T

# %%
#  Param that was optimised
res['param_opt_col']=res.params_opt.replace(
    {'vamp':'n_prior_components',
     'z_distance_cycle_weight_std':'z_distance_cycle_weight',
     'z_distance_cycle_weight_kl':'z_distance_cycle_weight',
     'z_distance_cycle_weight_nll':'z_distance_cycle_weight',
     'scvi':None})
res['param_opt_val']=res.apply(
    lambda x: x[x['param_opt_col']] if x['param_opt_col'] is not None else 0,axis=1)

# %%
res['params_opt']=pd.Categorical(res['params_opt'],sorted(res['params_opt'].unique()), True)

# %%
g=sb.catplot( x='param_opt_val', y="asw_group",  col='params_opt',
           kind="swarm", data=res,sharex=False,
          height=2.5,aspect=1.3,color='k')
for ax in g.axes.ravel():
    ax.axhline(0.52,lw=0.5,c='gray')

# %%
g=sb.catplot( x='param_opt_val', y="ilisi_system",  col='params_opt',
           kind="swarm", data=res,sharex=False,
          height=2.5,aspect=1.3,color='k')
for ax in g.axes.ravel():
    ax.axhline(0.1,lw=0.5,c='gray')
    ax.axhline(0.03,lw=0.5,c='gray')

# %%
rcParams['figure.figsize']=(2,2)
g=sb.scatterplot(x='ilisi_system',y='asw_group',
               hue='params_opt',
               data=res.groupby(['params_opt','param_opt_val'],observed=True
                               ).mean().reset_index(),
                palette='colorblind')
sb.move_legend(g, loc='upper right',bbox_to_anchor=(3, 1))

# %%
# Examples
for params in sorted(res['params_opt'].unique()):
    run_dir=res.query('params_opt==@params & asw_group>0.52'
                     ).sort_values('ilisi_system').index[-1]
    print(params,run_dir)
    display(res.loc[run_dir,['ilisi_system','asw_group']])
    with plt.rc_context({'figure.figsize':(40,10)}):
        path_run=path_integration+run_dir+'/'
        for img_fn in ['umap.png','losses.png']:
            img = mpimg.imread(path_run+img_fn)
            imgplot = plt.imshow(img)
            plt.show()

# %% [markdown]
# C: With z dist cycle NLL loss there seems to be overfitting based on val dataset?

# %%
# Example z_distance_cycle_kl run
for run_dir in res.query('params_opt=="z_distance_cycle_weight_kl" & param_opt_val==2').index:
    print(run_dir)
    display(res.loc[run_dir,['ilisi_system','asw_group']])
    with plt.rc_context({'figure.figsize':(40,10)}):
        path_run=path_integration+run_dir+'/'
        for img_fn in ['umap.png','losses.png']:
            img = mpimg.imread(path_run+img_fn)
            imgplot = plt.imshow(img)
            plt.show()

# %%
# Example z_distance_cycle_nll run
for run_dir in res.query('params_opt=="z_distance_cycle_weight_nll" & param_opt_val==2').index:
    print(run_dir)
    display(res.loc[run_dir,['ilisi_system','asw_group']])
    with plt.rc_context({'figure.figsize':(40,10)}):
        path_run=path_integration+run_dir+'/'
        for img_fn in ['umap.png','losses.png']:
            img = mpimg.imread(path_run+img_fn)
            imgplot = plt.imshow(img)
            plt.show()

# %% [markdown]
# C: It seems integration isnt that bad, but may be affected by lowq cells from human data - thus methods that are more agressive merge them to real cells of mouse, leading to more system mixing (while loosing bio?).

# %% [markdown]
# ## Integration pancreas_conditions - hpap2

# %%
path_integration=path_eval+'pancreas_conditions_hpap2nd/integration/'

# %%
# Load integration results - params and metrics
res=[]
for run in glob.glob(path_integration+'*/'):
    if os.path.exists(run+'args.pkl') and os.path.exists(run+'scib_metrics.pkl'):
        args=pd.Series(vars(pkl.load(open(run+'args.pkl','rb'))))
        metrics=pd.Series(pkl.load(open(run+'scib_metrics.pkl','rb')))
        data=pd.concat([args,metrics])
        data.name=run.split('/')[-2]
        res.append(data)
res=pd.concat(res,axis=1).T

# %%
#  Param that was optimised
res['param_opt_col']=res.params_opt.replace(
    {'vamp':'n_prior_components',
     'z_distance_cycle_weight_std':'z_distance_cycle_weight',
     'z_distance_cycle_weight_kl':'z_distance_cycle_weight',
     'z_distance_cycle_weight_nll':'z_distance_cycle_weight',
     'scvi':None})
res['param_opt_val']=res.apply(
    lambda x: x[x['param_opt_col']] if x['param_opt_col'] is not None else 0,axis=1)

# %%
res['params_opt']=pd.Categorical(res['params_opt'],sorted(res['params_opt'].unique()), True)

# %%
g=sb.catplot( x='param_opt_val', y="asw_group",  col='params_opt',
           kind="swarm", data=res,sharex=False,
          height=2.5,aspect=1.3,color='k')
for ax in g.axes.ravel():
    ax.axhline(0.55,lw=0.5,c='gray')

# %% [markdown]
# C: clisi does not help much here (not shown - constant high)

# %%
g=sb.catplot( x='param_opt_val', y="ilisi_system",  col='params_opt',
           kind="swarm", data=res,sharex=False,
          height=2.5,aspect=1.3,color='k')
for ax in g.axes.ravel():
    ax.axhline(0.05,lw=0.5,c='gray')
    ax.axhline(0.01,lw=0.5,c='gray')

# %%
g=sb.catplot( x='param_opt_val', y="asw_system",  col='params_opt',
           kind="swarm", data=res,sharex=False,
          height=2.5,aspect=1.3,color='k')
for ax in g.axes.ravel():
    ax.axhline(0.05,lw=0.5,c='gray')
    ax.axhline(0.01,lw=0.5,c='gray')

# %%
# Examples
for params in sorted(res['params_opt'].unique()):
    run_dir=res.query('params_opt==@params & asw_group>0.52'
                     ).sort_values('ilisi_system').index[-1]
    print(params,run_dir)
    display(res.loc[run_dir,['ilisi_system','asw_group']])
    with plt.rc_context({'figure.figsize':(40,10)}):
        path_run=path_integration+run_dir+'/'
        for img_fn in ['umap.png','losses.png']:
            img = mpimg.imread(path_run+img_fn)
            imgplot = plt.imshow(img)
            plt.show()

# %% [markdown]
# C: Some very high weights have good integration and bio preserv does not seem that bad - but what does that mean?
#
# Plot example umaps when using very high weight

# %%
# Example z_distance_cycle_std high w (but not extreme) run
for run_dir in res.query('params_opt=="z_distance_cycle_weight_std" & param_opt_val==10').index:
    print(run_dir)
    display(res.loc[run_dir,['ilisi_system','asw_group']])
    with plt.rc_context({'figure.figsize':(40,10)}):
        path_run=path_integration+run_dir+'/'
        for img_fn in ['umap.png','losses.png']:
            img = mpimg.imread(path_run+img_fn)
            imgplot = plt.imshow(img)
            plt.show()

# %%
# Example z_distance_cycle_std high w run
for run_dir in res.query('params_opt=="z_distance_cycle_weight_std" & param_opt_val==50').index:
    print(run_dir)
    display(res.loc[run_dir,['ilisi_system','asw_group']])
    with plt.rc_context({'figure.figsize':(40,10)}):
        path_run=path_integration+run_dir+'/'
        for img_fn in ['umap.png','losses.png']:
            img = mpimg.imread(path_run+img_fn)
            imgplot = plt.imshow(img)
            plt.show()

# %%
# Example z_distance_cycle high w (but not extreme) run
for run_dir in res.query('params_opt=="z_distance_cycle_weight" & param_opt_val==10').index:
    print(run_dir)
    display(res.loc[run_dir,['ilisi_system','asw_group']])
    with plt.rc_context({'figure.figsize':(40,10)}):
        path_run=path_integration+run_dir+'/'
        for img_fn in ['umap.png','losses.png']:
            img = mpimg.imread(path_run+img_fn)
            imgplot = plt.imshow(img)
            plt.show()

# %%
# Example z_distance_cycle high w run
for run_dir in res.query('params_opt=="z_distance_cycle_weight" & param_opt_val==50').index:
    print(run_dir)
    display(res.loc[run_dir,['ilisi_system','asw_group']])
    with plt.rc_context({'figure.figsize':(40,10)}):
        path_run=path_integration+run_dir+'/'
        for img_fn in ['umap.png','losses.png']:
            img = mpimg.imread(path_run+img_fn)
            imgplot = plt.imshow(img)
            plt.show()

# %% [markdown]
# Try kl and ll zdist cycle as have good bio

# %%
# Example z_distance_cycle_kl
for run_dir in res.query('params_opt=="z_distance_cycle_weight_kl" & param_opt_val==1').index:
    print(run_dir)
    display(res.loc[run_dir,['ilisi_system','asw_group']])
    with plt.rc_context({'figure.figsize':(40,10)}):
        path_run=path_integration+run_dir+'/'
        for img_fn in ['umap.png','losses.png']:
            img = mpimg.imread(path_run+img_fn)
            imgplot = plt.imshow(img)
            plt.show()

# %%
# Example z_distance_cycle_nll
for run_dir in res.query('params_opt=="z_distance_cycle_weight_nll" & param_opt_val==1.5').index:
    print(run_dir)
    display(res.loc[run_dir,['ilisi_system','asw_group']])
    with plt.rc_context({'figure.figsize':(40,10)}):
        path_run=path_integration+run_dir+'/'
        for img_fn in ['umap.png','losses.png']:
            img = mpimg.imread(path_run+img_fn)
            imgplot = plt.imshow(img)
            plt.show()

# %%
# Efect of different weights used in different cycle consistency losses
# Plot strongest integrated in each setting
for param in [
    'kl_weight', 'vamp',
     'z_distance_cycle_weight',
     'z_distance_cycle_weight_kl',
     'z_distance_cycle_weight_nll',
     'z_distance_cycle_weight_std']:
    for param_val in  sorted(res.query('params_opt==@param'
                     )['param_opt_val'].unique()):
        run_dir=res.query('params_opt==@param & param_opt_val==@param_val'
                     ).sort_values('ilisi_system').index[-1]
        print(param,param_val,run_dir)
        display(res.loc[run_dir,['ilisi_system','asw_group']])
        with plt.rc_context({'figure.figsize':(80,20)}):
            path_run=path_integration+run_dir+'/'
            for img_fn in ['umap.png','losses.png']:
                img = mpimg.imread(path_run+img_fn)
                imgplot = plt.imshow(img)
                plt.show()

# %%
rcParams['figure.figsize']=(2,2)
g=sb.scatterplot(x='ilisi_system',y='asw_group',
               hue='params_opt',
               data=res.groupby(['params_opt','param_opt_val'],observed=True
                               ).mean().reset_index(),
                #palette='colorblind'
                )
sb.move_legend(g, loc='upper right',bbox_to_anchor=(3, 1))

# %% [markdown]
# ## Integration pancreas organoid public

# %%
path_integration=path_eval+'pancreas_embryo_organoid_public/integration/'

# %%
# Load integration results - params and metrics
res=[]
for run in glob.glob(path_integration+'*/'):
    if os.path.exists(run+'args.pkl') and os.path.exists(run+'scib_metrics.pkl'):
        args=pd.Series(vars(pkl.load(open(run+'args.pkl','rb'))))
        metrics=pd.Series(pkl.load(open(run+'scib_metrics.pkl','rb')))
        data=pd.concat([args,metrics])
        data.name=run.split('/')[-2]
        res.append(data)
res=pd.concat(res,axis=1).T

# %%
#  Param that was optimised
res['param_opt_col']=res.params_opt.replace(
    {'vamp':'n_prior_components',
     'z_distance_cycle_weight_std':'z_distance_cycle_weight',
     'z_distance_cycle_weight_kl':'z_distance_cycle_weight',
     'z_distance_cycle_weight_nll':'z_distance_cycle_weight',
     'scvi':None})
res['param_opt_val']=res.apply(
    lambda x: x[x['param_opt_col']] if x['param_opt_col'] is not None else 0,axis=1)

# %%
res['params_opt']=pd.Categorical(res['params_opt'],sorted(res['params_opt'].unique()), True)

# %%
g=sb.catplot( x='param_opt_val', y="asw_group",  col='params_opt',
           kind="swarm", data=res,sharex=False,
          height=2.5,aspect=1.3,color='k')
for ax in g.axes.ravel():
    ax.axhline(0.52,lw=0.5,c='gray')

# %% [markdown]
# C: clisi does not help much here (not shown - constant high)

# %%
g=sb.catplot( x='param_opt_val', y="ilisi_system",  col='params_opt',
           kind="swarm", data=res,sharex=False,
          height=2.5,aspect=1.3,color='k')
for ax in g.axes.ravel():
    ax.axhline(0.05,lw=0.5,c='gray')
    ax.axhline(0.01,lw=0.5,c='gray')

# %%
# Examples
for params in sorted(res['params_opt'].unique()):
    run_dir=res.query('params_opt==@params & asw_group>0.5'
                     ).sort_values('ilisi_system').index[-1]
    print(params,run_dir)
    display(res.loc[run_dir,['ilisi_system','asw_group']])
    with plt.rc_context({'figure.figsize':(40,10)}):
        path_run=path_integration+run_dir+'/'
        for img_fn in ['umap.png','losses.png']:
            img = mpimg.imread(path_run+img_fn)
            imgplot = plt.imshow(img)
            plt.show()

# %% [markdown]
# C: Not a good example as too litloe bio overlap between systems

# %% [markdown]
# ## Integration retina organoid public

# %%
path_integration=path_eval+'retina_embryo_organoid/integration/'

# %%
# Load integration results - params and metrics
res=[]
for run in glob.glob(path_integration+'*/'):
    if os.path.exists(run+'args.pkl') and os.path.exists(run+'scib_metrics.pkl'):
        args=pd.Series(vars(pkl.load(open(run+'args.pkl','rb'))))
        metrics=pd.Series(pkl.load(open(run+'scib_metrics.pkl','rb')))
        data=pd.concat([args,metrics])
        data.name=run.split('/')[-2]
        res.append(data)
res=pd.concat(res,axis=1).T

# %%
#  Param that was optimised
res['param_opt_col']=res.params_opt.replace(
    {'vamp':'n_prior_components',
     'z_distance_cycle_weight_std':'z_distance_cycle_weight',
     'z_distance_cycle_weight_kl':'z_distance_cycle_weight',
     'z_distance_cycle_weight_nll':'z_distance_cycle_weight',
     'scvi':None})
res['param_opt_val']=res.apply(
    lambda x: x[x['param_opt_col']] if x['param_opt_col'] is not None else 0,axis=1)

# %%
res['params_opt']=pd.Categorical(res['params_opt'],sorted(res['params_opt'].unique()), True)

# %%
g=sb.catplot( x='param_opt_val', y="asw_group",  col='params_opt',
           kind="swarm", data=res,sharex=False,
          height=2.5,aspect=1.3,color='k')
for ax in g.axes.ravel():
    ax.axhline(0.52,lw=0.5,c='gray')

# %%
g=sb.catplot( x='param_opt_val', y="ilisi_system",  col='params_opt',
           kind="swarm", data=res,sharex=False,
          height=2.5,aspect=1.3,color='k')
for ax in g.axes.ravel():
    ax.axhline(0.06,lw=0.5,c='gray')
    ax.axhline(0.02,lw=0.5,c='gray')

# %%
# Examples
for params in sorted(res['params_opt'].unique()):
    run_dir=res.query('params_opt==@params & asw_group>0.5'
                     ).sort_values('ilisi_system').index[-1]
    print(params,run_dir)
    display(res.loc[run_dir,['ilisi_system','asw_group']])
    with plt.rc_context({'figure.figsize':(40,10)}):
        path_run=path_integration+run_dir+'/'
        for img_fn in ['umap.png','losses.png']:
            img = mpimg.imread(path_run+img_fn)
            imgplot = plt.imshow(img)
            plt.show()

# %% [markdown]
# C: Here scVI performs well, but problem of not so good annotation again and lack of cell state overlap potentially.

# %% [markdown]
# ## Integration pancreas organoid public - soon

# %%
path_integration=path_eval+'pancreas_embryo_organoid_public-soon/integration/'

# %%
# Load integration results - params and metrics
res=[]
for run in glob.glob(path_integration+'*/'):
    if os.path.exists(run+'args.pkl') and os.path.exists(run+'scib_metrics.pkl'):
        args=pd.Series(vars(pkl.load(open(run+'args.pkl','rb'))))
        metrics=pd.Series(pkl.load(open(run+'scib_metrics.pkl','rb')))
        data=pd.concat([args,metrics])
        data.name=run.split('/')[-2]
        res.append(data)
res=pd.concat(res,axis=1).T

# %%
#  Param that was optimised
res['param_opt_col']=res.params_opt.replace(
    {'vamp':'n_prior_components',
     'z_distance_cycle_weight_std':'z_distance_cycle_weight',
     'z_distance_cycle_weight_kl':'z_distance_cycle_weight',
     'z_distance_cycle_weight_nll':'z_distance_cycle_weight',
     'scvi':None})
res['param_opt_val']=res.apply(
    lambda x: x[x['param_opt_col']] if x['param_opt_col'] is not None else 0,axis=1)

# %%
res['params_opt']=pd.Categorical(res['params_opt'],sorted(res['params_opt'].unique()), True)

# %%
g=sb.catplot( x='param_opt_val', y="asw_group",  col='params_opt',
           kind="swarm", data=res,sharex=False,
          height=2.5,aspect=1.3,color='k')
for ax in g.axes.ravel():
    ax.axhline(0.52,lw=0.5,c='gray')
    ax.axhline(0.5,lw=0.5,c='gray')

# %%
g=sb.catplot( x='param_opt_val', y="ilisi_system",  col='params_opt',
           kind="swarm", data=res,sharex=False,
          height=2.5,aspect=1.3,color='k')
for ax in g.axes.ravel():
    ax.axhline(0.09,lw=0.5,c='gray')
    ax.axhline(0.04,lw=0.5,c='gray')

# %%
# Examples
for params in sorted(res['params_opt'].unique()):
    run_dir=res.query('params_opt==@params & asw_group>0.5'
                     ).sort_values('ilisi_system').index[-1]
    print(params,run_dir)
    display(res.loc[run_dir,['ilisi_system','asw_group']])
    with plt.rc_context({'figure.figsize':(80,20)}):
        path_run=path_integration+run_dir+'/'
        for img_fn in ['umap.png','losses.png']:
            img = mpimg.imread(path_run+img_fn)
            imgplot = plt.imshow(img)
            plt.show()

# %% [markdown]
# C: Not a good example as too litloe bio overlap between systems

# %% [markdown]
# C: It seems reconstruction loss is still falling quite a lot - maybe need more epochs

# %% [markdown]
# ## Integration pancreas organoid public - soon - longer runtime

# %%
path_integration=path_eval+'pancreas_embryo_organoid_public-soon-long/integration/'

# %%
# Load integration results - params and metrics
res=[]
for run in glob.glob(path_integration+'*/'):
    if os.path.exists(run+'args.pkl') and os.path.exists(run+'scib_metrics.pkl'):
        args=pd.Series(vars(pkl.load(open(run+'args.pkl','rb'))))
        metrics=pd.Series(pkl.load(open(run+'scib_metrics.pkl','rb')))
        data=pd.concat([args,metrics])
        data.name=run.split('/')[-2]
        res.append(data)
res=pd.concat(res,axis=1).T

# %%
#  Param that was optimised
res['param_opt_col']=res.params_opt.replace(
    {'vamp':'n_prior_components',
     'z_distance_cycle_weight_std':'z_distance_cycle_weight',
     'z_distance_cycle_weight_std_anneal':'z_distance_cycle_weight',
     'z_distance_cycle_weight_std_annealkl':'z_distance_cycle_weight',
     'z_distance_cycle_weight_std_vamp':'z_distance_cycle_weight',
     'z_distance_cycle_weight_kl':'z_distance_cycle_weight',
     'z_distance_cycle_weight_nll':'z_distance_cycle_weight',
     'scvi':None})
res['param_opt_val']=res.apply(
    lambda x: (x[x['param_opt_col']] if not isinstance(x[x['param_opt_col']],dict) else 
    x[x['param_opt_col']]['weight_end'])
    if x['param_opt_col'] is not None else 0,axis=1)

# %%
res['params_opt']=pd.Categorical(res['params_opt'],sorted(res['params_opt'].unique()), True)

# %%
g=sb.catplot( x='param_opt_val', y="asw_group",  col='params_opt',
           kind="swarm", data=res,sharex=False,
          height=2.5,aspect=1.5,color='k')
for ax in g.axes.ravel():
    ax.axhline(0.51,lw=0.5,c='gray')
    ax.axhline(0.5,lw=0.5,c='gray')

# %%
g=sb.catplot( x='param_opt_val', y="ilisi_system",  col='params_opt',
           kind="swarm", data=res,sharex=False,
          height=2.5,aspect=1.5,color='k')
for ax in g.axes.ravel():
    ax.axhline(0.15,lw=0.5,c='gray')
    ax.axhline(0.08,lw=0.5,c='gray')

# %%
# Examples
for params in sorted(res['params_opt'].unique()):
    run_dir=res.query('params_opt==@params & asw_group>0.5'
                     ).sort_values('ilisi_system').index[-1]
    print(params,run_dir)
    display(res.loc[run_dir,['ilisi_system','asw_group']])
    with plt.rc_context({'figure.figsize':(80,20)}):
        path_run=path_integration+run_dir+'/'
        for img_fn in ['umap.png','losses.png']:
            img = mpimg.imread(path_run+img_fn)
            imgplot = plt.imshow(img)
            plt.show()

# %%
# Efect of different weights used in different cycle consistency losses
# Plot strongest integrated in each setting
for param in [
    'kl_weight','vamp',
     'z_distance_cycle_weight',
     'z_distance_cycle_weight_kl',
     'z_distance_cycle_weight_nll',
     'z_distance_cycle_weight_std_anneal',
     'z_distance_cycle_weight_std_annealkl',
     'z_distance_cycle_weight_std_vamp',
     'z_distance_cycle_weight_std']:
    for param_val in  sorted(res.query('params_opt==@param'
                     )['param_opt_val'].unique()):
        run_dir=res.query('params_opt==@param & param_opt_val==@param_val'
                     ).sort_values('ilisi_system').index[-1]
        print(param,param_val,run_dir)
        display(res.loc[run_dir,['ilisi_system','asw_group']])
        with plt.rc_context({'figure.figsize':(80,20)}):
            path_run=path_integration+run_dir+'/'
            for img_fn in ['umap.png','losses.png']:
                img = mpimg.imread(path_run+img_fn)
                imgplot = plt.imshow(img)
                plt.show()

# %%
rcParams['figure.figsize']=(2,2)
g=sb.scatterplot(x='ilisi_system',y='asw_group',
               hue='params_opt',
               data=res.groupby(['params_opt','param_opt_val'],observed=True
                               ).mean().reset_index(),
                #palette='colorblind'
                )
sb.move_legend(g, loc='upper right',bbox_to_anchor=(3, 1))

# %%
# Subset of results for presentation
keep=['kl_weight','scvi', 'vamp','z_distance_cycle_weight', 'z_distance_cycle_weight_kl',
 'z_distance_cycle_weight_nll', 'z_distance_cycle_weight_std']
rcParams['figure.figsize']=(2,2)
res_temp=res.query('params_opt in @keep').copy()
res_temp['params_opt'].cat.remove_unused_categories(inplace=True)
g=sb.scatterplot(x='ilisi_system',y='asw_group',
               hue='params_opt',
               data=res_temp.groupby(
                   ['params_opt','param_opt_val'],observed=True
                               ).mean().reset_index(),
                #palette='colorblind'
                )
sb.move_legend(g, loc='upper right',bbox_to_anchor=(3, 1))
del res_temp

# %% [markdown]
# ## Integration adipose sc sn

# %%
path_integration=path_eval+'adipose_scsn/integration/'

# %%
# Load integration results - params and metrics
res=[]
for run in glob.glob(path_integration+'*/'):
    if os.path.exists(run+'args.pkl') and os.path.exists(run+'scib_metrics.pkl'):
        args=pd.Series(vars(pkl.load(open(run+'args.pkl','rb'))))
        metrics=pd.Series(pkl.load(open(run+'scib_metrics.pkl','rb')))
        data=pd.concat([args,metrics])
        data.name=run.split('/')[-2]
        res.append(data)
res=pd.concat(res,axis=1).T

# %%
#  Param that was optimised
res['param_opt_col']=res.params_opt.replace(
    {'vamp':'n_prior_components',
     'z_distance_cycle_weight_std':'z_distance_cycle_weight',
     'z_distance_cycle_weight_std_anneal':'z_distance_cycle_weight',
     'z_distance_cycle_weight_std_annealkl':'z_distance_cycle_weight',
     'z_distance_cycle_weight_std_vamp':'z_distance_cycle_weight',
     'z_distance_cycle_weight_kl':'z_distance_cycle_weight',
     'z_distance_cycle_weight_nll':'z_distance_cycle_weight',
     'scvi':None})
res['param_opt_val']=res.apply(
    lambda x: (x[x['param_opt_col']] if not isinstance(x[x['param_opt_col']],dict) else 
    x[x['param_opt_col']]['weight_end'])
    if x['param_opt_col'] is not None else 0,axis=1)

# %%
res['params_opt']=pd.Categorical(res['params_opt'],sorted(res['params_opt'].unique()), True)

# %%
g=sb.catplot( x='param_opt_val', y="asw_group",  col='params_opt',
           kind="swarm", data=res,sharex=False,
          height=2.5,aspect=1.5,color='k')
for ax in g.axes.ravel():
    ax.axhline(0.53,lw=0.5,c='gray')
    ax.axhline(0.49,lw=0.5,c='gray')

# %%
g=sb.catplot( x='param_opt_val', y="ilisi_system",  col='params_opt',
           kind="swarm", data=res,sharex=False,
          height=2.5,aspect=1.5,color='k')
for ax in g.axes.ravel():
    ax.axhline(0.15,lw=0.5,c='gray')
    ax.axhline(0.08,lw=0.5,c='gray')

# %%
g=sb.catplot( x='param_opt_val', y="asw_system",  col='params_opt',
           kind="swarm", data=res,sharex=False,
          height=2.5,aspect=1.5,color='k')
for ax in g.axes.ravel():
    ax.axhline(0.8,lw=0.5,c='gray')

# %% [markdown]
# C: Asw batch makes no sense based on the below plots

# %%
# Examples
for params in sorted(res['params_opt'].unique()):
    run_dir=res.query('params_opt==@params'
                     ).sort_values('ilisi_system').index[-1]
    print(params,run_dir)
    display(res.loc[run_dir,['ilisi_system','asw_group']])
    with plt.rc_context({'figure.figsize':(80,20)}):
        path_run=path_integration+run_dir+'/'
        for img_fn in ['umap.png','losses.png']:
            img = mpimg.imread(path_run+img_fn)
            imgplot = plt.imshow(img)
            plt.show()

# %%
# Efect of different weights used in different cycle consistency losses
# Plot strongest integrated in each setting
for param in [
    'kl_weight','vamp',
     'z_distance_cycle_weight',
     'z_distance_cycle_weight_kl',
     'z_distance_cycle_weight_nll',
     'z_distance_cycle_weight_std',
     'z_distance_cycle_weight_std_anneal',
     'z_distance_cycle_weight_std_annealkl',
     'z_distance_cycle_weight_std_vamp']:
    for param_val in  sorted(res.query('params_opt==@param'
                     )['param_opt_val'].unique()):
        run_dir=res.query('params_opt==@param & param_opt_val==@param_val'
                     ).sort_values('ilisi_system').index[-1]
        print(param,param_val,run_dir)
        display(res.loc[run_dir,['ilisi_system','asw_group']])
        with plt.rc_context({'figure.figsize':(80,20)}):
            path_run=path_integration+run_dir+'/'
            for img_fn in ['umap.png','losses.png']:
                img = mpimg.imread(path_run+img_fn)
                imgplot = plt.imshow(img)
                plt.show()

# %%
rcParams['figure.figsize']=(2,2)
g=sb.scatterplot(x='ilisi_system',y='asw_group',
               hue='params_opt',
               data=res.groupby(['params_opt','param_opt_val'],observed=True
                               ).mean().reset_index(),
                #palette='colorblind'
                )
sb.move_legend(g, loc='upper right',bbox_to_anchor=(3.1, 1))

# %%
# Subset of results for presentation
keep=['kl_weight','scvi', 'vamp','z_distance_cycle_weight', 'z_distance_cycle_weight_kl',
 'z_distance_cycle_weight_nll', 'z_distance_cycle_weight_std']
rcParams['figure.figsize']=(2,2)
res_temp=res.query('params_opt in @keep').copy()
res_temp['params_opt'].cat.remove_unused_categories(inplace=True)
g=sb.scatterplot(x='ilisi_system',y='asw_group',
               hue='params_opt',
               data=res_temp.groupby(
                   ['params_opt','param_opt_val'],observed=True
                               ).mean().reset_index(),
                #palette='colorblind'
                )
sb.move_legend(g, loc='upper right',bbox_to_anchor=(3, 1))
del res_temp

# %%

# %% [markdown]
# ## TODOs
# - add ari cluster label to better detect too spread out clusters; would be nice to add another batch metric but they give weird results in the scIB report
#
# ## General remarks:
# - z_dist_cyc kl seems to be good for bio preservation, maybe also nll at w=1

# %%
