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
# ## Integration

# %%
path_integration=path_eval+'pancreas_example_v1/integration/'

# %%
# Load integration results - params and metrics
res=[]
for run in glob.glob(path_integration+'*/'):
    args=pd.Series(vars(pkl.load(open(run+'args.pkl','rb'))))
    metrics=pd.Series(pkl.load(open(run+'scib_metrics.pkl','rb')))
    data=pd.concat([args,metrics])
    data.name=run.split('/')[-2]
    res.append(data)
res=pd.concat(res,axis=1).T

# %%
defaults={'max_epochs':50,'n_hidden':256,'n_layers':2,
          'n_prior_components':{'filter_col':'prior','filter_val':'vamp'}}
def parse_defaults(res,defaults=defaults):
    # Set values that equal defaults to nan (assumes they were not to be plotted)
    # As add_losses_summary uses nan to determine if they are to be added
    for param, default in defaults.items():
        if isinstance(default, dict):
            res[param]=res.apply(
                lambda x: np.nan if x[default['filter_col']]!=default['filter_val'] 
                else x[param],axis=1)
        else:
            res[param]=res[param].apply( lambda x: np.nan if x==default else x)


# %%
# Find modified losses in each experiment
losses=['kl_weight', 
          'kl_cycle_weight', 
          'reconstruction_weight', 
          'reconstruction_mixup_weight', 
          'reconstruction_cycle_weight',
          'z_distance_cycle_weight', 
          'translation_corr_weight', 
          'z_contrastive_weight',
       ]
other_params=['n_prior_components','n_hidden','n_layers','max_epochs']
def add_losses_summary(res,losses=losses,other_params=other_params):
    # Check which losses were modified
    # Assumes that defaults of losses are int and modified float!!!!
    # Also does not account for changes in mixup_alpha (assumes same) 
    # and system_decoders - this one can be separately shown as different architecture
    losses=[l for l in losses if l in res.columns]
    other_params=[l for l in other_params if l in res.columns]
    res['tested_losses']=res.apply(
    lambda x: '&'.join([loss.replace('_weight','') for loss in losses 
                        if not isinstance(x[loss],int)]), axis=1).fillna('')
    
    res['tested_losses']=res.apply(
        lambda x: '&'.join([x['tested_losses']]+[
            param for param in other_params if not np.isnan(x[param])]).strip('&'),axis=1)
    # If only one loss has been changed add here normalized tested loss values range
    loss_weight_norm={}
    for loss in losses+other_params:
        if loss in losses:
            loss_values=list(sorted(set([v for v in  res[loss] if isinstance(v,float)])))
        else:
            loss_values=list(sorted([v for v in 
                                     res.query(f'~{loss}.isna()',engine='python')
                                     [loss].unique()]))
        # Some losses dont have any hyperparam search
        if len(loss_values)>0:
            loss_weight_norm[loss.replace('_weight','')]=dict(zip(
                loss_values,
                # Need to round as else there is difference at last decimal 
                # sometimes due to precision
                [round(x,5) for x in minmax_scale(loss_values)]))
    # Adds loss weight rank if single loss was used, else adds 0
    res['primary_loss_weight_norm']=res.apply(
        lambda x: loss_weight_norm.get(x['tested_losses']
                                       )[x[x['tested_losses']+'_weight' 
                                           if x['tested_losses']+'_weight' in losses
                                          else x['tested_losses']]]
        if x['tested_losses'] in loss_weight_norm else 0, axis=1
    )
    
    res['tested_losses']=pd.Categorical(res['tested_losses'],sorted(res['tested_losses'].unique()),
                                        ordered=True,)


# %%
# Set to nan if default
parse_defaults(res)
add_losses_summary(res=res,losses=losses)


# %%
# MAke df for plotting with seaborn (metrics concatenated as rows)
def make_res_plot(res,metrics,cols=[]):
    res_plot=[]
    cols=[c for c in ['tested_losses','primary_loss_weight_norm','system_decoders']+cols
          if c in res.columns]
    for metric in metrics:
        res_plot_sub=res[cols+[metric]].copy()
        res_plot_sub['metric_type']=metric
        res_plot_sub.rename({metric:'metric'},axis=1,inplace=True)
        res_plot.append(res_plot_sub)
    res_plot=pd.concat(res_plot)
    return res_plot


# %%
metrics=['ilisi_system','asw_system','clisi','asw_group',
               'ilisi_batch_system-0','ilisi_batch_system-1',
              'asw_batch_system-0','asw_batch_system-1']
res_plot=make_res_plot(res=res,metrics=metrics)

# %%
sb.catplot(
    data=res_plot, sharex=False,
    y="tested_losses", x="metric", row="system_decoders",col='metric_type',
    hue='primary_loss_weight_norm',palette='cividis',
    kind="swarm", height=4, aspect=1.2,
)

# %% [markdown]
# Comparison of bio and batch metrics (separately for lisi and asw types) for 1D and 2D setting

# %%
fig,axs=plt.subplots(2,2,figsize=(10,10))
cmap=dict(zip(res['tested_losses'].unique(),
                 list(mcolors.TABLEAU_COLORS.values())[:res['tested_losses'].nunique()]))
g=sb.scatterplot(x='ilisi_system',y='clisi',hue='tested_losses',
                 data=res.query('system_decoders==True'),ax=axs[0,0], palette=cmap)
g.get_legend().remove()
g.set_title('single decoder')
g=sb.scatterplot(x='asw_system',y='asw_group',hue='tested_losses',
                 data=res.query('system_decoders==True'),ax=axs[0,1], palette=cmap)
g.legend(bbox_to_anchor=(1.8, 0.5), ncol=1,title='tested_losses')
g=sb.scatterplot(x='ilisi_system',y='clisi',hue='tested_losses',
                 data=res.query('system_decoders==False'),ax=axs[1,0], palette=cmap)
g.get_legend().remove()
g.set_title('two decoders')
g=sb.scatterplot(x='asw_system',y='asw_group',hue='tested_losses',
                 data=res.query('system_decoders==False'),ax=axs[1,1], palette=cmap)
g.get_legend().remove()

# %% [markdown]
# C: kl_cycle adds anotehr loss part that is similar to kl, thus less bio preservation?

# %% [markdown]
# Correspondence between lisi and asw for batch and bio

# %%
# Correspondence between lisi and asw
fig,axs=plt.subplots(1,2,figsize=(10,5))
cmap=dict(zip(res['tested_losses'].unique(),
                 list(mcolors.TABLEAU_COLORS.values())[:res['tested_losses'].nunique()]))
g=sb.scatterplot(x='ilisi_system',y='asw_system',hue='tested_losses',style='system_decoders',
                 data=res,ax=axs[0], palette=cmap)
g.get_legend().remove()
g=sb.scatterplot(x='clisi',y='asw_group',hue='tested_losses',style='system_decoders',
                 data=res,ax=axs[1], palette=cmap)
g.legend(bbox_to_anchor=(1.8, 0.5), ncol=1,title='tested_losses')


# %% [markdown]
# C: Lisi and ASW metrics correspond to some extent on system and group eval (but not too well), with lisi being less sensitive, but in batch integration within system they diverge even more (sometimes opposite patterns). clisi does not seem to distinguish well between different bio preservations in embeddings. ASW system thinks that some other losses perform much better in system integration.
#
# C: Based on ilisi z_distance_cycle has superior performance in cross species integration in single and double decoder settings based on, and is the only loss that strongly improves integration in separate species decoders. Interestingly increasing the weight too much leads to ilisi decline. However, on asw other losses can be superior.
#
# C: Based on ilisi reconstruction_cycle weight also helps with integration, also in the double decoder setting. It is also good performer based on asw.
#
# C: interestingly, too high kl weight harms integration. This is less evident for kl_cycle weight. Although this may be in part due to random variation (some other loss values vary greatly).
#
# C: interestingly, vamp prior helps both for integration and bio preservation (expected only bio preservation).

# %% [markdown]
# Bio and batch eval in 1D and 2D system based on selected metrics

# %%
# Asw_group vs ilisi_system
fig,axs=plt.subplots(1,2,figsize=(10,5))
cmap=dict(zip(res['tested_losses'].unique(),
                 list(mcolors.TABLEAU_COLORS.values())[:res['tested_losses'].nunique()]))
g=sb.scatterplot(x='ilisi_system',y='asw_group',hue='tested_losses',
                 data=res.query('system_decoders==False'),ax=axs[0], palette=cmap)
g.get_legend().remove()
g.set_title('single decoder')
g=sb.scatterplot(x='ilisi_system',y='asw_group',hue='tested_losses',
                 data=res.query('system_decoders==True'),ax=axs[1], palette=cmap)
g.legend(bbox_to_anchor=(1.7, 0.5), ncol=1,title='tested_losses')
_=g.set_title('two decoders')


# %% [markdown]
# C: It seems that using z_distance_cycle is superior.

# %% [markdown]
# Display integration embed and losses of selected runs - for every tested loss and for 1D/2D setting show best performing run with lowest and highest tesdted weight

# %%
# Select runs to display in detail
# Select based on bets islis_system
runs_show={}
for loss in sorted(res['tested_losses'].unique()):
    runs_show[loss+'_1D_low']=res.query(
        'tested_losses == @loss & system_decoders==False'+\
        '& primary_loss_weight_norm == 0'
    ).sort_values('ilisi_system',ascending=False).index[0]
    runs_show[loss+'_1D_high']=res.query(
        'tested_losses == @loss & system_decoders==False'+\
        '& primary_loss_weight_norm == 1' 
    ).sort_values('ilisi_system',ascending=False).index[0]
    runs=res.query(
        'tested_losses == @loss & system_decoders==True '+\
        '& primary_loss_weight_norm == 0'
    ).sort_values('ilisi_system',ascending=False)
    if len(runs)>0:
        runs_show[loss+'_2D_low']=runs.index[0]
    runs=res.query(
        'tested_losses == @loss & system_decoders==True'+\
        '& primary_loss_weight_norm == 1'
    ).sort_values('ilisi_system',ascending=False)
    if len(runs)>0:
        runs_show[loss+'_2D_high']=runs.index[0]
for k,v in runs_show.items():
    print(k,v)

# %%
for run_info, run_dir in runs_show.items():
    print(run_info,run_dir)
    print(res.loc[run_dir,metrics])
    with plt.rc_context({'figure.figsize':(40,10)}):
        path_run=path_integration+run_dir+'/'
        for img_fn in ['losses.png','umap.png']:
            img = mpimg.imread(path_run+img_fn)
            imgplot = plt.imshow(img)
            plt.show()

# %% [markdown]
# C: ASW_system sometimes does not capture well how badly data is integrated. clisi does not cpature well bad bio preservation. Thus use ASW_group and ilisi_system

# %% [markdown]
# Display integration embed and losses of selected runs - for 1D setting show best performing run among kl, reconstruction_cycle and z_distance_cycle - as represetatives of cVAE and thwo other losses that lead to better integration.

# %%
# Select runs to display in detail
# Select based on best ilis_system
runs_show={}
for loss in ['kl','reconstruction_cycle','z_distance_cycle']:
    runs_show[loss+'_1D']=res.query(
        'tested_losses == @loss & system_decoders==False'
    ).sort_values('ilisi_system',ascending=False).index[0]
for k,v in runs_show.items():
    print(k,v)

# %%
for run_info, run_dir in runs_show.items():
    print(run_info,run_dir)
    print(res.loc[run_dir,metrics])
    with plt.rc_context({'figure.figsize':(40,10)}):
        path_run=path_integration+run_dir+'/'
        for img_fn in ['losses.png','umap.png']:
            img = mpimg.imread(path_run+img_fn)
            imgplot = plt.imshow(img)
            plt.show()

# %% [markdown]
# C: reconstruction cycle loss seems to have better integration than cVAE but not higher ilisi

# %% [markdown]
# Effect of z_distance_cycle on islis_system as seems unlinear relationship. Also check for KL.

# %%
with plt.rc_context({'figure.figsize':(3,2)}):
    sb.swarmplot(
        x='z_distance_cycle_weight',y='ilisi_system',
        data=res.query('tested_losses =="z_distance_cycle" & system_decoders==False'))

# %%
with plt.rc_context({'figure.figsize':(3,2)}):
    sb.swarmplot(
        x='kl_weight',y='ilisi_system',
        data=res.query('tested_losses =="kl" & system_decoders==False'))

# %% [markdown]
# ## Translation

# %%
path_translation=path_eval+'pancreas_example_v0/translation/'

# %%
# Load params and translation eval metric
res=[]
for run in glob.glob(path_translation+'*/'):
    args=pd.Series(vars(pkl.load(open(run+'args.pkl','rb'))))
    transl_corr=pd.read_table(run+'translation_correlation.tsv',index_col=0)
    transl_corr=pd.Series({'ref-pred_ref':transl_corr.at['ref','pred_ref'],
                      'query-pred_query':transl_corr.at['query','pred_query'],
                          'ref-pred_query':transl_corr.at['ref','pred_query'],
                          'query-pred_ref':transl_corr.at['query','pred_ref'],})
    data=pd.concat([args,transl_corr])
    data.name=run.split('/')[-2]
    res.append(data)
res=pd.concat(res,axis=1).T

# %%
parse_defaults(res)
add_losses_summary(res=res,losses=losses)

# %%
metrics=['ref-pred_ref','query-pred_query','ref-pred_query','query-pred_ref']
res_plot=make_res_plot(res=res,metrics=metrics)

# %%
# Compare results of all runs, separately for loss settings
style=sb.axes_style()
sb.set_style("whitegrid")
g=sb.catplot(
    data=res_plot, sharex='col',
    y="tested_losses", x="metric", row="system_decoders",col='metric_type',
    hue='primary_loss_weight_norm',palette='cividis',
    kind="swarm", height=4, aspect=1.4,
)
sb.set_style(style)

# %%
res.sort_values('query-pred_query',ascending=False).head()[metrics]

# %% [markdown]
# C: Kl_cycle interestingly destroys self-reconstruction but not the translation. Probably makes latent space more Gaussian, which adds noise for self-reconstruction but is somehow ok for translation? Also seems to push the q and r more together based on q-pred_r and r-pred_q.
#
# C: Translation_corr reduces self reconstruction in 1D setting, but not otherwise. Enforces query and ref to pbe predicted more similarly in single decoder, so probably brings everything together - the predicted correlation of q-pred_r and r-pred_q is higher. Interestingly, this does not reduce q-pred_q in 2D setting (maybe even improves) - possibly since it is hard to bring the 2D together so this loss is needed. In 1D setting it must push single decoder together for 2 systems but in 2D system it may work by pushing more 2nd D to the 1st one - thus does not destroy self-recostruction but regularise translation. TODO could pass loss only to the 2nd decoder?! 
#
# C: Baseline cVAE  (at smallest kl weight) seems to perform quite well, although quite some variability.
#
# C: Mixup seems to help in 1D setting but not 2D setting. At some loss valiues it stops to improve - maybe other params could be also tuned. For 2D setting needs to be considered that some of these probably arent well integrated in 2D so adding this besides other losses may help. Indeed, when other losses are added the performance is improved in 2D setting.
#
# C: Losses important for bringing toggether latent space and prediction help in 2D setting (z_distance_cycle, translation_corr), depending also on weight or a lot of variation. Reconstruction_cycle does not help consistently (also not so clear for z_cycle_distance).
#
# C: Interestingly, some cases that lead to good integration do not lead to good translation and vice versa. But system integration is surely not the only thing affecting outcome (also bio preservation, regularisation, ...). 
#
# C: For some losses the weights dont matter much (e.g. transl corr) - maybe did not get the righ weight range? However, there is a small trend of betetr performance at not so low values.
#
# C: For some the variation is large accross weights (e.g. reconstruction cycle) - maybe unstable training? 
#
# C: Combinning multiple params in 2D setting didnt help much, except for improving mixup.
#
# C: Having two decoders improves prediction of reference while still enabling good prediction of query. Maybe having single decoder leads to regularisation (learn accross systems). Translation corr weight has similar effect, but unclear how this may work if we had different genes in the two decoders (e.g. how would be predicted genes that couldnt have correlation based regularisation).
#
# C: Combined weights for 2D: removing z_cycle_distance and reconstruction_cycle weights while keeping translation_corr and mixup does not lead to better performance than having all of them, despite shown below that the first two losses do not seem to lead to improvement. Could be due to lower number of runs where there would be an opportunity to improve. However, at least there are no runs at the lower eval end with the reduced setting. It also does not seem to perform as well as mixup only in 1D setting. Having mixup and z_distance_cycle does not work as well.
#
# C: Using more epochs doe snot help to make the results less variable.

# %%
# N of epochs
for run_dir in res.query(
    'tested_losses =="max_epochs" & max_epochs==200').index:
    print(run_dir)
    with plt.rc_context({'figure.figsize':(40,10)}):
        path_run=path_translation+run_dir+'/'
        for img_fn in ['losses.png']:
            img = mpimg.imread(path_run+img_fn)
            imgplot = plt.imshow(img)
            plt.show()

# %% [markdown]
# C: pancreas probably doe snot need more than 50 epochs

# %%
# 2D reconstruction_cycle variation at max and min weight
loss='reconstruction_cycle'
for run_dir in res.query(
    'tested_losses == @loss & system_decoders==True'+\
    '& (primary_loss_weight_norm == 1 |primary_loss_weight_norm == 0)'
).sort_values(['primary_loss_weight_norm','query-pred_query']).index:
    print(run_dir)
    print(res.loc[run_dir,metrics])
    with plt.rc_context({'figure.figsize':(40,10)}):
        path_run=path_translation+run_dir+'/'
        for img_fn in ['losses.png']:
            img = mpimg.imread(path_run+img_fn)
            imgplot = plt.imshow(img)
            plt.show()

# %% [markdown]
# C: Reconstruction loss cycle does not seem to have much effect on training. It is similar to normal reconstruction weight (already checked that code should be correct and there is no buggs in mixing up both reconst losses).

# %%
# 2D translation_correlation variation at max and min weight
loss='translation_corr'
for run_dir in res.query(
    'tested_losses == @loss & system_decoders==True'+\
    '& (primary_loss_weight_norm == 1 |primary_loss_weight_norm == 0)'
).sort_values(['primary_loss_weight_norm','query-pred_query']).index:
    print(run_dir)
    print(res.loc[run_dir,metrics])
    with plt.rc_context({'figure.figsize':(40,10)}):
        path_run=path_translation+run_dir+'/'
        for img_fn in ['losses.png']:
            img = mpimg.imread(path_run+img_fn)
            imgplot = plt.imshow(img)
            plt.show()

# %% [markdown]
# C: It seems taht not a high weight is needed to bring translation_corr down. The end loss is higher at lower weight but it seems from the prediction corr results that increasing it pushes systems together - could try lower weight to prevent too strong regularisation. However, if the loss weight is too small it is less smooth.

# %% [markdown]
# #### Check how joint losses affect the performance

# %% [markdown]
# ##### In the case without mixup

# %% [markdown]
# Combinations of different losses

# %%
# Prepare data for plott
x=res.query(
    'tested_losses=="reconstruction_cycle&z_distance_cycle&translation_corr"').copy()
# Ensure color wont be categorical
x['query-pred_query']=x['query-pred_query'].astype(float)
# Set min of col to 0 so that size scaling is done properly by plotly
#x['query-pred_query_size']=x['query-pred_query']-x['query-pred_query'].min()
# Add jitter
for col in ['reconstruction_cycle_weight', 'z_distance_cycle_weight',
           'translation_corr_weight']:
    col_range=x[col].max()-x[col].min()
    x[col]=x[col]+col_range*np.random.randint(-1,1,size=x.shape[0])*0.05

# %%
fig = px.scatter_3d(x, 
     x='reconstruction_cycle_weight', y='z_distance_cycle_weight', z='translation_corr_weight',
              color='query-pred_query',#size='query-pred_query',size_max=50
                   )
fig.show()
# Some points seem to be missing, maybe jitter causes problem somehow?

# %% [markdown]
# Effect of individual losses (irrespective of other losses in combination). 
#
# As full grid search was run on all losses the effect of other params being changed should average out when looking at single loss if assuming no interactions.

# %%
fig,axs=plt.subplots(1,3, sharey=True, figsize=(9,3))
for i,col in enumerate(['reconstruction_cycle_weight', 'z_distance_cycle_weight',
           'translation_corr_weight']):
    sb.boxplot(
        x=col,y='query-pred_query',whis=3,
        data=res.query(
        'tested_losses=="reconstruction_cycle&z_distance_cycle&translation_corr"'),
        ax=axs[i])
fig.tight_layout()

# %% [markdown]
# C: Reconstruction cycle weight needs to be small (maybe disable?). Incrteasing translation cor weight and z distance cycle weight may improve a bit.

# %% [markdown]
# ##### In case with mixup

# %% [markdown]
# Effect of loss components individually

# %%
fig,axs=plt.subplots(1,4, sharey=True, figsize=(12,3))
for i,col in enumerate(['reconstruction_mixup_weight','reconstruction_cycle_weight', 
                        'z_distance_cycle_weight','translation_corr_weight']):
    sb.boxplot(
        x=col,y='query-pred_query',whis=10,
        data=res.query(
        'tested_losses=="reconstruction_mixup&reconstruction_cycle&z_distance_cycle&translation_corr"'),
        ax=axs[i])
fig.tight_layout()

# %% [markdown]
# C: In case of mixup strong mixup weight seems to help (at certain valus) and x_distance-cycle may need to be lower. Other losses have same pattern as without mixup.

# %% [markdown]
# Combinations of different losses; dont compare reconstruction_cycle as know that it does not perform well.

# %%
# Prepare data for plott
x=res.query(
    'tested_losses=="reconstruction_mixup&reconstruction_cycle&z_distance_cycle&translation_corr"').copy()
# Ensure color wont be categorical
x['query-pred_query']=x['query-pred_query'].astype(float)
# Set min of col to 0 so that size scaling is done properly by plotly
#x['query-pred_query_size']=x['query-pred_query']-x['query-pred_query'].min()
# Add jitter
for col in ['reconstruction_mixup_weight', 'z_distance_cycle_weight',
           'translation_corr_weight']:
    col_range=x[col].max()-x[col].min()
    x[col]=x[col]+col_range*np.random.randint(-1,1,size=x.shape[0])*0.05

# %%
fig = px.scatter_3d(x, 
     x='reconstruction_mixup_weight', y='z_distance_cycle_weight', z='translation_corr_weight',
              color='query-pred_query',#size='query-pred_query',size_max=50
                   )
fig.show()
# Some points seem to be missing, maybe jitter causes problem somehow?

# %% [markdown]
# C: even at higher values of z_distance_weight there are some well performing results.

# %%
res.query(
    'tested_losses=="reconstruction_mixup&reconstruction_cycle&z_distance_cycle&translation_corr"'
).sort_values('query-pred_query',ascending=False).head(n=10)[metrics]

# %% [markdown]
# C: Some top perfroming runs have higher z_distance_cycle weight

# %% [markdown]
# ##### With mixup and translation_corr

# %%
fig,axs=plt.subplots(1,2, sharey=True, figsize=(6,3))
for i,col in enumerate(['reconstruction_mixup_weight','translation_corr_weight']):
    sb.boxplot(
        x=col,y='query-pred_query',whis=10,
        data=res.query(
        'tested_losses=="reconstruction_mixup&translation_corr"'),
        ax=axs[i])
fig.tight_layout()

# %% [markdown]
# ##### With mixup and z_cycle_distance

# %%
fig,axs=plt.subplots(1,2, sharey=True, figsize=(6,3))
for i,col in enumerate(['reconstruction_mixup_weight','z_distance_cycle_weight']):
    sb.boxplot(
        x=col,y='query-pred_query',whis=10,
        data=res.query(
        'tested_losses=="reconstruction_mixup&z_distance_cycle"'),
        ax=axs[i])
fig.tight_layout()

# %% [markdown]
# C: Higher mixup improves performance  when z_distance_cycle is used and vice versa when transl_corr is used.
#
# C: z_distance_cycle probably isnt that useful here.

# %% [markdown]
# ### Translation comparison when using tabula data 
# Hoped that this will perform better, especially when using mixup

# %%
path_translation_tabula=path_eval+'pancreas_tabula_example_v0/translation/'

# %%
# Load params and translation eval metric
res_tab=[]
for run in glob.glob(path_translation_tabula+'*/'):
    # Add this so that code can be tested while not all runs finished
    if os.path.exists(run+'args.pkl') and os.path.exists(run+'translation_correlation.tsv'):
        args=pd.Series(vars(pkl.load(open(run+'args.pkl','rb'))))
        transl_corr=pd.read_table(run+'translation_correlation.tsv',index_col=0)
        transl_corr=pd.Series({'ref-pred_ref':transl_corr.at['ref','pred_ref'],
                          'query-pred_query':transl_corr.at['query','pred_query'],
                              'ref-pred_query':transl_corr.at['ref','pred_query'],
                              'query-pred_ref':transl_corr.at['query','pred_ref'],})
        data=pd.concat([args,transl_corr])
        data.name=run.split('/')[-2]
        res_tab.append(data)
res_tab=pd.concat(res_tab,axis=1).T

# %%
parse_defaults(res_tab)
add_losses_summary(res=res_tab)

# %%
res_sub=[]
for idx,specs in res_tab.drop_duplicates(['tested_losses','system_decoders']).iterrows():
    sub=res.query(
        f'tested_losses=="{specs.at["tested_losses"]}" & '+
        f'system_decoders == {specs.at["system_decoders"]}')
    if sub.shape[0]>0:
        res_sub.append(sub)
res_sub=pd.concat(res_sub)

# %%
res_sub['dataset']='pancreas'
res_tab['dataset']='pancreas+tabula'
res_tab_combined=pd.concat([res_sub,res_tab])
# Update losses info as need top re-scale weights based on both runs
add_losses_summary(res=res_tab_combined,losses=losses)
res_tab_combined['tested_losses-dataset']=res_tab_combined.apply(
    lambda x: x.tested_losses+'-'+x.dataset, axis=1)

# %%
metrics=['ref-pred_ref','query-pred_query','ref-pred_query','query-pred_ref']
res_plot_tab_combined=make_res_plot(res=res_tab_combined,metrics=metrics,
                                    cols=['tested_losses-dataset'])

# %%
# Compare results of all runs, separately for loss settings
style=sb.axes_style()
sb.set_style("whitegrid")
g=sb.catplot(
    data=res_plot_tab_combined,
    y="tested_losses-dataset", x="metric", col='metric_type',
    hue='primary_loss_weight_norm',palette='cividis',
    kind="swarm", height=4, aspect=1.4,
)
sb.set_style(style)

# %% [markdown]
# C: Adding tabula dataset does not work well. In general decreases performance (self prediction nd translation).
#
# C: Adding VAMP prior helps and mixup, but still does not achieve performance of using pancreatic data only.
#
# C: Increasing network capacity (layers, hidden) does not help.

# %%
for loss in res_tab.query('dataset=="pancreas+tabula"')['tested_losses'].unique():
    run_dir=res_tab.query('dataset=="pancreas+tabula" & tested_losses ==@loss'
                 ).sort_values('query-pred_query',ascending=False).index[0]
    print(loss,run_dir)
    with plt.rc_context({'figure.figsize':(40,10)}):
        path_run=path_translation_tabula+run_dir+'/'
        for img_fn in ['losses.png']:
            img = mpimg.imread(path_run+img_fn)
            imgplot = plt.imshow(img)
            plt.show()

# %% [markdown]
# Training of all kl-pancreas+tabula runs as they have very different final performance

# %%
for run_dir in res_tab.query(
    'dataset=="pancreas+tabula" & tested_losses =="max_epochs" & max_epochs==100').index:
    print(run_dir)
    with plt.rc_context({'figure.figsize':(40,10)}):
        print(res_tab.loc[run_dir,metrics])
        path_run=path_translation_tabula+run_dir+'/'
        for img_fn in ['losses.png']:
            img = mpimg.imread(path_run+img_fn)
            imgplot = plt.imshow(img)
            plt.show()

# %% [markdown]
# C: there is different behaviour wrt reconstruction and KL accross runs.

# %%
for run_dir in res_tab.query(
    'dataset=="pancreas+tabula" & tested_losses =="max_epochs" & max_epochs==200').index:
    print(run_dir)
    with plt.rc_context({'figure.figsize':(40,10)}):
        print(res_tab.loc[run_dir,metrics])
        path_run=path_translation_tabula+run_dir+'/'
        for img_fn in ['losses.png']:
            img = mpimg.imread(path_run+img_fn)
            imgplot = plt.imshow(img)
            plt.show()

# %% [markdown]
# C: Having 50 (or a bit more, e.g. 70) epochs is probably also ok for tabula+pancreas as soon after start overfitting in some cases or at least no longer improve val loss

# %% [markdown]
# How does final loss affect translation performance - test for cVAE case where kl=1

# %%
# Df with loss and metric values
res_tab_loss=[]
for loss in ['max_epochs', 'kl']:
    for run_dir in res_tab.query(
        'dataset=="pancreas+tabula" & tested_losses ==@loss & kl_weight==1').index:
        sub=res_tab.loc[run_dir,metrics].to_dict()
        sub['run_dir']=run_dir
        history=pkl.load(open(path_translation_tabula+run_dir+'/losses.pkl','rb'))
        for l in ['loss', 'reconstruction_loss', 'kl_local']:
            for l_type in ['train','validation']:
                l_name=l+'_'+l_type
                sub[l_name]=history[l_name].values.ravel()[-1] if l_name in history else np.nan
        res_tab_loss.append(sub)
res_tab_loss=pd.DataFrame(res_tab_loss)

# %%
# Plot relationship between training losses and metrics
metrics_plot_loss=['ref-pred_ref','query-pred_query']
fig,ax=plt.subplots(2,3,figsize=(3*3.5,2*3))
for i,loss in enumerate(['loss', 'reconstruction_loss', 'kl_local']):
    res_tab_loss_plot=[]
    for loss_type in ['train','validation']:
        sub=res_tab_loss[metrics_plot_loss+[loss+'_'+loss_type]]
        sub.rename({loss+'_'+loss_type:loss},axis=1,inplace=True)
        sub['loss_type']=loss_type
        res_tab_loss_plot.append(sub)
    res_tab_loss_plot=pd.concat(res_tab_loss_plot)
    for j,metric in enumerate(metrics_plot_loss):
        sb.scatterplot(x=loss,y=metric,hue='loss_type',data=res_tab_loss_plot, ax=ax[j,i])
        if i==2 and j==1:
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        else:
            ax[j,i].get_legend().remove()
plt.tight_layout()

# %% [markdown]
# ## Performance on smaller training data

# %%
path_subset=path_eval+'subset_v1/translation/'

# %%
# Load params and translation eval metric
res_tab=[]
for run in glob.glob(path_subset+'*/'):
    # Add this so that code can be tested while not all runs finished
    if os.path.exists(run+'args.pkl') and os.path.exists(run+'translation_correlation.tsv'):
        args=pd.Series(vars(pkl.load(open(run+'args.pkl','rb'))))
        transl_corr=pd.read_table(run+'translation_correlation.tsv',index_col=0)
        transl_corr=pd.Series({'ref-pred_ref':transl_corr.at['ref','pred_ref'],
                          'query-pred_query':transl_corr.at['query','pred_query'],
                              'ref-pred_query':transl_corr.at['ref','pred_query'],
                              'query-pred_ref':transl_corr.at['query','pred_ref'],})
        data=pd.concat([args,transl_corr])
        data.name=run.split('/')[-2]
        res_tab.append(data)
res_tab=pd.concat(res_tab,axis=1).T

# %%
res_tab['dataset']=res_tab['path_adata'].map({
    '/lustre/groups/ml01/workspace/karin.hrovatin/data/cross_species_prediction/pancreas_example/combined_orthologues.h5ad':'pancreas',
    '/lustre/groups/ml01/workspace/karin.hrovatin/data/cross_species_prediction/pancreas_example/combined_tabula_orthologues.h5ad':'pancreas+tabula'})

# %%
metrics=['ref-pred_ref','query-pred_query','ref-pred_query','query-pred_ref']
res_plot=make_res_plot(res=res_tab,metrics=metrics,cols=['dataset','train_size'])
# Must convert this to str as sb expects it for swarmplot not to be numeric
res_plot['train_size']=pd.Categorical(
    values=res_plot['train_size'].astype(str),
    categories=[str(s) for s  in sorted(res_plot['train_size'].unique())], ordered=True)

# %%
# Compare results of all runs, separately for loss settings
style=sb.axes_style()
sb.set_style("whitegrid")
g=sb.catplot(
    data=res_plot,
    y="train_size", x="metric", col='metric_type',row='dataset',
    kind="swarm", height=3, aspect=1.5,
)
sb.set_style(style)

# %% [markdown]
# C: The performance doesnt change much even with very small training dataset, there are options:
# - The model is at max capacity and does not improve upon increasing data. Interestingly, despite having larger val than test set at 0.1 it still performes better when combining both test+val in prediction (not so much in favour of memorisation then?).
# - The cells that are added are very similar (random subset of cells from same cell types). Thus adding them does not lead to much better performance. 
# - The model may be just memorising data. Adding more diverse cell types (e.g. pancreas only and pancreas&tabula) leads to lover performance, so model isnt able to deal with this complexity? Indeed, the model may be just memorising as also self-prediction is potentially worse when more data is used (would need more replicates to be sure).

# %% [markdown]
# Self correlation on train and val data

# %%
# Load params and prediction on train/val sets
res_pred=[]
for run in glob.glob(path_subset+'*/'):
    # Add this so that code can be tested while not all runs finished
    if os.path.exists(run+'args.pkl') and os.path.exists(run+'prediction_correlation_trainval.tsv'):
        args=pd.Series(vars(pkl.load(open(run+'args.pkl','rb'))))
        data=pd.read_table(run+'prediction_correlation_trainval.tsv',index_col=0)
        for k,v in args.items():
            data[k]=v
        res_pred.append(data)
res_pred=pd.concat(res_pred)

# %%
res_pred['dataset']=res_pred['path_adata'].map({
    '/lustre/groups/ml01/workspace/karin.hrovatin/data/cross_species_prediction/pancreas_example/combined_orthologues.h5ad':'pancreas',
    '/lustre/groups/ml01/workspace/karin.hrovatin/data/cross_species_prediction/pancreas_example/combined_tabula_orthologues.h5ad':'pancreas+tabula'})

# %%
# Plot self-corr distn over cells
style=sb.axes_style()
sb.set_style("whitegrid")
g=sb.catplot(
    data=res_pred,
    y="corr", x="name", hue='set', col='train_size',row='dataset',
    kind="violin", height=3, aspect=1.2,
)
sb.set_style(style)

# %% [markdown]
# C: Correlation on train and val set seems to perform similarly.

# %% [markdown]
# Training history examples

# %%
for dataset in res_tab['dataset'].unique():
    for train_size in [0.1,0.9]:
        run_dir=res_tab.query('dataset==@dataset & train_size ==@train_size'
                     ).sort_values('query-pred_query',ascending=False).index[0]
        print(dataset, train_size, run_dir)
        with plt.rc_context({'figure.figsize':(40,10)}):
            path_run=path_subset+run_dir+'/'
            for img_fn in ['losses.png']:
                img = mpimg.imread(path_run+img_fn)
                imgplot = plt.imshow(img)
                plt.show()

# %% [markdown]
# ## LR optimisation

# %%
path_subset=path_eval+'lr/translation/'

# %%
# Load params and translation eval metric
res=[]
for run in glob.glob(path_subset+'*/'):
    # Add this so that code can be tested while not all runs finished
    if os.path.exists(run+'args.pkl') and os.path.exists(run+'translation_correlation.tsv'):
        args=pd.Series(vars(pkl.load(open(run+'args.pkl','rb'))))
        transl_corr=pd.read_table(run+'translation_correlation.tsv',index_col=0)
        transl_corr=pd.Series({'ref-pred_ref':transl_corr.at['ref','pred_ref'],
                          'query-pred_query':transl_corr.at['query','pred_query'],
                              'ref-pred_query':transl_corr.at['ref','pred_query'],
                              'query-pred_ref':transl_corr.at['query','pred_ref'],})
        history_end=pd.Series({'hisEnd_'+l:vals.values.ravel()[-1]
                           for l,vals in pkl.load(open(run+'losses.pkl','rb')).items()})
        history_min=pd.Series({'hisMin_'+l:vals.min()
                           for l,vals in pkl.load(open(run+'losses.pkl','rb')).items()})
        data=pd.concat([args,history_end,history_min,transl_corr])
        data.name=run.split('/')[-2]
        res.append(data)
res=pd.concat(res,axis=1).T

# %%
res['dataset']=res['path_adata'].map({
    '/lustre/groups/ml01/workspace/karin.hrovatin/data/cross_species_prediction/pancreas_example/combined_orthologues.h5ad':'pancreas',
    '/lustre/groups/ml01/workspace/karin.hrovatin/data/cross_species_prediction/pancreas_example/combined_tabula_orthologues.h5ad':'pancreas+tabula'})

# %%
params_opt=['lr','lr_patience', 'lr_factor', 'lr_min','lr_threshold']
losses_hist=['_'.join([h,l,t]) for h,l,t in itertools.product(['hisMin','hisEnd'],
                              ['loss', 'reconstruction_loss','kl_local'],
                              ['train','validation'])]
res_sub_plot=res.query('dataset=="pancreas"')[
       ['ref-pred_ref','query-pred_query']+losses_hist+params_opt]
res_sub_plot[losses_hist+params_opt]=pd.DataFrame(
    minmax_scale(res_sub_plot[losses_hist+params_opt]),
    columns=losses_hist+params_opt,index=res_sub_plot.index)
res_sub_plot=res_sub_plot.astype(float)

# %%
for p in params_opt:
    print(p,sorted(res.query('dataset=="pancreas"')[p].unique()))

# %%
rcParams['figure.figsize']=(5,7)
sb.heatmap(res_sub_plot.sort_values('query-pred_query',ascending=False),yticklabels=False)

# %%
rcParams['figure.figsize']=(5,7)
sb.heatmap(res_sub_plot.sort_values('hisMin_loss_train'),yticklabels=False)

# %%
# Plot relationship between training losses and metrics
metrics_plot_loss=['ref-pred_ref','query-pred_query']
fig,ax=plt.subplots(2,3,figsize=(3*3.5,2*3))
for i,loss in enumerate(['hisEnd_loss', 'hisEnd_reconstruction_loss', 'hisEnd_kl_local']):
    res_loss_plot=[]
    for loss_type in ['train','validation']:
        sub=res.query('dataset=="pancreas"')[metrics_plot_loss+[loss+'_'+loss_type]]
        sub.rename({loss+'_'+loss_type:loss},axis=1,inplace=True)
        sub['loss_type']=loss_type
        res_loss_plot.append(sub)
    res_loss_plot=pd.concat(res_loss_plot)
    for j,metric in enumerate(metrics_plot_loss):
        sb.scatterplot(x=loss,y=metric,hue='loss_type',data=res_loss_plot, ax=ax[j,i])
        if i==2 and j==1:
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        else:
            ax[j,i].get_legend().remove()
plt.tight_layout()

# %%
rcParams['figure.figsize']=(3,3)
sb.scatterplot(x='ref-pred_ref',y='query-pred_query',data=res.query('dataset=="pancreas"'))

# %% [markdown]
# Variance of translation metric depending on lr setting (accross 3 random re-runs).

# %%
x=res.query('dataset=="pancreas"').groupby(
    ['lr','lr_patience', 'lr_factor', 'lr_min','lr_threshold']
).var()['query-pred_query'].reset_index()
x=pd.DataFrame(minmax_scale(x),columns=x.columns,index=x.index).astype(float)

# %%
rcParams['figure.figsize']=(2,3)
sb.heatmap(x.sort_values('query-pred_query'),yticklabels=False)

# %%
