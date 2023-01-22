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
# Find modified losses in each experiment
losses=['kl_weight', 
          'kl_cycle_weight', 
          'reconstruction_weight', 
          'reconstruction_mixup_weight', 
          'reconstruction_cycle_weight',
          'z_distance_cycle_weight', 
          'translation_corr_weight', 
          'z_contrastive_weight']
def add_losses_summary(res,losses):
    # Check which losses were modified
    # Assumes that defaults of losses are int and modified float!!!!
    # Also does not account for changes in mixup_alpha (assumes same) 
    # and system_decoders - this one can be separately shown as different architecture
    res['tested_losses']=res.apply(
    lambda x: '&'.join([loss.replace('_weight','') for loss in losses 
                        if not isinstance(x[loss],int)]), axis=1)
    # If only one loss has been changed add here normalized tested loss values range
    loss_weight_norm={}
    for loss in losses:
        loss_values=list(sorted(set([v for v in  res[loss] if isinstance(v,float)])))
        # Some losses dont have any hyperparam search
        if len(loss_values)>0:
            loss_weight_norm[loss]=dict(zip(
                loss_values,
                # Need to round as else there is difference at last decimal 
                # sometimes due to precision
                [round(x,5) for x in minmax_scale(loss_values)]))
    # Adds loss weight rank if single loss was used, else adds 0
    res['primary_loss_weight_norm']=res.apply(
        lambda x: loss_weight_norm.get(x['tested_losses']+'_weight'
                                       )[x[x['tested_losses']+'_weight']]
        if x['tested_losses']+'_weight' in loss_weight_norm else 0, axis=1
    )
    res['tested_losses']=pd.Categorical(res['tested_losses'],sorted(res['tested_losses'].unique()),
                                        ordered=True,)


# %%
add_losses_summary(res=res,losses=losses)


# %%
# MAke df for plotting with seaborn (metrics concatenated as rows)
def make_res_plot(res,metrics,cols=[]):
    res_plot=[]
    for metric in metrics:
        res_plot_sub=res[
            ['tested_losses','primary_loss_weight_norm','system_decoders',metric]+cols].copy()
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
    runs_show[loss+'_2D_low']=res.query(
        'tested_losses == @loss & system_decoders==True '+\
        '& primary_loss_weight_norm == 0'
    ).sort_values('ilisi_system',ascending=False).index[0]
    runs_show[loss+'_2D_high']=res.query(
        'tested_losses == @loss & system_decoders==True'+\
        '& primary_loss_weight_norm == 1'
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
os.path.exists(run+'args.pkl')

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
add_losses_summary(res=res_tab,losses=losses)

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

# %%
for loss in res_tab.query('dataset=="pancreas+tabula"')['tested_losses'].unique():
    run_dir=res_tab.query('dataset=="pancreas+tabula" & tested_losses ==@loss'
                 ).sort_values('query-pred_query',ascending=False).index[0]
    print(run_dir)
    with plt.rc_context({'figure.figsize':(40,10)}):
        path_run=path_translation_tabula+run_dir+'/'
        for img_fn in ['losses.png']:
            img = mpimg.imread(path_run+img_fn)
            imgplot = plt.imshow(img)
            plt.show()
