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
import warnings

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
path_integration=path_eval+'presFeb23/integration/'

# %%
# Load integration results - params and metrics
res=[]
for run in glob.glob(path_integration+'*/'):
    if os.path.exists(run+'args.pkl') and os.path.exists(run+'scib_metrics.pkl'):
        args=pd.Series(vars(pkl.load(open(run+'args.pkl','rb'))))
        metrics=pd.Series(pkl.load(open(run+'scib_metrics.pkl','rb')))
        history_end=pd.Series({'hisEnd_'+l:vals.values.ravel()[-1]
                           for l,vals in pkl.load(open(run+'losses.pkl','rb')).items()})
        data=pd.concat([args,history_end,metrics])
        data.name=run.split('/')[-2]
        res.append(data)
res=pd.concat(res,axis=1).T

# %%
res.params_opt.unique()

# %%
#  Param that was optimised
res['param_opt_col']=res.params_opt.replace(
    {'vamp':'n_prior_components',
     'z_distance_cycle_weight_std':'z_distance_cycle_weight',
    'z_distance_cycle_weight_nll':'z_distance_cycle_weight',
    'z_distance_cycle_weight_kl':'z_distance_cycle_weight',
     'vamp_sys':'prior_components_system',
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
    ax.axhline(0.65,lw=0.5,c='gray')
    ax.axhline(0.75,lw=0.5,c='gray')

# %%
g=sb.catplot( x='param_opt_val', y="ilisi_system",  col='params_opt',
           kind="swarm", data=res,sharex=False,
          height=2.5,aspect=1.3,color='k')
for ax in g.axes.ravel():
    ax.axhline(0.05,lw=0.5,c='gray')
    ax.axhline(0.1,lw=0.5,c='gray')

# %% [markdown]
# C: MSE_std based z_dits_cycle performs well, but is unstable.

# %%
rcParams['figure.figsize']=(2,2)
g=sb.scatterplot(x='ilisi_system',y='asw_group',
               hue='params_opt',
               data=res.groupby(['params_opt','param_opt_val'],observed=True
                               ).mean().reset_index(),
                palette='colorblind')
sb.move_legend(g, loc='upper right',bbox_to_anchor=(3, 1))

# %% [markdown]
# #### Effect of seed on different params

# %%
sb.catplot(x='seed',y='ilisi_system',hue='param_opt_val',col='params_opt',
           data=res,kind='swarm',height=2,aspect=1.5)

# %%
sb.catplot(x='seed',y='asw_group',hue='param_opt_val',col='params_opt',
           data=res,kind='swarm',height=2,aspect=1.5)

# %% [markdown]
# C: it desnt seem that same seed would have same effect on different params

# %% [markdown]
# #### Loss vs integration scores

# %%
# Plot relationship between training losses and metrics
metrics_plot_loss=['ilisi_system','asw_group']
fig,ax=plt.subplots(1,2,figsize=(2*5,1*3))
for i,loss in enumerate(['hisEnd_loss']):
    res_loss_plot=[]
    for loss_type in ['train']:
        sub=res[metrics_plot_loss+[loss+'_'+loss_type]+['params_opt']]
        sub.rename({loss+'_'+loss_type:loss},axis=1,inplace=True)
        sub['loss_type']=loss_type
        res_loss_plot.append(sub)
    res_loss_plot=pd.concat(res_loss_plot)
    for j,metric in enumerate(metrics_plot_loss):
        ax_sub=ax[j]
        sb.scatterplot(x=metric,y=loss,hue='params_opt',data=res_loss_plot, ax=ax_sub)
        if i==0 and j==1:
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        else:
            ax_sub.get_legend().remove()
plt.tight_layout()

# %% [markdown]
# C: The final loss does not correspond to integration metrics (even within parameter group). TODO would need to check how loss corresponds to integration metric within run - as between runs with different params it may not be comparable.

# %%
# Plot relationship between training losses and metrics
metrics_plot_loss=['ilisi_system','asw_group']
fig,ax=plt.subplots(1,2,figsize=(2*5,1*3))
for i,loss in enumerate(['hisEnd_reconstruction_loss']):
    res_loss_plot=[]
    for loss_type in ['train']:
        sub=res[metrics_plot_loss+[loss+'_'+loss_type]+['params_opt']]
        sub.rename({loss+'_'+loss_type:loss},axis=1,inplace=True)
        sub['loss_type']=loss_type
        res_loss_plot.append(sub)
    res_loss_plot=pd.concat(res_loss_plot)
    for j,metric in enumerate(metrics_plot_loss):
        ax_sub=ax[j]
        sb.scatterplot(x=metric,y=loss,hue='params_opt',data=res_loss_plot, ax=ax_sub)
        if i==0 and j==1:
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        else:
            ax_sub.get_legend().remove()
plt.tight_layout()

# %% [markdown]
# C: Relationship between bio preserv and reconstruction loss may be even oposite ...

# %% [markdown]
# Relationship between losses and ilisi for single selected weight setting per selected params.

# %%
# Subset to reruns of one param-weight combination
res_sub=[]
for param,val in {'z_distance_cycle_weight_nll':2, 
                  'z_distance_cycle_weight_kl':1, 
                  'z_distance_cycle_weight':50, 
                  'z_distance_cycle_weight_std':5,
                  'kl_weight':5, 
                  'vamp':10 }.items():
    res_sub.append(res.query('params_opt==@param & param_opt_val==@val'))
res_sub=pd.concat(res_sub)
res_sub['params_opt']=res_sub['params_opt'].astype(str)

# %%
for l in ['z_distance_cycle','kl_local']:
    for sub in ['train','validation']:
        sb.relplot(x=f'hisEnd_{l}_{sub}',y='ilisi_system',
           data=res_sub,col='params_opt',facet_kws=dict(sharex=False),
           height=3,aspect=1.2,s=40)

# %% [markdown]
# C: The optimised loss (KL or z distance cycle) simpli do not correspond to the integration metric. Maybe 5 points is not enough to see that but if the patterns was strong one would need to be able to see it.

# %% [markdown]
# Correlation between metric and final loss accross restarts with same seed

# %%
cor=pd.concat([
    res.groupby(['params_opt','param_opt_val']
           ).apply(lambda x: np.corrcoef(
    x[['hisEnd_loss_train','ilisi_system']].T.values.astype(float))[0,1]
                  ).rename('ilisi_system'),
    res.groupby(['params_opt','param_opt_val']
           ).apply(lambda x: np.corrcoef(
    x[['hisEnd_loss_train','asw_group']].T.values.astype(float))[0,1]
                  ).rename('asw_group')],axis=1)

# %%
rcParams['figure.figsize']=(0.7,7)
sb.heatmap(cor,cmap='coolwarm',vmin=-1,vmax=1,)
plt.xticks(rotation=90) 

# %% [markdown]
# C: Ideal would be to see negative correlation between loss and metrics, but not really the case.

# %% [markdown]
# How does loss change accross random seeds?

# %%
for loss in ['loss','reconstruction_loss','kl_local','z_distance_cycle']:
    g=sb.catplot( x='param_opt_val', y=f"hisEnd_{loss}_train",  col='params_opt',
           kind="swarm", data=res,sharex=False,sharey=False,
          height=2.5,aspect=1.3,color='k')

# %%
for loss in ['loss','reconstruction_loss','kl_local','z_distance_cycle']:
    g=sb.catplot( x='param_opt_val', y=f"hisEnd_{loss}_validation",  col='params_opt',
           kind="swarm", data=res,sharex=False,sharey=False,
          height=2.5,aspect=1.3,color='k')

# %% [markdown]
# C: The loss does not change so much accross seeds. 
#
# C: High z dist cycle loss reduces reconstruction and slightly reduces kl.
#
# C: Similar on train and val data - makse sense as similar cells

# %% [markdown]
# Was thinking in also looking into loss associated with integration (kl, zdist) and integration metric, maybe per param setting. But this may not work as if latent space changes this loss also gets a different value, so it is not really comparable. E.g. zdist may be small because integration is good or because it disabled many units/pushed them to 0. Similar for kl.

# %% [markdown]
# #### Latent space distn

# %%
# Dont import above as time consuming 
import scanpy as sc
from scipy.stats import entropy

# %%
latent_entropies = {}
latent_corrs = {}
for run in res.index:
    embed=sc.read(path_integration+run+'/embed.h5ad')
    # Subset to single system, use human as more cells in some subtypes
    cells= embed.obs.system=='1'
    embed_sub=embed[cells,:]
    # Minmax scale the embed comonents so that perceived var is not affected by range
    x=pd.DataFrame(minmax_scale(embed_sub.X),index=embed_sub.obs_names)
    x['cell_type']=embed_sub.obs['cell_type_final']
    x=x.groupby('cell_type').mean()
    latent_entropies[run]=entropy(x)
    latent_corrs[run]=np.corrcoef(x.T)[np.triu_indices(x.shape[1], k=1)]

# %%
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    
    for metric,metric_data in [('entropy',latent_entropies),('corr',latent_corrs)]:
        nrow=res.params_opt.nunique()
        ncol=res.groupby('params_opt')['param_opt_val'].nunique().max()
        fig,axs=plt.subplots(nrow,ncol,figsize=(ncol*3,nrow*2),sharey=True)
        for j,param in enumerate(sorted(res.params_opt.unique())):
            for i,val in enumerate(sorted(res.query('params_opt==@param')['param_opt_val'].unique())):
                runs=res.query('params_opt==@param & param_opt_val==@val')
                plot_df=[]
                for run,run_data in runs.iterrows():
                    plot_df.append(pd.DataFrame({
                        metric:metric_data[run],
                        'rep':len(metric_data[run])*[run_data['name']]
                                   }))
                plot_df=pd.concat(plot_df,axis=0)
                plot_df['rep']=pd.Categorical(plot_df['rep'],
                                              sorted(plot_df['rep'].unique()),True)
                ax=axs[j,i]
                sb.swarmplot(x='rep',y=metric,data=plot_df,ax=axs[j,i],s=3)
                ax.set_title(param.replace('_weight','')+' '+str(val))
        fig.tight_layout()


# %% [markdown]
# Save data for later in different var name

# %%
# Save for later
res_integration=res.copy()

# %% [markdown]
# ## Translation

# %%
path_translation=path_eval+'presFeb23/translation/'

# %%
# Load integration results - params and metrics
res=[]
for run in glob.glob(path_translation+'*/'):
    if os.path.exists(run+'args.pkl') and os.path.exists(run+'translation_correlation.tsv'):
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
res['dataset']=res['path_adata'].map({
    '/lustre/groups/ml01/workspace/karin.hrovatin/data/cross_species_prediction/pancreas_example/combined_orthologues.h5ad':'pancreas',
    '/lustre/groups/ml01/workspace/karin.hrovatin/data/cross_species_prediction/pancreas_example/combined_tabula_orthologues.h5ad':'pancreas+tabula'})

# %%
res.params_opt.unique()

# %%
#  Param that was optimised
res['param_opt_col']=res.params_opt.replace(
    {'vamp':'n_prior_components',
     'z_distance_cycle_weight_std':'z_distance_cycle_weight',
    'z_distance_cycle_weight_nll':'z_distance_cycle_weight',
    'z_distance_cycle_weight_kl':'z_distance_cycle_weight',
     'vamp_sys':'prior_components_system',
    'nn_size':None})
res['param_opt_val']=res.apply(
    lambda x: x[x['param_opt_col']] if x['param_opt_col'] is not None else 0,axis=1)

# %%
res['params_opt']=pd.Categorical(res['params_opt'],sorted(res['params_opt'].unique()), True)
res['dataset']=pd.Categorical(res['dataset'],sorted(res['dataset'].unique()), True)

# %%
sb.catplot( x='param_opt_val', y="ref-pred_ref",  col='params_opt',row='dataset',
           kind="swarm", data=res,sharex=False,
          height=2.5,aspect=1.3,margin_titles=True,color='k')

# %% [markdown]
# C: high z dist cycle weight with MSE_std reduces self translation. 

# %%
g=sb.catplot( x='param_opt_val', y="query-pred_query",  col='params_opt',row='dataset',
           kind="swarm", data=res,sharex=False,
          height=2.5,aspect=1.3,margin_titles=True, color='k')
for ax in g.axes.ravel():
    ax.axhline(0.8,lw=0.5,c='gray')
    ax.axhline(0.6,lw=0.5,c='gray')

# %% [markdown]
# C: Mixup helps translation
#
# C: Vamp prior helps translation. 
#
# C: Increasing whole network size does not help.
#
# C: While z dist cycle with MSE distance reduces translation the one with standardised MSE may even improve it.
#
# C: Sample_feature var may help in some cases, but not all - harms tabula.

# %% [markdown]
# Look at training of z_dist_cycle w=100 as it seems relatively unstable accross seeds

# %%
for run in res.query(
    'dataset=="pancreas" & param_opt_val==100 & params_opt=="z_distance_cycle_weight"'
).sort_values('ref-pred_ref').index:
    print(run)
    with plt.rc_context({'figure.figsize':(40,10)}):
        path_run=path_translation+run+'/'
        for img_fn in ['losses.png']:
            img = mpimg.imread(path_run+img_fn)
            imgplot = plt.imshow(img)
            plt.show()

# %% [markdown]
# Also look at reconstruction mixup weight which was more stable

# %%
for run in res.query(
    'dataset=="pancreas" & param_opt_val==100 & params_opt=="reconstruction_mixup_weight"'
).sort_values('ref-pred_ref').index:
    print(run)
    with plt.rc_context({'figure.figsize':(40,10)}):
        path_run=path_translation+run+'/'
        for img_fn in ['losses.png']:
            img = mpimg.imread(path_run+img_fn)
            imgplot = plt.imshow(img)
            plt.show()

# %% [markdown]
# C: It may be that to predict query well one needs to be less good at predicting ref.

# %%
sb.relplot( x='ref-pred_ref', y="query-pred_query",  hue='params_opt',col='dataset',
           kind="scatter", data=res,
          height=2.5,aspect=1,s=5)

# %%
res_translation=res.copy()

# %% [markdown]
# C: in final performance prediction of ref and query correlate, maybe the above loss plots hold within run's training and not accross runs

# %% [markdown]
# ### Integration vs translation
# Compare integration and translation metric (avg across runs with same param setting)

# %%
# Avg metric accross runs per param setting
res_integration_metricavg=res_integration.groupby(['params_opt','param_opt_val'],
                        observed=True)['ilisi_system'].mean()
res_bio_metricavg=res_integration.groupby(['params_opt','param_opt_val'],
                        observed=True)['asw_group'].mean()
res_translation_metricavg=res_translation.query('dataset=="pancreas"'
                                               ).groupby(['params_opt','param_opt_val'],
                        observed=True)['query-pred_query'].mean()

# %%
res_integr_transl_metricavg=pd.concat(
    [res_integration_metricavg,res_bio_metricavg,
     res_translation_metricavg],axis=1).reset_index().dropna(axis=0)
#res_integr_transl_metricavg['query-pred_query']=\
#    res_integr_transl_metricavg['query-pred_query'].astype(float)

# %%
rcParams['figure.figsize']=(2,2)
g=sb.scatterplot(x='ilisi_system',y='query-pred_query',
               hue='params_opt',
               data=res_integr_transl_metricavg, palette='colorblind')
sb.move_legend(g, loc='upper right',bbox_to_anchor=(3, 1))

# %%
rcParams['figure.figsize']=(2,2)
g=sb.scatterplot(x='asw_group',y='query-pred_query',
               hue='params_opt',
               data=res_integr_transl_metricavg, palette='colorblind')
sb.move_legend(g, loc='upper right',bbox_to_anchor=(3, 1))

# %%
rcParams['figure.figsize']=(2,2)
g=sb.scatterplot(x='asw_group',y='ilisi_system',
               hue='query-pred_query',style='params_opt',
               data=res_integr_transl_metricavg)
sb.move_legend(g, loc='upper right',bbox_to_anchor=(3, 1))

# %% [markdown]
# ### Pretraining

# %%
path_translation=path_eval+'pretrain/translation/'

# %%
# Load integration results - params and metrics
res_pt=[]
for run in glob.glob(path_translation+'*/'):
    if os.path.exists(run+'args.pkl') and os.path.exists(run+'translation_correlation.tsv'):
        args=pd.Series(vars(pkl.load(open(run+'args.pkl','rb'))))
        transl_corr=pd.read_table(run+'translation_correlation.tsv',index_col=0)
        transl_corr=pd.Series({'ref-pred_ref':transl_corr.at['ref','pred_ref'],
                          'query-pred_query':transl_corr.at['query','pred_query'],
                              'ref-pred_query':transl_corr.at['ref','pred_query'],
                              'query-pred_ref':transl_corr.at['query','pred_ref'],})
        data=pd.concat([args,transl_corr])
        data.name=run.split('/')[-2]
        res_pt.append(data)
res_pt=pd.concat(res_pt,axis=1).T

# %%
# Add integration res from above
# !!!! Params that match are manually slected!
res_sub=res_translation.query(
    'dataset=="pancreas+tabula" & params_opt=="kl_weight" & kl_weight==1'
).copy()

# %%
res_sub['pretrain']=False
res_pt['pretrain']=True
res_comb=pd.concat([res_pt,res_sub])

# %%
rcParams['figure.figsize']=(2,2)
sb.swarmplot(x='pretrain',y='query-pred_query',data=res_comb)

# %%
rcParams['figure.figsize']=(2,2)
sb.swarmplot(x='pretrain',y='ref-pred_ref',data=res_comb)

# %% [markdown]
# ## SWA

# %% [markdown]
# ### Integration

# %%
path_integration=path_eval+'swa/integration/'

# %%
# Load integration results - params and metrics
res_swa=[]
for run in glob.glob(path_integration+'*/'):
    if os.path.exists(run+'args.pkl') and os.path.exists(run+'scib_metrics.pkl'):
        args=pd.Series(vars(pkl.load(open(run+'args.pkl','rb'))))
        metrics=pd.Series(pkl.load(open(run+'scib_metrics.pkl','rb')))
        data=pd.concat([args,metrics])
        data.name=run.split('/')[-2]
        res_swa.append(data)
res_swa=pd.concat(res_swa,axis=1).T

# %%
res_swa['params_opt']=res_swa.apply(lambda x: 'swa_'+x['optimizer']+'_'+str(x['swa_lr']),axis=1)

# %%
# Add integration res from above
# !!!! Params that match are manually slected!
res_sub=res_integration.query('params_opt=="z_distance_cycle_weight" & z_distance_cycle_weight==10').copy()

# %%
res_sub['params_opt']=res_sub.apply(lambda x: x['optimizer']+'_'+str(x['lr']),axis=1)

# %%
res_comb=pd.concat([res_swa,res_sub])

# %%
res_comb['params_opt']=pd.Categorical(res_comb['params_opt'],
                                      sorted(res_comb['params_opt'].unique()), ordered=True)

# %%
rcParams['figure.figsize']=(2,2)
sb.swarmplot(x='params_opt',y='ilisi_system',data=res_comb, color='k')
_=plt.xticks(rotation=90)

# %%
rcParams['figure.figsize']=(2,2)
sb.swarmplot(x='params_opt',y='asw_group',data=res_comb, color='k')
_=plt.xticks(rotation=90)

# %% [markdown]
# C: SWA does not seem to help much with integration.

# %% [markdown]
# ### Translation

# %%
path_translation=path_eval+'swa/translation/'

# %%
# Load integration results - params and metrics
res_swa=[]
for run in glob.glob(path_translation+'*/'):
    if os.path.exists(run+'args.pkl') and os.path.exists(run+'translation_correlation.tsv'):
        args=pd.Series(vars(pkl.load(open(run+'args.pkl','rb'))))
        transl_corr=pd.read_table(run+'translation_correlation.tsv',index_col=0)
        transl_corr=pd.Series({'ref-pred_ref':transl_corr.at['ref','pred_ref'],
                          'query-pred_query':transl_corr.at['query','pred_query'],
                              'ref-pred_query':transl_corr.at['ref','pred_query'],
                              'query-pred_ref':transl_corr.at['query','pred_ref'],})
        data=pd.concat([args,transl_corr])
        data.name=run.split('/')[-2]
        res_swa.append(data)
res_swa=pd.concat(res_swa,axis=1).T

# %%
res_swa['params_opt']=res_swa.apply(lambda x: 'swa_'+x['optimizer']+'_'+str(x['swa_lr']),axis=1)

# %%
# Add integration res from above
# !!!! Params that match are manually slected!
res_sub=res_translation.query(
    'dataset=="pancreas" & params_opt=="z_distance_cycle_weight" & z_distance_cycle_weight==10'
).copy()

# %%
res_sub['params_opt']=res_sub.apply(lambda x: x['optimizer']+'_'+str(x['lr']),axis=1)

# %%
res_comb=pd.concat([res_swa,res_sub])

# %%
res_comb['params_opt']=pd.Categorical(res_comb['params_opt'],
                                      sorted(res_comb['params_opt'].unique()), ordered=True)

# %%
rcParams['figure.figsize']=(2,2)
sb.swarmplot(x='params_opt',y='query-pred_query',data=res_comb, color='k')
_=plt.xticks(rotation=90)

# %%
rcParams['figure.figsize']=(2,2)
sb.swarmplot(x='params_opt',y='ref-pred_ref',data=res_comb, color='k')
_=plt.xticks(rotation=90)

# %% [markdown]
# C: SWA may help with translation score & stability with lower learning rate.

# %%
