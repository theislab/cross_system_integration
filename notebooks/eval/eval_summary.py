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

from matplotlib import rcParams
import matplotlib.pyplot as plt
import seaborn as sb
import matplotlib.image as mpimg

# %%
path_eval='/lustre/groups/ml01/workspace/karin.hrovatin/data/cross_species_prediction/eval/pancreas_example_v0/'

# %% [markdown]
# ## Integration

# %%
res=[]
for run in glob.glob(path_eval+'integration/*/'):
    args=pd.Series(vars(pkl.load(open(run+'args.pkl','rb'))))
    metrics=pd.Series(pkl.load(open(run+'scib_metrics.pkl','rb')))
    data=pd.concat([args,metrics])
    data.name=run.split('/')[-2]
    res.append(data)
res=pd.concat(res,axis=1).T

# %%
# Check which losses were modified
# Assumes that defaults of losses are int and modified float!!!!
# Also does not account for changes in mixup_alpha (assumes same) 
# and system_decoders - this one can be separately shown as different architecture
losses=['kl_weight', 
          'kl_cycle_weight', 
          'reconstruction_weight', 
          'reconstruction_mixup_weight', 
          'reconstruction_cycle_weight',
          'z_distance_cycle_weight', 
          'translation_corr_weight', 
          'z_contrastive_weight']
res['tested_losses']=res.apply(
    lambda x: '&'.join([loss.replace('_weight','') for loss in losses 
                        if not isinstance(x[loss],int)]), axis=1)
# If only one loss has been changed add here rank of tested loss values
loss_weight_ranks={}
for loss in losses:
    loss_values=list(sorted(set([v for v in  res[loss] if isinstance(v,float)])))
    loss_weight_ranks[loss]=dict(zip(loss_values,range(len(loss_values))))
# Adds loss weight rank if single loss was used, else adds 0
res['primary_loss_weight_rank']=res.apply(
    lambda x: loss_weight_ranks.get(x['tested_losses']+'_weight'
                                   )[x[x['tested_losses']+'_weight']]
    if x['tested_losses']+'_weight' in loss_weight_ranks else 0, axis=1
)
res['tested_losses']=pd.Categorical(res['tested_losses'],sorted(res['tested_losses'].unique()),
                                    ordered=True,)

# %%
res_plot=[]
for metric in ['ilisi_system','clisi','ilisi_batch_system-0','ilisi_batch_system-1']:
    res_plot_sub=res[
        ['tested_losses','primary_loss_weight_rank','system_decoders',metric]].copy()
    res_plot_sub['metric_type']=metric
    res_plot_sub.rename({metric:'metric'},axis=1,inplace=True)
    res_plot.append(res_plot_sub)
res_plot=pd.concat(res_plot)

# %%
sb.catplot(
    data=res_plot, sharex=False,
    y="tested_losses", x="metric", row="system_decoders",col='metric_type',
    hue='primary_loss_weight_rank',palette='cividis',
    kind="swarm", height=4, aspect=1.2,
)

# %% [markdown]
# C: clisi does not seem to distinguish well between different bio preservations in embeddings.
#
# C: z_distance_cycle has superior performance in cross species integration in single and double decoder settings, and is the only loss that strongly improves integration in separate species decoders. Interestingly increasing the weight too much leads to ilisi decline.
#
# C: reconstruction_cycle weight also helps with integration, also in the double decoder setting.
#
# C: interestingly, too high kl weight harms integration. This does not seem to be the case for kl_cycle weight (or not as much). Although this may be in part due to random variation (some other loss values vary greatly).

# %%
# Select runs to display in detail
runs_show={}
runs_show['high_kl']=res_plot.query(
    'tested_losses=="kl" & metric_type=="ilisi_system" &'+\
    'primary_loss_weight_rank==3 & system_decoders==False'
).sort_values('metric',ascending=False).index[0]
runs_show['low_kl']=res_plot.query(
    'tested_losses=="kl" & metric_type=="ilisi_system" &'+\
    'primary_loss_weight_rank==0 & system_decoders==False'
).sort_values('metric',ascending=False).index[0]
runs_show['high_kl_cycle']=res_plot.query(
    'tested_losses=="kl_cycle" & metric_type=="ilisi_system" &'+\
    'primary_loss_weight_rank==3 & system_decoders==False'
).sort_values('metric',ascending=False).index[0]
runs_show['low_kl_cycle']=res_plot.query(
    'tested_losses=="kl_cycle" & metric_type=="ilisi_system" &'+\
    'primary_loss_weight_rank==0 & system_decoders==False'
).sort_values('metric',ascending=False).index[0]
runs_show['reconstruction_cycle']=res_plot.query(
    'tested_losses=="reconstruction_cycle" & metric_type=="ilisi_system" &'+\
    'system_decoders==False'
).sort_values('metric',ascending=False).index[0]
runs_show['high_z_distance_cycle']=res_plot.query(
    'tested_losses=="z_distance_cycle" & metric_type=="ilisi_system" &'+\
    'primary_loss_weight_rank==3 & system_decoders==False'
).sort_values('metric',ascending=False).index[0]
runs_show['low_z_distance_cycle']=res_plot.query(
    'tested_losses=="z_distance_cycle" & metric_type=="ilisi_system" &'+\
    'primary_loss_weight_rank==0 & system_decoders==False'
).sort_values('metric',ascending=False).index[0]
runs_show['reconstruction_cycle_2D']=res_plot.query(
    'tested_losses=="reconstruction_cycle" & metric_type=="ilisi_system" &'+\
    'system_decoders==True'
).sort_values('metric',ascending=False).index[0]
runs_show['high_z_distance_cycle_2D']=res_plot.query(
    'tested_losses=="z_distance_cycle" & metric_type=="ilisi_system" &'+\
    'primary_loss_weight_rank==3 & system_decoders==True'
).sort_values('metric',ascending=False).index[0]
runs_show['low_z_distance_cycle_2D']=res_plot.query(
    'tested_losses=="z_distance_cycle" & metric_type=="ilisi_system" &'+\
    'primary_loss_weight_rank==0 & system_decoders==True'
).sort_values('metric',ascending=False).index[0]
for k,v in runs_show.items():
    print(k,v)

# %%
for run_info, run_dir in runs_show.items():
    print(run_info,run_dir)
    print(res.loc[run_dir,
                  ['ilisi_system','clisi','ilisi_batch_system-0','ilisi_batch_system-1']])
    with plt.rc_context({'figure.figsize':(40,10)}):
        path_run=path_eval+'integration/'+run_dir+'/'
        for img_fn in ['losses.png','umap.png']:
            img = mpimg.imread(path_run+img_fn)
            imgplot = plt.imshow(img)
            plt.show()

# %% [markdown]
# ## Translation

# %%
res=[]
for run in glob.glob(path_eval+'translation/*/'):
    args=pd.Series(vars(pkl.load(open(run+'args.pkl','rb'))))
    metrics=pd.Series(pkl.load(open(run+'scib_metrics.pkl','rb')))
    data=pd.concat([args,metrics])
    data.name=run.split('/')[-2]
    res.append(data)
res=pd.concat(res,axis=1).T

# %%
run='/lustre/groups/ml01/workspace/karin.hrovatin/data/cross_species_prediction/eval/pancreas_example_v0/translation/ST1GTbetaMANoneSDTrueKLW1KLCW10.0RW1RMW0RCW0ZDCW0TCW0ZCW0_QcOuTN0i/'

# %%
transl_corr=pd.read_table(run+'translation_correlation.tsv',index_col=0)
transl_corr=pd.Series({'ref-pred_ref':transl_corr.at['ref','pred_ref'],
                      'query-pred_query':transl_corr.at['query','pred_query']})
print(transl_corr)

# %%
