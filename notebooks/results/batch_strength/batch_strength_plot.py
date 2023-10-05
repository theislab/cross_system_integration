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
import numpy as np
import pandas as pd
import scanpy as sc
import pickle as pkl

import seaborn as sb
from matplotlib import rcParams
import matplotlib.pyplot as plt

# %%
path_data='/om2/user/khrovati/data/cross_system_integration/'
path_res_panc=path_data+'pancreas_conditions_MIA_HPAP2/'

# %%
path_fig=path_data+'figures/'

# %%
distances=pkl.load(open(path_res_panc+'combined_orthologuesHVG_PcaSysBatchDist.pkl','rb'))

# %%
# Prepare df for plotting
plot=[]
ct='delta'
dat=distances[ct]
for comparison,dist in dat.items():
    dist=pd.DataFrame(dist,columns=['dist'])
    dist['group']=ct
    dist['Comparison']=comparison
    plot.append(dist)
plot=pd.concat(plot)

# %%
# Make data prettier
plot.rename({'dist':'Distance'},axis=1,inplace=True)
plot.replace({'s0_within':'Mouse\n(within datasets)',
              's0_between':'Mouse\n(between datasets)',
              's1':'Human',
              's0s1':'Mouse and human'},inplace=True)

# %%
fig,ax=plt.subplots(figsize=(1.5,2))
sb.violinplot(y='Comparison',x='Distance',data=plot,inner=None,linewidth=0.5,ax=ax)
fig.set(facecolor = (0,0,0,0))
ax.set(facecolor = (0,0,0,0))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig(path_fig+f'batch_strength-pancreas_{ct}-violin.pdf',dpi=300,bbox_inches='tight')
plt.savefig(path_fig+f'batch_strength-pancreas_{ct}-violin.png',dpi=300,bbox_inches='tight')

# %%
# N comparisons per group
plot.groupby('Comparison').size()

# %%
