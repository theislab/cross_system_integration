# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.3
#   kernelspec:
#     display_name: csi
#     language: python
#     name: csi
# ---

# %% [markdown]
# # Evaluate runtime

# %%
import scanpy as sc
import pickle as pkl
import pandas as pd
import numpy as np
# Make random number for seed before scvi import sets seed to 0
seed=np.random.randint(0,1000000)
import argparse
import os
import pathlib
import string
import subprocess
import scvi
import gc
import time
import torch


from memory_profiler import memory_usage
from datetime import datetime
from matplotlib.pyplot import rcParams
import matplotlib.pyplot as plt
import seaborn as sns

from cross_system_integration.model._xxjointmodel import XXJointModel
import pytorch_lightning as pl

# Otherwise the seed remains constant
from scvi._settings import ScviConfig
config=ScviConfig()
config.seed=seed

# %%
import wandb
from pytorch_lightning.loggers.wandb import WandbLogger

# %%
path_adata = '/om2/user/khrovati/data/cross_system_integration/pancreas_conditions_MIA_HPAP2/combined_orthologuesHVG.h5ad'
system_key = 'system'
batch_key = 'batch'

# %%
config.seed=1

# %% [markdown]
# ## Integration

# %% [markdown]
# ### Prepare data

# %%
# Load data
adata=sc.read(path_adata)

# %% [markdown]
# ### Training

# %%
# ! nvidia-smi

# %%

# %%
details_filename = pathlib.Path('/home/moinfar/io/csi/runtime/details.pkl')
results_filename = pathlib.Path('/home/moinfar/io/csi/runtime/results.csv')

# %%
if details_filename.exists():
    details = pkl.load(open(details_filename, 'rb'))
else:
    details = {}

# %%
api = wandb.Api()
wandb_project = f"sysvi_measure_runtime_etc"

# %%
results = []
max_epochs = 50

for n_samples in [10_000, 50_000, 100_000, 200_000, 300_000, 450_000]:
    for model_name in ['scVI', 'sysVI']:
        run_key = f"{model_name}#{n_samples}#{max_epochs}"
        
        if run_key in details:
            continue
        adata_training = sc.pp.subsample(adata, n_obs=n_samples, random_state=0, copy=True)

        wandb.finish()
        wb_logger = WandbLogger(project=wandb_project, name=run_key, save_dir='/home/moinfar/tmp/wandb',
                                reinit=True, settings=wandb.Settings(start_method="fork"),
                                config={'n_samples': n_samples, 'model_name': model_name,
                                        'max_epochs': max_epochs, 'path_adata': path_adata})

        gc.collect()
        time.sleep(5)
        torch.cuda.empty_cache()
        mem_before = torch.cuda.max_memory_allocated() / (1024 * 1024)  # in MB
        train_start = datetime.now()
        def run_training(adata_training, system_key, batch_key):
            if model_name == 'sysVI':
                # Setup adata
                adata_training = XXJointModel.setup_anndata(
                    adata=adata_training,
                    system_key=system_key,
                    group_key=None,
                    categorical_covariate_keys=[batch_key],
                )
                
                # Define pseudoinputs
                n_prior_components=5
                pdi=None
                
                # Train
                model = XXJointModel(
                    adata=adata_training,
                    out_var_mode='feature',
                    mixup_alpha=None,
                    system_decoders=False,
                    prior='vamp', 
                    n_prior_components=n_prior_components,
                    pseudoinputs_data_init=True,
                    pseudoinputs_data_indices=pdi,
                    trainable_priors=True,
                    encode_pseudoinputs_on_eval_mode=True,
                    z_dist_metric = 'MSE_standard',
                    n_layers=2,
                    n_hidden=256,
                )
                model.train(max_epochs=max_epochs,
                            log_every_n_steps=1,
                            check_val_every_n_epoch=1,
                            val_check_interval=1.0 if True else 1,  # Assuming log_on_epoch is True by default
                            train_size=0.9,            
                            callbacks=[pl.callbacks.LearningRateMonitor(logging_interval='epoch')] +\
                                [pl.callbacks.StochasticWeightAveraging(
                                    swa_lrs=0.001, 
                                    swa_epoch_start=10, 
                                    annealing_epochs=10)] 
                                if False else [],  # Assuming swa is False by default
                            plan_kwargs={
                                'optimizer': "Adam",
                                'lr': 0.001,
                                'reduce_lr_on_plateau': False,
                                'lr_scheduler_metric': 'loss_train',
                                'lr_patience': 5.,
                                'lr_factor': 0.1,
                                'lr_min': 1e-7,
                                'lr_threshold_mode': 'rel',
                                'lr_threshold': 0.1,
                                'log_on_epoch': True,
                                'log_on_step': False,
                                'loss_weights':{
                                   'kl_weight': 1.,
                                   'kl_cycle_weight': 0.,
                                   'reconstruction_weight': 1.,
                                   'reconstruction_mixup_weight': 0.,
                                   'reconstruction_cycle_weight': 0.,
                                   'z_distance_cycle_weight': 0.,
                                   'translation_corr_weight': 0.,
                                   'z_contrastive_weight': 0.,
                               
                           }}, logger=[wb_logger])
            elif model_name == 'scVI':
                scvi.model.SCVI.setup_anndata(
                    adata_training, 
                    layer="counts", 
                    batch_key=system_key,
                    categorical_covariate_keys=[batch_key])
                model = scvi.model.SCVI(adata_training, 
                            n_layers=2, n_hidden=256, n_latent=15, 
                            gene_likelihood="nb")
                
                model.train(
                    max_epochs = max_epochs,
                    # Uses n_steps_kl_warmup (128*5000/400) if n_epochs_kl_warmup is not set
                    plan_kwargs=dict(
                        n_epochs_kl_warmup = None,
                        n_steps_kl_warmup = 1600,
                        max_kl_weight=1.,
                    ),
                    log_every_n_steps=1,
                    check_val_every_n_epoch=1,
                    callbacks=[pl.callbacks.LearningRateMonitor(logging_interval='epoch')],
                    logger=[wb_logger],
                )
        mem_usage = memory_usage((run_training, (adata_training, system_key, batch_key), {}), interval=1, timeout=None, max_usage=True)
        runtime = (datetime.now() - train_start).total_seconds()
        mem_after = torch.cuda.max_memory_allocated() / (1024 * 1024)  # in MB
        cuda_mem_usage = mem_after - mem_before
        results.append((model_name, n_samples, runtime, mem_usage, cuda_mem_usage))
        wb_logger.log_hyperparams({'runtime': runtime})
        wb_logger.log_hyperparams({'mem_usage': mem_usage})
        wb_logger.log_hyperparams({'cuda_mem_usage': cuda_mem_usage})
        wandb.finish()
        details[run_key] = results[-1]
        with open(details_filename, 'wb') as f:
            pkl.dump(details, f)
        print(results[-1])

# %%

# %%
result_df = pd.DataFrame(results, columns=['model', 'n_samples', 'runtime', 'mem_usage', 'cuda_mem_usage'])
result_df.to_csv(results_filename)

# %%
result_df

# %%

# %%
result_df = pd.read_csv(results_filename, index_col=0)
result_df

# %%
import pandas as pd 
import wandb
api = wandb.Api()
api.flush()

# Project is specified by <entity/project-name>
runs = api.runs("moinfar_proj/sysvi_measure_runtime_etc")

result_list = []
for run in runs: 
    system_metrics = run.history(stream='systemMetrics').max()
    system_metrics.index = system_metrics.index.str.replace('\\.', '_')
    result_list.append({
        'name': run.name,
        'summary': run.summary._json_dict,
        'config': {k: v for k,v in run.config.items() if not k.startswith('_')},
        'system_metrics': system_metrics.to_dict(),
    })

runs_df = pd.json_normalize(result_list, sep='_')
runs_df

# %%
runs_df.columns

# %% [markdown]
# ## Plotting

# %%
# Define the order for the x-axis (n_samples)
x_order = list(sorted(runs_df.drop_duplicates(subset=['config_n_samples'])['config_n_samples']))

# Convert runtime to minutes
runs_df['n_samples'] = runs_df['config_n_samples']
runs_df['runtime_minutes'] = runs_df['config_runtime'] / 60

lineplot = sns.lineplot(
    data=runs_df,
    x='n_samples',
    y='runtime_minutes',
    hue='config_model_name',
    hue_order=sorted(runs_df['config_model_name'].unique()),
    palette='tab10',
    alpha=1.0,  # Transparency of the lines
    marker='o',  # Add markers at the data points
    markersize=8,  # Size of the markers
    linewidth=2,  # Thickness of the lines
)

# Overlay scatter plot for better visibility of points
# scatterplot = sns.scatterplot(
#     data=result_df,
#     x='n_samples',
#     y='runtime_minutes',
#     hue='model',
#     hue_order=sorted(result_df['model'].unique()),
#     palette='tab10',
#     s=100,  # Size of the points
#     alpha=0.8,  # Transparency of the points
#     legend=False,  # Avoid duplicate legend entries
# )

# Customize the plot
plt.xlabel('Number of samples', fontsize=12)
plt.ylabel('Runtime (Minutes)', fontsize=12)
plt.title(f'Runtime for VAMP+CYC versus scVI on different\n subsamples of the pancreas data (NVIDIA A100 GPU)', fontsize=14)
plt.xticks(fontsize=10, rotation=90, ha='center')
plt.yticks(fontsize=10)
plt.legend(title='Model', fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.3)

# Show the plot
plt.tight_layout()
plt.savefig("/home/moinfar/io/csi/figures/runtime_of_sysvi_vs_scvi_pancreas_scatter.pdf", bbox_inches='tight')
plt.savefig("/home/moinfar/io/csi/figures/runtime_of_sysvi_vs_scvi_pancreas_scatter.png", bbox_inches='tight')
plt.show()

# %%
runs_df = runs_df.sort_values("config_runtime", ascending=False)
if x_order is None:
    x_order = list(sorted(runs_df.drop_duplicates(subset=['config_n_samples'])['config_n_samples']))

runs_df['n_samples'] = runs_df['config_n_samples']
runs_df['runtime_ps_minutes'] = runs_df['config_runtime'] * 1000 / runs_df['n_samples'] / max_epochs

plt.figure(figsize=(7, 5))
barplot = sns.barplot(
    data=runs_df,
    x='n_samples',
    y='runtime_ps_minutes',
    order=x_order,
    hue='config_model_name',
    hue_order=sorted(runs_df['config_model_name'].unique()),
    dodge=True,  # Avoid gaps by overlapping bars slightly
    palette='tab10'
)

# Customize the plot
plt.xlabel('Number of samples', fontsize=12)
plt.ylabel('Runtime of one iteration\nper sample (milliseconds)', fontsize=12)
plt.title(f'Runtime for VAMP+CYC versus scVI on different\n subsamples of the pancreas data (NVIDIA A100 GPU)', fontsize=14)
plt.xticks(fontsize=10, rotation=90, ha='center')
plt.yticks(fontsize=10)
plt.legend(title='Model', fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.3)

# Show the plot
plt.tight_layout()

plt.savefig("/home/moinfar/io/csi/figures/runtime_per_sample_of_sysvi_vs_scvi_pancreas.pdf", bbox_inches='tight')
plt.savefig("/home/moinfar/io/csi/figures/runtime_per_sample_of_sysvi_vs_scvi_pancreas.png", bbox_inches='tight')
plt.show()

# %%

# %%

# %%
# Define the order for the x-axis (n_samples)
if x_order is None:
    x_order = list(sorted(runs_df.drop_duplicates(subset=['config_n_samples'])['config_n_samples']))


# Convert runtime to minutes
runs_df['n_samples'] = runs_df['config_n_samples']
runs_df['mem_usage_gb'] = runs_df['config_mem_usage'] / 1024

lineplot = sns.lineplot(
    data=runs_df,
    x='n_samples',
    y='mem_usage_gb',
    hue='config_model_name',
    hue_order=sorted(runs_df['config_model_name'].unique()),
    palette='tab10',
    alpha=1.0,  # Transparency of the lines
    marker='o',  # Add markers at the data points
    markersize=8,  # Size of the markers
    linewidth=2,  # Thickness of the lines
)

# Customize the plot
plt.xlabel('Number of samples', fontsize=12)
plt.ylabel('Memory usage (GB)', fontsize=12)
plt.title(f'Momory requirement of VAMP+CYC versus scVI on different\n subsamples of the pancreas data', fontsize=14)
plt.xticks(fontsize=10, rotation=90, ha='center')
plt.yticks(fontsize=10)
plt.legend(title='Model', fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.ylim((0, runs_df['mem_usage_gb'].max() * 1.1))

# Show the plot
plt.tight_layout()
plt.savefig("/home/moinfar/io/csi/figures/mem_usage_of_sysvi_vs_scvi_pancreas_scatter.pdf", bbox_inches='tight')
plt.savefig("/home/moinfar/io/csi/figures/mem_usage_of_sysvi_vs_scvi_pancreas_scatter.png", bbox_inches='tight')
plt.show()

# %%
# Define the order for the x-axis (n_samples)
if x_order is None:
    x_order = list(sorted(runs_df.drop_duplicates(subset=['config_n_samples'])['config_n_samples']))


# Convert runtime to minutes
runs_df['n_samples'] = runs_df['config_n_samples']
runs_df['gpu_mem_usage_gb'] = runs_df['system_metrics_system_gpu_0_memoryAllocatedBytes'] / 1024 / 1024 / 1024

lineplot = sns.lineplot(
    data=runs_df,
    x='n_samples',
    y='gpu_mem_usage_gb',
    hue='config_model_name',
    hue_order=sorted(runs_df['config_model_name'].unique()),
    palette='tab10',
    alpha=1.0,  # Transparency of the lines
    marker='o',  # Add markers at the data points
    markersize=8,  # Size of the markers
    linewidth=2,  # Thickness of the lines
)

# Customize the plot
plt.xlabel('Number of samples', fontsize=12)
plt.ylabel('GPU memory usage (GB)', fontsize=12)
plt.title(f'GPU momory requirement of VAMP+CYC versus scVI on different\n subsamples of the pancreas data (NVIDIA A100 GPU)', fontsize=14)
plt.xticks(fontsize=10, rotation=90, ha='center')
plt.yticks(fontsize=10)
plt.legend(title='Model', fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.ylim((0, runs_df['gpu_mem_usage_gb'].max() * 1.1))

# Show the plot
plt.tight_layout()
plt.savefig("/home/moinfar/io/csi/figures/gpu_mem_usage_of_sysvi_vs_scvi_pancreas_scatter.pdf", bbox_inches='tight')
plt.savefig("/home/moinfar/io/csi/figures/gpu_mem_usage_of_sysvi_vs_scvi_pancreas_scatter.png", bbox_inches='tight')
plt.show()

# %%
