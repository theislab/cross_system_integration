# Seml run script for the eval experiments, runs individual eval scripts
# Used either with the seml.yaml or seml_test.yaml for testing out that everything works
import logging
from sacred import Experiment
import seml
import subprocess

# Set up experiment

ex = Experiment()
seml.setup_logger(ex)

@ex.post_run_hook
def collect_stats(_run):
    seml.collect_exp_stats(_run)

@ex.config
def config():
    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(seml.create_mongodb_observer(db_collection, overwrite=overwrite))

@ex.automain
def run(eval_type:str, name:str=None, params_opt:str=None,
        path_adata:str=None, path_save:str=None, system_key:str=None, 
        system_translate:str=None, group_key:str=None, group_translate:str=None, 
        batch_key:str=None, cells_eval:str=None, genes_eval:str=None, train_size:str=None,
        mixup_alpha:str=None, system_decoders:str=None, 
        prior:str=None, n_prior_components:str=None,
        n_layers:str=None, n_hidden:str=None,
        max_epochs:str=None, epochs_detail_plot:str=None,
        kl_weight:str=None, kl_cycle_weight:str=None, 
        reconstruction_weight:str=None, reconstruction_mixup_weight:str=None, 
        reconstruction_cycle_weight:str=None, z_distance_cycle_weight:str=None, 
        translation_corr_weight:str=None, z_contrastive_weight:str=None, testing:str=None,
        optimizer:str=None,lr:str=None,reduce_lr_on_plateau:str=None,
        lr_scheduler_metric:str=None,lr_patience:str=None,lr_factor:str=None,
        lr_min:str=None,lr_threshold_mode:str=None,lr_threshold:str=None):
    params_info={
                 "eval_type": eval_type,         
                 "name": name, 
                 "params_opt":params_opt,
                 "path_adata": path_adata, 
                 "path_save": path_save, 
                 "system_key": system_key, 
                 "system_translate": system_translate, 
                 "group_key": group_key, 
                 "group_translate": group_translate, 
                 "batch_key": batch_key, 
                 "cells_eval": cells_eval,
                 "genes_eval": genes_eval,
                 "train_size":train_size,
                 "mixup_alpha": mixup_alpha, 
                 "system_decoders": system_decoders, 
                 "prior":prior,
                 "n_prior_components":n_prior_components,
                 "n_layers":n_layers,
                 "n_hidden":n_hidden,
                 "max_epochs": max_epochs, 
                 "epochs_detail_plot": epochs_detail_plot,
                 "kl_weight": kl_weight, 
                 "kl_cycle_weight": kl_cycle_weight, 
                 "reconstruction_weight": reconstruction_weight, 
                 "reconstruction_mixup_weight": reconstruction_mixup_weight, 
                 "reconstruction_cycle_weight": reconstruction_cycle_weight, 
                 "z_distance_cycle_weight": z_distance_cycle_weight, 
                 "translation_corr_weight": translation_corr_weight, 
                 "z_contrastive_weight": z_contrastive_weight,
                
                'optimizer':optimizer,
                'lr':lr,
                'reduce_lr_on_plateau':reduce_lr_on_plateau,
                'lr_scheduler_metric':lr_scheduler_metric,
                'lr_patience':lr_patience,
                'lr_factor':lr_factor,
                'lr_min':lr_min,
                'lr_threshold_mode':lr_threshold_mode,
                'lr_threshold':lr_threshold,
        
                 "testing":testing,
    }
    logging.info('Received the following configuration:')
    logging.info(str(params_info))
    #print('All params:')
    #print(params_info)
    
    # Prepare args for running script
    args=[]
    for k,v in params_info.items():
        if k!='eval_type' and v is not None:
            # Integration does not have the translation specific args
            # For now also not params that define which genes/cells to use for eval
            if not (eval_type=='integration' and 
                    k in ['system_translate','group_translate',
                         'cells_eval','genes_eval']):
                # Set path save based on eval type
                # Expects that dirs were created before
                if k=='path_save':
                    v=v+eval_type+'/'
                args.append('--'+k)
                args.append(str(v))
    #print('Script args:')
    #print(args)
    logging.info('Using the following args for the script')
    logging.info(str(args))
    
    # Script to run - based on eval_type
    script_dir='/lustre/groups/ml01/code/karin.hrovatin/cross_species_prediction/notebooks/eval/'
    script=f'{script_dir}eval_{eval_type}.py'
    logging.info('Running script')
    logging.info(script)
    
    # Run eval script
    # Send stderr to stdout and stdout pipe to output to save in log 
    # (print statements from the child script would else not be saved)
    process = subprocess.Popen(['python',script]+args, 
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    # Make sure that process has finished
    res=process.communicate()
    # Save stdout from the child script
    for line in res[0].decode(encoding='utf-8').split('\n'):
         logging.info(line)
    # Check that child process did not fail - if this was not checked then
    # the status of the whole job would be succesfull 
    # even if the child failed as error wouldn be passed upstream
    if process.returncode != 0:
        raise ValueError('Process failed with', process.returncode)
        
    logging.info('Finished wrapper script!')
