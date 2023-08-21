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
# Seed must be given as seed_num not to clash?
def run(eval_type:str,
        conda_env:str,
        model:str=None, 
        name:str=None, 
        seed_num:str=None, 
        log_on_epoch:str=None, 
        params_opt:str=None,
        path_adata:str=None, 
        path_adata_2:str=None, 
        fn_expr:str=None, 
        fn_moransi:str=None,
        gene_graph:str=None,
        saturn_emb:str=None,
        saturn_code:str=None,
        path_save:str=None, 
        system_key:str=None, 
        system_values:str=None,
        system_translate:str=None, 
        group_key:str=None, 
        group_translate:str=None, 
        batch_key:str=None, 
        pca_key:str=None,
        cluster_key:str=None,
        var_name_keys:str=None,
        cells_eval:str=None, 
        genes_eval:str=None, 
        pretrain_key:str=None, 
        pretrain_value:str=None, 
        train_size:str=None,
        mixup_alpha:str=None, 
        system_decoders:str=None, 
        out_var_mode:str=None,
        prior:str=None, 
        n_prior_components:str=None, 
        prior_components_system:str=None,
        z_dist_metric:str=None,
        n_latent:str=None,
        n_layers:str=None, 
        n_hidden:str=None,
        max_epochs:str=None,
        max_epochs_pretrain:str=None, 
        epochs_detail_plot:str=None,
        use_n_steps_kl_warmup:str=None,
        kl_weight:str=None, 
        kl_cycle_weight:str=None, 
        reconstruction_weight:str=None, 
        reconstruction_mixup_weight:str=None, 
        reconstruction_cycle_weight:str=None, 
        z_distance_cycle_weight:str=None, 
        translation_corr_weight:str=None, 
        z_contrastive_weight:str=None, 
        testing:str=None,
        optimizer:str=None,
        lr:str=None,
        reduce_lr_on_plateau:str=None,
        lr_scheduler_metric:str=None,
        lr_patience:str=None,
        lr_factor:str=None,
        lr_min:str=None,
        lr_threshold_mode:str=None,
        lr_threshold:str=None,
        swa:str=None,
        swa_lr:str=None,
        swa_epoch_start:str=None,
        swa_annealing_epochs:str=None,
        rel_gene_weight:str=None,
        lam_data:str=None,
        lam_kl:str=None,
        lam_graph:str=None,
        lam_align:str=None,
        n_cells_eval:str=None,
        hv_span:str=None,
        hv_genes:str=None,
        num_macrogenes:str=None,
        pe_sim_penalty:str=None,
       ):
    
    params_info=locals()
    params_info['seed']=params_info['seed_num']
    del params_info['seed_num']

    logging.info('Received the following configuration:')
    logging.info(str(params_info))
    #print('All params:')
    #print(params_info)
    
    # Prepare args for running script
    args=[]
    for k,v in params_info.items():
        if k!='eval_type' and k!='model' and v is not None:
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
    
    # Run integration script
    
    # Script to run - based on eval_type
    #script_dir='/lustre/groups/ml01/code/karin.hrovatin/cross_species_prediction/notebooks/eval/'
    model_fn_part='_'+model if model is not None else ''
    #script=f'{script_dir}eval_{eval_type}{model_fn_part}.py' 
    script=f'run_{eval_type}{model_fn_part}.py' 
    logging.info('Running integration script')
    logging.info(script)
    
    # Send stderr to stdout and stdout pipe to output to save in log 
    # (print statements from the child script would else not be saved)
    process_integration = subprocess.Popen('conda run -n'.split()+[conda_env, 'python',script]+args, 
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    # Make sure that process has finished
    res=process_integration.communicate()
    # Save stdout from the child script
    for line in res[0].decode(encoding='utf-8').split('\n'):
        if line.startswith('PATH_SAVE='):
            path_save=line.replace('PATH_SAVE=','').strip(' ')
        logging.info(line)
    # Check that child process did not fail - if this was not checked then
    # the status of the whole job would be succesfull 
    # even if the child failed as error wouldn be passed upstream
    if process_integration.returncode != 0:
        raise ValueError('Process integration failed with', process_integration.returncode)
    
    # Run neighbours script
    logging.info('Run neighbours script')

    args_neigh=[
        '--path',path_save,
        '--system_key',system_key,
        '--group_key',group_key,
        '--batch_key',batch_key,
        '--testing',str(testing) if testing is not None else '0',
    ]
    process_neigh = subprocess.Popen(['python','run_neighbors.py']+args_neigh, 
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    # Make sure that process has finished
    res=process_neigh.communicate()
    # Save stdout from the child script
    for line in res[0].decode(encoding='utf-8').split('\n'):
         logging.info(line)
    # Check that child process did not fail - if this was not checked then
    # the status of the whole job would be succesfull 
    # even if the child failed as error wouldn be passed upstream
    if process_neigh.returncode != 0:
        raise ValueError('Process neighbours failed with', process_neigh.returncode)

    
    # Integration metrics

    logging.info('Run integration metrics script')

    args_metrics=[
            '--path',path_save,
            '--system_key',system_key,
            '--group_key',group_key,
            '--batch_key',batch_key,
            '--fn_expr',fn_expr if fn_expr is not None else path_adata,
            '--fn_moransi',fn_moransi,
            '--testing',str(testing) if testing is not None else '0',
        ]
    for scaled in ['0','1']:
        logging.info('Computing metrics with param scaled='+scaled)
        args_metrics_sub=args_metrics.copy()
        args_metrics_sub.extend(['--scaled',scaled])
        process_metrics = subprocess.Popen(['python','run_metrics.py']+args_metrics_sub, 
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        # Make sure that process has finished
        res=process_metrics.communicate()
        # Save stdout from the child script
        for line in res[0].decode(encoding='utf-8').split('\n'):
             logging.info(line)
        # Check that child process did not fail - if this was not checked then
        # the status of the whole job would be succesfull 
        # even if the child failed as error wouldn be passed upstream
        if process_metrics.returncode != 0:
            raise ValueError('Process failed with', process_metrics.returncode)
    
    logging.info('Finished wrapper script!')
