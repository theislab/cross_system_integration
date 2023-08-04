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
def run(path_save:str=None,subdir:str=None, fn_expr:str=None, fn_moransi:str=None,
        group_key:str=None,batch_key:str=None,system_key:str=None,
        scaled:str=None,cluster_optimized:str=None,
       ):
    params_info={
                 "path":path_save+subdir+'/',
                 "fn_expr": fn_expr, 
                 "fn_moransi": fn_moransi, 
                 "system_key": system_key, 
                 "group_key": group_key, 
                 "batch_key": batch_key, 
                 "scaled":scaled,
                 "cluster_optimized":cluster_optimized,
    }
    #print('All params:')
    #print(params_info)
    
    # Prepare args for running script
    args=[]
    for k,v in params_info.items():
        if v is not None:
            args.append('--'+k)
            args.append(str(v))
    #print('Script args:')
    #print(args)
    logging.info('Using the following args for the script')
    logging.info(str(args))
    
    # Script to run - based on eval_type
    script=f'run_metrics.py' 
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
