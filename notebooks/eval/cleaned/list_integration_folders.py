# List integration folders to copied to seml script for metrics of neighbours
from glob import glob
import pickle as pkl

task='retina_adult_organoid'
#task='pancreas_conditions_MIA_HPAP2'
#task='adipose_sc_sn_updated'

dir_parent=f'/om2/user/khrovati/data/cross_system_integration/eval/{task}/integration/'

# Folders to have nn computed on
for f in glob(dir_parent+'*/embed.h5ad'):
    print('- '+f.split('/')[-2])

    
# Files with missing metrics - can be used to detect folders that didnt run through and should be deleated
' '.join(list(set(glob(dir_parent+'*'))-set([i.replace('/scib_metrics_scaled.pkl','') for i in glob(dir_parent+'*/scib_metrics_scaled.pkl')])))

# Files with missing embed - can be used to detect folders that didnt run through and should be deleated
' '.join(list(set(glob(dir_parent+'*'))-set([i.replace('/embed.h5ad','') for i in glob(dir_parent+'*/embed.h5ad')])))

# All folders that finished eval ocmpletely and have seed=1
for f in glob(dir_parent+'*/scib_metrics_scaled.pkl'):
    dir_name=f.replace('scib_metrics_scaled.pkl','')
    if pkl.load(open(dir_name+'args.pkl','rb')).seed==1:
        print('- '+dir_name.split('/')[-2])
        
# All folders that finished eval ocmpletely and ...
for f in glob(dir_parent+'*/scib_metrics_scaled.pkl'):
    dir_name=f.replace('scib_metrics_scaled.pkl','')
    args=pkl.load(open(dir_name+'args.pkl','rb'))
    if args.params_opt=="scvi_kl_anneal" & args.kl_weight is None:
        print('- '+dir_name.split('/')[-2])
        
# All folders that finished eval ocmpletely and ...
for f in glob(dir_parent+'*/scib_metrics_scaled.pkl'):
    dir_name=f.replace('scib_metrics_scaled.pkl','')
    args=pkl.load(open(dir_name+'args.pkl','rb'))
    if args.params_opt=="scgen_sample_kl" or args.params_opt=="vamp_z_distance_cycle_weight_std_eval":
        print('- '+dir_name.split('/')[-2])