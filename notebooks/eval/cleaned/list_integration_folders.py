# List integration folders to copied to seml script for metrics of neighbours
from glob import glob

#task='retina_adult_organoid'
task='pancreas_conditions_MIA_HPAP2'
#task='adipose_sc_sn_updated'

dir_parent=f'/om2/user/khrovati/data/cross_system_integration/eval/{task}/integration/'

# Folders to have nn computed on
#for f in glob(dir_parent+'*/embed.h5ad'):
#    print('- '+f.split('/')[-2])

    
# Files with missing metrics - can be used to detect folders that didnt run through and should be deleated
' '.join(list(set(glob(dir_parent+'*'))-set([i.replace('/scib_metrics_scaled.pkl','') for i in glob(dir_parent+'*/scib_metrics_scaled.pkl')])))

# Files with missing embed - can be used to detect folders that didnt run through and should be deleated
' '.join(list(set(glob(dir_parent+'*'))-set([i.replace('/embed.h5ad','') for i in glob(dir_parent+'*/embed.h5ad')])))