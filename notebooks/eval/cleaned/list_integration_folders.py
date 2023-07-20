# List integration folders to copied to seml script for metrics of neighbours
from glob import glob

#task='retina_adult_organoid'
task='pancreas_conditions_MIA_HPAP2'

dir_parent=f'/om2/user/khrovati/data/cross_system_integration/eval/{task}/integration/'

# Folders to have nn computed on
for f in glob(dir_parent+'*/embed.h5ad'):
    print('- '+f.split('/')[-2])

    
# Files with missing metrics - can be used to detect folders that didnt run through and should be deleated
' '.join(list(set([i.replace('embed.h5ad','') for i in glob(dir_parent+'*/embed.h5ad')])-set([i.replace('scib_metrics.pkl','') for i in glob(dir_parent+'*/scib_metrics.pkl')])))

# Files with missing embed - can be used to detect folders that didnt run through and should be deleated
' '.join(list(set(glob(dir_parent+'*'))-set([i.replace('/embed.h5ad','') for i in glob(dir_parent+'*/embed.h5ad')])))