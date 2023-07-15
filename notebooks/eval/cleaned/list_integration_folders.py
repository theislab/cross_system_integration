# List integration folders to copied to seml script for metrics of neighbours
from glob import glob

dir_parent='/om2/user/khrovati/data/cross_system_integration/eval/retina_adult_organoid/integration/'

for f in glob(dir_parent+'*/embed.h5ad'):
    print('- '+f)