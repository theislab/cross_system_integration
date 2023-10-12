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
import pandas as pd
import pickle as pkl

# %%
path_data='/om2/user/khrovati/data/cross_system_integration/'
path_names=path_data+'names_parsed/'
path_tables=path_data+'tables/'

# %%
dataset_map=pkl.load(open(path_names+'datasets.pkl','rb'))
system_map=pkl.load(open(path_names+'systems.pkl','rb'))
cell_type_map=pkl.load(open(path_names+'cell_types.pkl','rb'))

# %%
# Load results
dataset_metric_fns={
    'pancreas_conditions_MIA_HPAP2':'combined_orthologuesHVG',
    'retina_adult_organoid':'combined_HVG',
    'adipose_sc_sn_updated':'adiposeHsSAT_sc_sn',
}
writer = pd.ExcelWriter(path_tables+'PcaSysBatchDist_Signif.xlsx',engine='xlsxwriter') 
for dataset,fn_part in dataset_metric_fns.items():
    table_sub=pd.read_table(f'{path_data}{dataset}/{fn_part}_PcaSysBatchDist_Signif.tsv')
    table_sub['cell_type']=table_sub['cell_type'].map(cell_type_map[dataset])
    table_sub['system']=table_sub['system'].\
        str.replace('s0',system_map[dataset]['0']).\
        str.replace('s1',system_map[dataset]['1'])
    dataset_name=dataset_map[dataset]
    table_sub.to_excel(writer, sheet_name=dataset_name,index=False)   
    print(dataset_name)
    display(table_sub) 
writer.close()

# %%
