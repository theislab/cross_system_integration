# Analysis reproducibility

Below are described steps required to reproduce results from Hrovatin et al. 

For all Python notebook files (ipynb) we also provide matching python (py) file conversion.

## Data preprocessing

Data for integration is prepared and analyzed within _data_ directory:
- mouse-human: pancreas_conditions_MIA_HPAP2.ipynb
- organoid-tissue: retina_adult_organoid.ipynb
- cell-nuclei: adipose_sc_sn_updated.ipynb

## Integration evaluation

### Integration and evaluation

All the scripts are in eval/cleaned/, except where otherwise specified.

Integration benchmark was performed with [seml](https://github.com/TUM-DAML/seml). For this, we used yaml config files to specify the parameter settings of individual models to be run, a Python script that executes integration and integration evaluation based on specified parameters (run_integration_seml.py) by calling additional scripts for running individual tasks: integration with different models (see below) and evaluation (run_neighbors.py to compute neighbors and UMAPs on integrated data and run_metrics.py for metric computation).

Integration method scripts:
- Main benchmark:
  - newly implemented models (cVAE, VAMP, CYC, VAMP+CYC): run_integration.py
  - scVI: run_integration_scvi.py
  - GLUE: run_integration_glue.py
  - SATURN: run_integration_saturn.py' 
- Other:
  - scGEN (used only in the supplement and not part of the main benchmark): run_integration_scgen.py

Parameter yaml files:
- Main benchmark:
  - mouse-human: seml_pancreas_conditions_MIA_HPAP2.yaml
  - organoid-tissue: seml_retina_adult_organoid.yaml
  - cell-nuclei: seml_adipose_sc_sn_updated.yaml
- Other:
  - analysis of VampPrior initialization with one/multiple cell types/systems: seml_pancreas_conditions_MIA_HPAP2_priorLoc.yaml

Based on the integration evaluation results, the optimal parameter values were selected in eval_summary.ipynb. Parameter values considered for selection are specified in results/names_parsed.ipynb (as for some models we ran additional parameter settings on some of the datasets before deciding what would be a sensible parameter range to use on all datasets).

Additional scripts:
- Integration metrics helper functions, including some metric adjustments/implementations: metrics.py
- Mapping of run names used in the yaml files to the names of the optimized parameters and adjusted run names: params_opt_maps.py
  
### Additional analysis of integration results

To make the final plotting of data-setting subsets quicker, the integration results of cell subsets were preprocessed in advance (all notebooks in eval/cleaned/analysis/):
- Moran's I of gene groups known to be variable in healthy adult beta cells: moransi_examples.ipynb
- Correct sample and cell type alignment in the retinal dataset: bio_preservation_analysis_retina.ipynb

### Analysis of VampPrior model

## Preparation of final plots/tables for the paper

All scripts are within results/, except where otherwise specified.

Pre-integration batch strength analysis (within batch_strength/ sub-directory):
- Batch effects within and between systems: plots batch_strength/batch_strength_plot.ipynb, supplementary table batch_strength_table.ipynb
- Comparison of batch effect strength between data settings (including significance test): batch_strength_across_datasets.ipynb
