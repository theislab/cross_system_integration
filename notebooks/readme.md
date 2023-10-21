# Analysis reproducibility

Below are described steps required for the reproduction of results from Hrovatin et al. 

For all python notebook files (ipynb) we also provide matching python (py) file conversion.

## Data preprocessing

Data for integration is prepared and analysed within _data_ directory:
- mouse-human: pancreas_conditions_MIA_HPAP2.ipynb
- organoid-tissue: retina_adult_organoid.ipynb
- cell-nuclei: adipose_sc_sn_updated.ipynb

## Integration evaluation

### Integration

Integration benchmark was performed with [seml](https://github.com/TUM-DAML/seml). For this, we used yaml config files to specify the parameter settings of individual models to be run, a python script that executes integration and integration evaluation based on specified parameters (run_integration_seml.py) by calling additional scripts for running individual tasks: integration with different models (see below) and evaluation (run_neighbors.py to compute neighbours and UMAPs on integrated data and run_metrics.py for metric computation).

Parameter yaml files:

Integration method scripts:
- newly implemented models (cVAE, VAMP, CYC, VAMP+CYC):
- scVI:
- GLUE:
- SATURN:


### Analysis of integration results

### Analysis of VampPrior model

### Analysis of integration metrics

## Preparation of final plots/tables for the paper
