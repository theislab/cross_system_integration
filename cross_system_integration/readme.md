The model is based on the [scvi-tools](https://scvi-tools.org/) framework.

## Training suggestions

For examples of how to perform integration, see the tutorial: https://github.com/theislab/cross_system_integration/blob/main/tutorials/integration_VAMP%2BCYC.ipynb

The model expects as the input normalized and log+1 transformed expression. 
We recommend using as _system_ the covariate corresponding to the substantial batch effects (e.g., species, organoid-tissue, etc.; currently implemented only for exactly two systems) and as _covariate keys_ any other covariates to be corrected for, such as batches within systems (samples or datasets).

The model hyperparameters are for reproducibility set to defaults used for development. However, in practice, the following parameters should be used instead of the defaults while keeping all other parameters at their default values. The below list also lists parameters that should not be changed from the below-specified values.

For adata setup:
- group_key and input_gene_key should be kept at None

For model and module (passed via model):
- mixup_alpha=None
- system_decoders=False
- trainable_priors=True
- pseudoinputs_data_init=True
- encode_pseudoinputs_on_eval_mode=True
- n_prior_components=5
- z_dist_metric="MSE_standard"
- prior="vamp" - this will lead to using VampPrior, which is a recommended setting
- adata_eval=None

For loss (passed via train function):
- kl_cycle_weight=0.0
- reconstruction_mixup_weight=0.0
- reconstruction_cycle_weight=0.0
- z_distance_cycle_weight - should be tunned; a sensible value when also using VampPrior as described above will likely be around 1-10 or even higher (<100) if requiring stronger batch correction
- translation_corr_weight=0.0
- z_contrastive_weight=0.0


