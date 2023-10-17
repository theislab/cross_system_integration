The model is based on the [scvi-tools](https://scvi-tools.org/) framework.

## Training suggestions

For examples of how to train the model, see the benchmarking script https://github.com/theislab/cross_system_integration/blob/main/notebooks/eval/cleaned/run_integration.py 

The model expects as input normalized and log+1 transformed expression. 
We recommend using as _system_ the covariate corresponding to substantial batch effects and as _covariate keys_ any other covariates to be corrected for, such as batches (samples or datasets) within systems.

The model hyperparameters are for reproducibility set to defaults used for development. However, in practice the following parameters should be used instead of defaults:

For model and module (passed via model):
- use_group=False
- mixup_alpha=None
- encode_pseudoinputs_on_eval_mode=True
- n_prior_components=5
- z_dist_metric="MSE_standard"
- prior="vamp" - this will lead to using VampPrior, which is a recommended setting

For loss (passed via train function):
- kl_cycle_weight=0.0
- reconstruction_mixup_weight=0.0
- reconstruction_cycle_weight=0.0
- z_distance_cycle_weight - should be tunned; a sensible value when also using VampPrior as described above will likely be around 1-10 or even higher (<100) if requiring stronger batch correction
- translation_corr_weight=0.0
- z_contrastive_weight=0.0


