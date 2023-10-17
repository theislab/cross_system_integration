The model is based on the [scvi-tools](https://scvi-tools.org/) framework.

## Training suggestions

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
- z_distance_cycle_weight - should be tunned; a sensible value when also using VampPrior as described above may be around 1-10 or even higher (<100) if requiring stronger batch correction
- translation_corr_weight=0.0
- z_contrastive_weight=0.0


