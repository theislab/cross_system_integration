The model is based on the [scvi-tools](https://scvi-tools.org/) framework.

## Training suggestions

The model hyperparameters are for reproducibility set to defaults used for development. However, in practice the following parameters should be used instead of defaults:

For model and module (passed via model):
- use_group=False
- mixup_alpha=None
- encode_pseudoinputs_on_eval_mode=True
- n_prior_components=5
- z_dist_metric="MSE_standard"
  
We also recommend to use the VAMP+CYC model which can be done by setting the following parameters:
- prior="vamp"

