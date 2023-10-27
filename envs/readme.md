# Conda environments for reproducibility

Different environments were used for different analysis and integration steps. For each environment we provide `conda env export` yaml file and `pip list` text file.


List of env names for individual analyses:
- Data preparation for integration: analysis
- Integration with new models (cVAE, CYC, VAMP) and scVI as well as analysis of integration results and metrics: csi
- Integration with GLUE: scglue
- Integration with SATURN: saturn. SATURN was installed from https://github.com/moinfar/SATURN_fix.git (last commit cf5474e16308bdd6207ee7edd590cdaabbee2f1d) as the original repository contained a bug that required fixing.
- Integration with scGEN: perturb. scGEN was installed after a bug fix (last commit d79e1f04233c30f9a4eb5b8d57718127909807d7).
