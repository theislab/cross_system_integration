from typing import Union

from anndata import AnnData
#from scvi.data._anndata import _check_anndata_setup_equivalence as _check_anndata_setup_equivalence_scvi


def _check_anndata_setup_equivalence(
        adata_source: AnnData, adata_target: AnnData
) -> bool:
    """
    Checks if target setup is equivalent to source.

    Parameters
    ----------
    adata_source
        Either AnnData already setup or scvi_setup_dict as the source
    adata_target
        Target AnnData to check setup equivalence

    Returns
    -------
    Whether the adata_target should be run through `transfer_anndata_setup`
    """
    transfer_setup = _check_anndata_setup_equivalence_scvi(adata_source=adata_source.uns['_scvi'],
                                                           adata_target=adata_target)
    # Additional checks not in scvi
    # TODO maybe rewrite this to print out any mismatches
    # Var names match
    transfer_setup = transfer_setup or (not all(adata_source.var_names.values == adata_target.var_names.values))
    # Categoricals match (due to different adata setup this needs to be checked separately)
    transfer_setup = transfer_setup or adata_source.uns['orders'] != adata_target.uns['orders']
    transfer_setup = transfer_setup or adata_source.uns['covariates'] != adata_target.uns['covariates']
    # Covariates data
    transfer_setup = transfer_setup or \
                     adata_target.obsm['species_ratio'].shape[1] != len(adata_source.uns['orders']['species'])
    transfer_setup = transfer_setup or \
                     adata_target.obsm['cov_species'].shape[1] != len(adata_source.uns['orders']['species'])
    transfer_setup = transfer_setup or \
                     adata_target.obsm['cov_species'].shape[2] != (sum([len(adata_source.uns['orders'][cov]) for cov in
                                                                        adata_source.uns['covariates']['categorical']])
                                                                   + len(adata_source.uns['covariates']['continuous']))
    # Eval_o is present
    transfer_setup = transfer_setup or adata_target.obsm['eval_o'].shape[1] != 1

    # TODO also maybe check orthologues mapping and var names species, ... Is this even used from new adata?

    return transfer_setup
