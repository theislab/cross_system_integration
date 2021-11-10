import logging
from typing import Optional

import torch

from anndata import AnnData
from scvi.data import setup_anndata, register_tensor_from_anndata
from scvi.model.base import BaseModelClass, UnsupervisedTrainingMixin
from scvi.utils import setup_anndata_dsp

from cross_species_prediction.module._multied import Multied
from cross_species_prediction.model._gene_maps import GeneMap

logger = logging.getLogger(__name__)


class Model(UnsupervisedTrainingMixin, BaseModelClass):
    """
    Skeleton for an scvi-tools model.

    Please use this skeleton to create new models.

    Parameters
    ----------
    adata
        AnnData object that has been registered via :func:`~scvi.data.setup_anndata`.
    n_hidden
        Number of nodes per hidden layer.
    n_latent
        Dimensionality of the latent space.
    n_layers
        Number of hidden layers used for encoder and decoder NNs.
    **model_kwargs
        Keyword args for :class:`~mypackage.MyModule`
    Examples
    --------
    # TODO correct
    >>> adata = anndata.read_h5ad(path_to_anndata)
    # Data should be normalised and log(expression+1) transformed
    >>> cross_species_prediction.Model.setup_anndata(adata, batch_key="batch",species_key='species')
    >>> model = cross_species_prediction.Model(adata)
    >>> model.train()
    >>> adata.obsm["X_mymodel"] = model.get_latent_representation()
    """

    def __init__(
            self,
            adata: AnnData,
            n_latent: int = 10,
            n_hidden: int = 128,
            n_layers: int = 1,
            dropout_rate: float = 0.1,
            use_batch_norm: bool = True,
            use_layer_norm: bool = False,
            use_activation: bool = True,
            bias: bool = True,
            inject_covariates: bool = True,
            activation_fn: torch.nn.Module = torch.nn.ReLU,
            var_mode: str = 'feature',
    ):
        super(Model, self).__init__(adata)
        # self.summary_stats provides information about anndata dimensions and other tensor info
        n_species = len(adata.uns['species_order'])
        n_cov = adata.obsm['cov_species'].shape[2]
        model_params = dict(
            n_hidden=n_hidden,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            use_activation=use_activation,
            bias=bias,
            inject_covariates=inject_covariates,
            activation_fn=activation_fn,
        )
        self.module = Multied(
            n_latent=n_latent,
            n_cov=n_cov,
            n_species=n_species,
            gene_maps=GeneMap(adata=adata),
            encoder_params=model_params,
            decoder_params={**model_params, **dict(var_mode=var_mode)}
        )

        self._model_summary_string = (
            "Cross species prediction with multiple encoders and decoders"
        )

        # necessary line to get params that will be used for saving/loading
        self.init_params_ = self._get_init_params(locals())

        logger.info("The model has been initialized")

    # TODO modify!
    # def _validate_anndata(
    #         self, adata: Optional[AnnData] = None, copy_if_view: bool = True
    # ):
    #     """Validate anndata has been properly registered, transfer if necessary."""
    #     if adata is None:
    #         adata = self.adata
    #     if adata.is_view:
    #         if copy_if_view:
    #             logger.info("Received view of anndata, making copy.")
    #             adata = adata.copy()
    #         else:
    #             raise ValueError("Please run `adata = adata.copy()`")
    #
    #     if "_scvi" not in adata.uns_keys():
    #         logger.info(
    #             "Input adata not setup with scvi. "
    #             + "attempting to transfer anndata setup"
    #         )
    #         transfer_anndata_setup(self.scvi_setup_dict_, adata)
    #
    #     needs_transfer = _check_anndata_setup_equivalence(self.scvi_setup_dict_, adata)
    #     if needs_transfer:
    #         transfer_anndata_setup(self.scvi_setup_dict_, adata)
    #
    #     # Make sure right fields are in adata
    #     if self.scvi_setup_dict_['extra_categoricals']['keys'] != ['species']:
    #         raise ValueError('extra_categoricals should contain only species')
    #     if self.scvi_setup_dict_['summary_states']['n_labels'] != 1:
    #         raise ValueError('Labels are not used')
    #     if self.scvi_setup_dict_['summary_states']['n_proteins'] != 0:
    #         raise ValueError('Proteins are not used')
    #     if self.scvi_setup_dict_['summary_states']['n_continous_covs'] != 0:
    #         raise ValueError('Continuous covariates should not be present')
    #
    #     return adata

    @staticmethod
    @setup_anndata_dsp.dedent
    # TODO modify!
    def setup_anndata(
            adata: AnnData,
    ):
        # TODO add options like copy, ...
        """
        %(summary)s.

        Parameters
        ----------
        %(param_adata)s
        %(param_batch_key)s
        %(param_labels_key)s
        %(param_layer)s
        %(param_cat_cov_keys)s
        %(param_cont_cov_keys)s
        %(param_copy)s

        Returns
        -------
        %(returns)s
        """
        setup_anndata(adata) #  TODO  FutureWarning: Function setup_anndata is deprecated; Please use the model-specific setup_anndata methods instead. The global method will be removed in version 0.15.0.
        register_tensor_from_anndata(
            adata=adata,
            adata_attr_name="obsm",
            adata_key_name='species_ratio',
            registry_key="species_ratio",
            is_categorical=False,
        )
        register_tensor_from_anndata(
            adata=adata,
            adata_attr_name="obsm",
            adata_key_name='eval_o',
            registry_key="eval_o",
            is_categorical=False,
        )
        register_tensor_from_anndata(
            adata=adata,
            adata_attr_name="obsm",
            adata_key_name='cov_species',
            registry_key="cov_species",
            is_categorical=False,
        )
