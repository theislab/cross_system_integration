import logging
from typing import List, Optional, Union, Sequence
import pandas as pd
import numpy as np
import torch

from anndata import AnnData
from scvi import REGISTRY_KEYS
from scvi.data import AnnDataManager
from scvi.data.fields import (
    CategoricalJointObsField,
    CategoricalObsField,
    LayerField,
    NumericalJointObsField,
)
from scvi.model.base import BaseModelClass, UnsupervisedTrainingMixin, VAEMixin
from scvi.utils import setup_anndata_dsp

from constraint_pancreas_example.module._xylinmodule import XYLinModule
from constraint_pancreas_example.model._gene_maps import GeneMapEmbedding
from constraint_pancreas_example.model._xymodel import XYModel

logger = logging.getLogger(__name__)


class XYLinModel(XYModel):
    """
    Skeleton for an scvi-tools model.

    Please use this skeleton to create new models.

    Parameters
    ----------
    adata
        AnnData object that has been registered via :meth:`~mypackage.MyModel.setup_anndata`.
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
    """

    regression_constraint = False

    def __init__(
            self,
            adata: AnnData,
            **model_kwargs,
    ):
        super(XYLinModel, self).__init__(adata)

        # self.summary_stats provides information about anndata dimensions and other tensor info

        self._model_summary_string = "Overwrite this attribute to get an informative representation for your model"
        # necessary line to get params that will be used for saving/loading
        self.init_params_ = self._get_init_params(locals())

        logger.info("The model has been initialized")

    def _set_model(self, adata, **model_kwargs):
        self.module = XYLinModule(
            n_input=adata.uns['xsplit'],
            n_output=adata.shape[1] - adata.uns['xsplit'],
            gene_map=GeneMapEmbedding(adata=adata, build_constraint=True),
            **model_kwargs,
        )

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
            cls,
            adata: AnnData,
            xy_key,
            gene_embed_key,
            gene_mean_key,
            gene_std_key,
            batch_key: Optional[str] = None,
            labels_key: Optional[str] = None,
            layer: Optional[str] = None,
            categorical_covariate_keys: Optional[List[str]] = None,
            continuous_covariate_keys: Optional[List[str]] = None,
            **kwargs,
    ) -> Optional[AnnData]:
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

        Returns
        -------
        %(returns)s
        """
        adata = cls._setup_adata_split(adata=adata, xy_key=xy_key)

        # Set up constraints
        adata.varm['embed'] = adata.varm[gene_embed_key]
        adata.var['mean'] = adata.var[gene_mean_key]
        adata.var['std'] = adata.var[gene_std_key]

        cls._setup_adata(adata=adata,
                         layer=layer,
                         batch_key=batch_key,
                         labels_key=labels_key,
                         categorical_covariate_keys=categorical_covariate_keys,
                         continuous_covariate_keys=continuous_covariate_keys,
                         **kwargs)
        return adata
