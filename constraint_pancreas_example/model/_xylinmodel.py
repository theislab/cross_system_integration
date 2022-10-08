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

    @torch.no_grad()
    def embed(
            self,
            adata: AnnData,
            indices: Optional[Sequence[int]] = None,
            batch_size: Optional[int] = None,
            as_numpy: bool = True
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Translate expression - based on input expression and metadata
        predict expression of data as if it had given metadata.
        expression, metadata (from adata) -> latent
        latent, new metadata -> translated expression
        :param prediction_metadata: Metadata for which samples should be predicted
        :param species_key: Species col name in prediction_metadata
        :param adata: Input adata based on which latent representation is obtained.
        :param indices:
        :param give_mean: In latent and expression prediction use mean rather than samples
        :param batch_size:
        :param as_numpy:  Move output tensor to cpu and convert to numpy
        :return:
        """
        # Check model and adata
        self._check_if_trained(warn=False)
        adata = self._validate_anndata(adata)
        if indices is None:
            indices = np.arange(adata.n_obs)
        # Prediction
        # Do not shuffle to retain order
        tensors_fwd = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size, shuffle=False
        )
        predicted = []
        for tensors in tensors_fwd:
            # Inference
            fwd_inputs = self.module._get_fwd_input(tensors)
            z = self.module.fwd_embed(x=fwd_inputs['x'])
            predicted += [z]

        predicted = torch.cat(predicted)

        if as_numpy:
            predicted = predicted.cpu().numpy()
        return predicted


    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
            cls,
            adata: AnnData,
            xy_key,
            gene_embed_key,
            gene_mean_key: str = None,
            gene_std_key: str = None,
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
        if gene_mean_key is not None:
            adata.var['mean'] = adata.var[gene_mean_key]
        else:
            adata.var['mean'] = 0.0
        if gene_std_key is not None:
            adata.var['std'] = adata.var[gene_std_key]
        else:
            adata.var['std'] = 1.0

        cls._setup_adata(adata=adata,
                         layer=layer,
                         batch_key=batch_key,
                         labels_key=labels_key,
                         categorical_covariate_keys=categorical_covariate_keys,
                         continuous_covariate_keys=continuous_covariate_keys,
                         **kwargs)
        return adata
