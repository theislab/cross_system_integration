import logging
from typing import List, Optional, Union, Sequence, Dict
import pandas as pd
import numpy as np
import torch

from anndata import AnnData
from scvi import REGISTRY_KEYS
from scvi.data import AnnDataManager
from scvi.data.fields import (
    LayerField, ObsmField, NumericalObsField,
)
from scvi.model.base import BaseModelClass, VAEMixin
from scvi.utils import setup_anndata_dsp

from constraint_pancreas_example.model._training import TrainingMixin
from constraint_pancreas_example.module._xxjointmodule import XXJointModule
from constraint_pancreas_example.model._gene_maps import GeneMapInput
from constraint_pancreas_example.model._utils import prepare_metadata

logger = logging.getLogger(__name__)


class XXJointModel(VAEMixin, TrainingMixin, BaseModelClass):
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

    def __init__(
            self,
            adata: AnnData,
            **model_kwargs,
    ):
        super(XXJointModel, self).__init__(adata)

        # self.summary_stats provides information about anndata dimensions and other tensor info
        self.module = XXJointModule(
            # TODO!!!!
            n_input=adata.var['input'].sum(),
            n_output=adata.shape[1],
            gene_map=GeneMapInput(adata=adata),  # TODO!!!!
            n_cov=adata.obsm['covariates'].shape[1] + adata.obsm['system'].shape[1],
            **model_kwargs)

        self._model_summary_string = "Overwrite this attribute to get an informative representation for your model"
        # necessary line to get params that will be used for saving/loading
        self.init_params_ = self._get_init_params(locals())

        logger.info("The model has been initialized")

    @torch.no_grad()
    def translate(
            self,
            adata: AnnData,
            indices: Optional[Sequence[int]] = None,
            give_mean: bool = True,
            batch_size: Optional[int] = None,
            as_numpy: bool = True
    ) -> (Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]):
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
        predicted_y = []
        for tensors in tensors_fwd:
            # Inference
            inference_inputs = self.module._get_inference_input(tensors)
            generative_inputs = self.module._get_generative_input(
                tensors=tensors, inference_outputs=self.module.inference(**inference_inputs))
            generative_outputs = self.module.generative(**generative_inputs, x_x=False, x_y=True)
            if give_mean:
                y = generative_outputs["y_m"]
            else:
                y = generative_outputs["y"]
            predicted_y += [y]

        predicted_y = torch.cat(predicted_y)

        if as_numpy:
            predicted_y = predicted_y.cpu().numpy()
        return predicted_y

    @torch.no_grad()
    def embed(
            self,
            adata: AnnData,
            indices: Optional[Sequence[int]] = None,
            give_mean: bool = True,
            batch_size: Optional[int] = None,
            as_numpy: bool = True
    ) -> (Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]):
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
            inference_inputs = self.module._get_inference_input(tensors)
            z = self.module.inference(**inference_inputs)
            if give_mean:
                predicted += [z['z_m']]
            else:
                predicted += [z['z']]

        predicted = torch.cat(predicted)

        if as_numpy:
            predicted = predicted.cpu().numpy()
        return predicted

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
            cls,
            adata: AnnData,
            system_key: str,
            class_key: str,
            input_gene_key: Optional[str] = None,
            layer: Optional[str] = None,
            categorical_covariate_keys: Optional[List[str]] = None,
            continuous_covariate_keys: Optional[List[str]] = None,
            covariate_orders: Optional[Dict] = None,
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

        # Make sure var names are unique
        if adata.shape[1] != len(set(adata.var_names)):
            raise ValueError('Adata var_names are not unique')

        adata = adata.copy()

        # Make system to categorical for cov
        if adata.obs[system_key].nunique() != 2:
            raise ValueError('There must be exactly two systems')
        system_order = sorted(adata.obs[system_key].unique())
        systems_dict = dict(zip(system_order, [0.0, 1.0]))
        adata.uns['system_order'] = system_order
        adata.obsm['system'] = adata.obs[system_key].map(systems_dict).to_frame()

        adata.obs['class'] = adata.obs[class_key]

        # Which genes to use as input
        if input_gene_key is None:
            adata.var['input'] = 1
        else:
            adata.var['input'] = adata.var[input_gene_key]

        # Set up covariates
        # TODO this could be handled by specific field type in registry
        if covariate_orders is None:
            covariate_orders = {}

        # System and class must not be in cov
        if class_key in categorical_covariate_keys or class_key in continuous_covariate_keys \
                or system_key in categorical_covariate_keys or system_key in continuous_covariate_keys:
            raise ValueError('class_key or system_key should not be within covariate keys')

        covariates, orders_dict, cov_dict = prepare_metadata(
            meta_data=adata.obs,
            cov_cat_keys=categorical_covariate_keys,
            cov_cont_keys=continuous_covariate_keys,
            orders=covariate_orders
        )

        adata.uns['covariate_orders'] = orders_dict
        adata.uns['covariates_dict'] = cov_dict
        adata.obsm['covariates'] = covariates

        # Anndata setup
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=False),
            ObsmField('covariates', 'covariates'),
            ObsmField('system', 'system'),
        ]
        adata_manager = AnnDataManager(
            fields=anndata_fields, setup_method_args=setup_method_args
        )
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

        return adata
