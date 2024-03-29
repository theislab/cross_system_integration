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

from cross_system_integration.module._xymodule import XYModule
from cross_system_integration.model._gene_maps import GeneMapRegression

logger = logging.getLogger(__name__)


class XYModel(VAEMixin, UnsupervisedTrainingMixin, BaseModelClass):
    """
    Predict from one (encoder) to another system (decoder). A loss based on known gene relationships is added

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
        super(XYModel, self).__init__(adata)

        # self.summary_stats provides information about anndata dimensions and other tensor info
        self._set_model(adata=adata, **model_kwargs)
        self._model_summary_string = "Overwrite this attribute to get an informative representation for your model"
        # necessary line to get params that will be used for saving/loading
        self.init_params_ = self._get_init_params(locals())

        logger.info("The model has been initialized")

    def _set_model(self, adata, **model_kwargs):
        self.module = XYModule(
            n_input=adata.uns['xsplit'],
            n_output=adata.shape[1] - adata.uns['xsplit'],
            gene_map=GeneMapRegression(adata=adata, build_constraint=self.regression_constraint),
            constraint_regression_loss=self.regression_constraint,
            **model_kwargs,
        )

    @torch.no_grad()
    def translate(
            self,
            adata: AnnData,
            indices: Optional[Sequence[int]] = None,
            give_mean: bool = True,
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
            fwd_outputs = self.module.fwd(**fwd_inputs)
            if give_mean:
                y = fwd_outputs["y_m"]
            else:
                y = self.module.sample_expression(x_m=fwd_outputs["y_m"], x_v=fwd_outputs["y_v"])
            predicted += [y]

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
            y_corr_key: Optional[str] = None,
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

        if cls.regression_constraint and y_corr_key is None:
            adata.uns['y_corr'] = pd.DataFrame(columns=['gx', 'gy', 'intercept', 'coef'])
        else:
            if set(adata.uns[y_corr_key].columns) != {'gx', 'gy', 'intercept', 'coef'}:
                raise ValueError('y corr columns must be gx, gy, intercept, coef')
            adata.uns['y_corr'] = adata.uns[y_corr_key]

        cls._setup_adata(adata=adata,
                         layer=layer,
                         batch_key=batch_key,
                         labels_key=labels_key,
                         categorical_covariate_keys=categorical_covariate_keys,
                         continuous_covariate_keys=continuous_covariate_keys,
                         **kwargs)
        return adata

    @classmethod
    def _setup_adata_split(cls, adata, xy_key):
        # Order genes to be first x and then y
        if set(adata.var[xy_key].unique()) != {'x', 'y'}:
            raise ValueError('values in xy_key column must be x and y')
        adata = adata[:, pd.Series(pd.Categorical(values=adata.var[xy_key], categories=['x', 'y'], ordered=True
                                                  ), index=adata.var_names).sort_values().index].copy()
        adata.var['xy'] = adata.var[xy_key]
        adata.uns['xsplit'] = adata.var[xy_key].tolist().index('y')
        return adata

    @classmethod
    def _setup_adata(cls, adata, layer, batch_key, labels_key, categorical_covariate_keys, continuous_covariate_keys,
                     **kwargs):
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, labels_key),
            CategoricalJointObsField(
                REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys
            ),
            NumericalJointObsField(
                REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys
            ),
        ]
        adata_manager = AnnDataManager(
            fields=anndata_fields, setup_method_args=setup_method_args
        )
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)
