import logging
from typing import List, Optional, Union, Sequence, Dict
import pandas as pd
import numpy as np
import torch

from anndata import AnnData
from scvi import REGISTRY_KEYS
from scvi.data import AnnDataManager
from scvi.data.fields import (
    ObsmField,
    NumericalObsField,
    LayerField,
)
from scvi.model.base import BaseModelClass, UnsupervisedTrainingMixin, VAEMixin
from scvi.utils import setup_anndata_dsp

from cross_system_integration.module._xylinmodule import XYLinModule
from cross_system_integration.model._gene_maps import GeneMapEmbedding
from cross_system_integration.model._xymodel import XYModel

logger = logging.getLogger(__name__)


class XYLinModel(XYModel):
    """
    Single encoder and two decoders with one encoder having linear reconstruction based on gene embedding

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
            n_cov_x=adata.obsm['covariates_x'].shape[1],
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
            z = self.module.fwd_embed(x=fwd_inputs['x'], cov_x = fwd_inputs['cov_x'])[2]
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
            train_y_key,
            gene_mean_key: str = None,
            gene_std_key: str = None,
            layer: Optional[str] = None,
            # TODO also make for y
            categorical_covariate_keys_x: Optional[List[str]] = None,
            continuous_covariate_keys_x: Optional[List[str]] = None,
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
        train_y_key
            Whether y should be used for training, 0 - False, 1 - True

        Returns
        -------
        %(returns)s
        """
        adata = cls._setup_adata_split(adata=adata, xy_key=xy_key)

        # Set up covariates
        # TODO this could be handled by specific field type in registry
        if covariate_orders is None:
            covariate_orders = {}

        covariates_x, orders_dict_x, cov_dict_x = _prepare_metadata(
            meta_data=adata.obs,
            cov_cat_keys=categorical_covariate_keys_x,
            cov_cont_keys=continuous_covariate_keys_x,
            orders=covariate_orders
        )

        adata.uns['covariate_orders'] = {**orders_dict_x}
        adata.uns['covariates'] = {'x': cov_dict_x}
        adata.obsm['covariates_x'] = covariates_x

        # Make sure that train y key is of correct type
        if len(set(adata.obs[train_y_key].unique()) - {0, 1}) > 0:
            raise ValueError('Obs field train_y_key can contain only 0 and 1')

        # Set up constraints
        # TODO Could be somehow added in registry?
        adata.varm['embed'] = adata.varm[gene_embed_key]
        if gene_mean_key is not None:
            adata.var['mean'] = adata.var[gene_mean_key]
        else:
            adata.var['mean'] = 0.0
        if gene_std_key is not None:
            adata.var['std'] = adata.var[gene_std_key]
        else:
            adata.var['std'] = 1.0

        # Anndata setup
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=False),
            ObsmField('covariates_x', 'covariates_x'),
            NumericalObsField('train_y', train_y_key)
        ]
        adata_manager = AnnDataManager(
            fields=anndata_fields, setup_method_args=setup_method_args
        )
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

        return adata


def _prepare_metadata(meta_data: pd.DataFrame,
                      cov_cat_keys: Optional[list] = None,
                      cov_cont_keys: Optional[list] = None,
                      orders=None):
    """

    :param meta_data: Dataframe containing species and covariate info, e.g. from non-registered adata.obs
    :param cov_cat_keys: List of categorical covariates column names.
    :param cov_cont_keys: List of continuous covariates column names.
    :param orders: Defined orders for species or categorical covariates. Dict with keys being
    'species' or categorical covariates names and values being lists of categories. May contain more/less
    categories than data.
    :return:
    """
    if cov_cat_keys is None:
        cov_cat_keys = []
    if cov_cont_keys is None:
        cov_cont_keys = []

    def dummies_categories(values: pd.Series, categories: Union[List, None] = None):
        """
        Make dummies of categorical covariates. Use specified order of categories.
        :param values: Categories for each observation.
        :param categories: Order of categories to use.
        :return: dummies, categories. Dummies - one-hot encoding of categories in same order as categories.
        """
        if categories is None:
            categories = pd.Categorical(values).categories.values
        values = pd.Series(pd.Categorical(values=values, categories=categories, ordered=True),
                           index=values.index, name=values.name)
        dummies = pd.get_dummies(values, prefix=values.name)
        return dummies, list(categories)

    # Covariate encoding
    # Save order of covariates and categories
    cov_dict = {'categorical': cov_cat_keys, 'continuous': cov_cont_keys}
    # One-hot encoding of categorical covariates
    orders_dict = {}
    cov_cat_data = []
    for cov_cat_key in cov_cat_keys:
        cat_dummies, cat_order = dummies_categories(
            values=meta_data[cov_cat_key], categories=orders.get(cov_cat_key, None))
        cov_cat_data.append(cat_dummies)
        orders_dict[cov_cat_key] = cat_order
    # Prepare single cov array for all covariates and in per-species format
    cov_data_parsed = pd.concat(cov_cat_data + [meta_data[cov_cont_keys]], axis=1)
    return cov_data_parsed, orders_dict, cov_dict
