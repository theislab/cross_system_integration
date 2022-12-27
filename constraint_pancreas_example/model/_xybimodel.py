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
from constraint_pancreas_example.module._xybimodule import XYBiModule
from constraint_pancreas_example.model._gene_maps import GeneMapInput

logger = logging.getLogger(__name__)


class XYBiModel(VAEMixin, TrainingMixin, BaseModelClass):
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
        super(XYBiModel, self).__init__(adata)

        # self.summary_stats provides information about anndata dimensions and other tensor info
        self.module = XYBiModule(
            # TODO make option to have less input than output genes
            n_input_x=(adata.var['input'].values[:adata.uns['xsplit']]).sum(),
            n_input_y=(adata.var['input'].values[adata.uns['xsplit']:]).sum(),
            n_output_x=adata.uns['xsplit'],
            n_output_y=adata.shape[1] - adata.uns['xsplit'],
            gene_map=GeneMapInput(adata=adata),
            n_cov_x=adata.obsm['covariates_x'].shape[1],
            n_cov_y=adata.obsm['covariates_y'].shape[1],
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
        predicted_x = []
        predicted_y = []
        for tensors in tensors_fwd:
            # Inference
            inference_inputs = self.module._get_inference_input(tensors)
            generative_inputs = self.module._get_generative_input(
                tensors=tensors, inference_outputs=self.module.inference(**inference_inputs))
            generative_outputs = self.module.generative(**generative_inputs, pad_output=True, x_x=False, y_y=False)
            if give_mean:
                x = generative_outputs["y_x_m"]
                y = generative_outputs["x_y_m"]
            else:
                x = generative_outputs["y_x"]
                y = generative_outputs["x_y"]
            predicted_x += [x]
            predicted_y += [y]

        predicted_x = torch.cat(predicted_x)
        predicted_y = torch.cat(predicted_y)

        if as_numpy:
            predicted_x = predicted_x.cpu().numpy()
            predicted_y = predicted_y.cpu().numpy()
            # TODO make sure that it is known which genes are at which feature index (gene names!) - since
            #  setup adata mixes indices of features
        return predicted_x, predicted_y

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
        predicted_x = []
        predicted_y = []
        for tensors in tensors_fwd:
            # Inference
            inference_inputs = self.module._get_inference_input(tensors)
            z = self.module.inference(**inference_inputs, pad_output=True)
            if give_mean:
                predicted_x += [z['z_x_m']]
                predicted_y += [z['z_y_m']]
            else:
                predicted_x += [z['z_x']]
                predicted_y += [z['z_y']]

        predicted_x = torch.cat(predicted_x)
        predicted_y = torch.cat(predicted_y)

        if as_numpy:
            predicted_x = predicted_x.cpu().numpy()
            predicted_y = predicted_y.cpu().numpy()
        return predicted_x, predicted_y

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
            cls,
            adata: AnnData,
            xy_key,
            train_x_key,
            train_y_key,
            orthology_key: Optional[str] = None,
            input_gene_key: Optional[str] = None,
            layer: Optional[str] = None,
            categorical_covariate_keys_x: Optional[List[str]] = None,
            continuous_covariate_keys_x: Optional[List[str]] = None,
            categorical_covariate_keys_y: Optional[List[str]] = None,
            continuous_covariate_keys_y: Optional[List[str]] = None,
            covariate_orders: Optional[Dict] = None,
            fill_covariates: bool = True,
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
        # Make sure that train x/y key is of correct type
        if len(set(adata.obs[train_x_key].unique()) - {0, 1}) > 0 or \
                len(set(adata.obs[train_y_key].unique()) - {0, 1}) > 0:
            raise ValueError('Obs fields train_x_key and train_y_key can contain only 0 and 1')

        # Make sure var names are unique
        if adata.shape[1] != len(set(adata.var_names)):
            raise ValueError('Adata var_names are not unique')

        # This also copied adata!
        adata = cls._setup_anndata_split(adata=adata, xy_key=xy_key)

        if orthology_key is not None:
            if set(adata.uns[orthology_key].columns) != {'x', 'y'}:
                raise ValueError('Orthology DataFrame must have columns x and y')
            adata.uns['orthology'] = adata.uns[orthology_key]
        else:
            adata.uns['orthology'] = pd.DataFrame(columns=['x', 'y'])

        # Which genes to use as input
        if input_gene_key is None:
            adata.var['input'] = 1
        else:
            adata.var['input'] = adata.var[input_gene_key]

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

        covariates_y, orders_dict_y, cov_dict_y = _prepare_metadata(
            meta_data=adata.obs,
            cov_cat_keys=categorical_covariate_keys_y,
            cov_cont_keys=continuous_covariate_keys_y,
            orders=covariate_orders
        )

        adata.uns['covariate_orders'] = {**orders_dict_x, **orders_dict_y}
        adata.uns['covariates'] = {'x': cov_dict_x, 'y': cov_dict_y}
        adata.obsm['covariates_x'] = covariates_x
        adata.obsm['covariates_y'] = covariates_y

        # Set cov of species cells not used for training to cov of another cell from that species
        # Will be used later if predicting the other species (which is unknown)
        def fill_cov(train_key, cov_key):
            train = adata.obs[train_key].astype('bool').values.ravel()
            cov_samples = adata.obsm[cov_key].values[train, :]
            idx = np.random.randint(cov_samples.shape[0], size=(~train).sum())
            cov_samples = cov_samples[idx, :]
            adata.obsm[cov_key].loc[adata.obs_names[~train], :] = cov_samples
            # TODO ensure that at leas some samples from both x and y are used for training

        if fill_covariates:
            fill_cov(train_key=train_x_key, cov_key='covariates_x')
            fill_cov(train_key=train_y_key, cov_key='covariates_y')

        # Anndata setup
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=False),
            ObsmField('covariates_x', 'covariates_x'),
            ObsmField('covariates_y', 'covariates_y'),
            NumericalObsField('train_x', train_x_key),
            NumericalObsField('train_y', train_y_key),
        ]
        adata_manager = AnnDataManager(
            fields=anndata_fields, setup_method_args=setup_method_args
        )
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

        return adata

    @classmethod
    def _setup_anndata_split(cls, adata, xy_key):
        # Order genes to be first x and then y
        if set(adata.var[xy_key].unique()) != {'x', 'y'}:
            raise ValueError('values in xy_key column must be x and y')
        adata = adata[:, pd.Series(pd.Categorical(values=adata.var[xy_key], categories=['x', 'y'], ordered=True
                                                  ), index=adata.var_names).sort_values().index].copy()
        adata.var['xy'] = adata.var[xy_key]
        adata.uns['xsplit'] = adata.var[xy_key].tolist().index('y')
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
