import logging
import warnings
from typing import Optional, Sequence, Dict, List, Union
import numpy as np
import pandas as pd
import scanpy as sc

import torch

from anndata import AnnData
from pynndescent import NNDescent
from scipy import sparse
#from scvi.data import register_tensor_from_anndata
#from scvi.data._anndata import _setup_anndata, transfer_anndata_setup, _register_anndata
from scvi.model.base import BaseModelClass, UnsupervisedTrainingMixin
from scvi.utils import setup_anndata_dsp

from cross_system_integration_old.module._multied import Multied
from cross_system_integration_old.model._gene_maps import GeneMap
from cross_system_integration_old.data._anndata import _check_anndata_setup_equivalence

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
        n_species = len(adata.uns['orders']['species'])
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

    def _validate_anndata(
            self, adata: Optional[AnnData] = None, copy_if_view: bool = True
    ):
        """Validate anndata has been properly registered"""
        # TODO make transfer
        if adata is None:
            adata = self.adata
        if adata.is_view:
            if copy_if_view:
                logger.info("Received view of anndata, making copy.")
                adata = adata.copy()
            else:
                raise ValueError("Please run `adata = adata.copy()`")

        if "_scvi" not in adata.uns_keys():
            #  TODO make sure this would work on my setup
            # logger.info(
            #     "Input adata not setup with scvi. "
            #     + "attempting to transfer anndata setup"
            # )
            # transfer_anndata_setup(self.scvi_setup_dict_, adata)
            raise ValueError('Adata is not registered with scvi')

        needs_transfer = _check_anndata_setup_equivalence(self.adata, adata)
        if needs_transfer:
            #  TODO make sure this would work on my setup
            # transfer_anndata_setup(self.scvi_setup_dict_, adata)
            raise ValueError('Adata does not match the model adata')

        return adata

    @torch.no_grad()
    def get_latent_representation(
            self,
            adata: Optional[AnnData] = None,
            indices: Optional[Sequence[int]] = None,
            give_mean: bool = True,
            batch_size: Optional[int] = None,
            as_numpy: bool = True
    ) -> Union[np.ndarray, torch.Tensor]:
        r"""
        Return the latent representation for each cell.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        give_mean
            Give mean of distribution (qz_m) or sample from it (z).
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        as_numpy
            Move output tensor to cpu and convert to numpy
        Returns
        -------
        latent_representation : np.ndarray
            Low-dimensional representation for each cell
        """
        self._check_if_trained(warn=False)

        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )
        latent = []
        for tensors in scdl:
            inference_inputs = self.module._get_inference_input(tensors)
            outputs = self.module.inference(**inference_inputs)

            if give_mean:
                zm = outputs["qz_m"]
            else:
                zm = outputs["z"]

            latent += [zm]
        latent = torch.cat(latent)
        if as_numpy:
            latent = latent.cpu().numpy()
        return latent

    @torch.no_grad()
    def translate(
            self,
            prediction_metadata: pd.DataFrame,
            species_key,
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

        # Prepare prediction adata
        adata_prediction = adata.copy()
        # Prepare species and covariates metadata for prediction
        species_parsed, cov_data_parsed, orders_dict, cov_dict = _prepare_metadata(
            meta_data=prediction_metadata, species_key=species_key,
            cov_cat_keys=self.adata.uns['covariates']['categorical'],
            cov_cont_keys=self.adata.uns['covariates']['continuous'],
            orders=self.adata.uns['orders']
        )
        adata_prediction.obsm['species_ratio'] = species_parsed
        adata_prediction.obsm['cov_species'] = cov_data_parsed

        # Prediction
        # Do not shuffle to retain order
        tensors_inference = self._make_data_loader(
            adata=adata_prediction, indices=indices, batch_size=batch_size, shuffle=False
        )
        tensors_generative = self._make_data_loader(
            adata=adata_prediction, indices=indices, batch_size=batch_size, shuffle=False
        )
        predicted = []
        # Pass jointly inference and generative data which differ in metadata
        for tensors_inference, tensors_generative in zip(tensors_inference, tensors_generative):
            # Inference
            inference_inputs = self.module._get_inference_input(tensors_inference)
            inference_outputs = self.module.inference(**inference_inputs)
            if give_mean:
                z = inference_outputs["qz_m"]
            else:
                z = inference_outputs["z"]
            inference_outputs = {'z': z}
            # Generative
            generative_inputs = self.module._get_generative_input(
                tensors=tensors_generative, inference_outputs=inference_outputs)
            generative_outputs = self.module.generative(**generative_inputs)
            if give_mean:
                x = generative_outputs["x_m"]
            else:
                x = self.module.sample_expression(**generative_outputs)
            predicted += [x]

        predicted = torch.cat(predicted)

        if as_numpy:
            predicted = predicted.cpu().numpy()
        return predicted

    @staticmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
            adata: AnnData,
            species_key: str = 'species',
            cov_cat_keys: list = [],
            cov_cont_keys: list = [],
            orthologues_map_key: str = 'orthologues',
            orders: Dict = {},
            seed: int = 0,
            alpha: float = 0.4,
            random_mixup_ratio: float = 1,
            similar_mixup_ratio: float = 1,
            n_hvgs: int = 2000,
            n_pcs: int = 15,
            k: int = 30,
    ) -> sc.AnnData:
        """
        Prepare adata for being used with the model and registers it with scvi.
        Also prepares mixup.
        :param adata: Adata. Adata should contain data from all species with genes not being
        merged across species.
        :param species_key: Species col name in obs (species of each cell) and var (species of each gene)
        :param cov_cat_keys: List of categorical covariates in obs.
        :param cov_cont_keys: List of continuous covariates in obs.
        :param orthologues_map_key: Key in uns of orthologue map of one-to-one orthologues of all species.
        A dataframe with n_cols=species and each row specifying one orthologue with values being var names.
        Should only contain genes present in adata.
        :param orders: Defined orders for species or categorical covariates. Dict with keys being
        'species' or categorical covariates names and values being lists of categories. May contain more/less
        categories than data.
        :param seed: Random seed for mixup generation. If None do not set.
        :param alpha: Alpha for mixup generation - higher leads to stronger mixing of cell pairs.
        :param random_mixup_ratio: Size of random cross-species mixup as ratio of original adata size.
        If 0 no mixup is performed.
        :param similar_mixup_ratio: Size of similar-cells cross-species mixup as ratio of original adata size.
        If 0 no mixup is performed.
        :param n_hvgs: N HVGs used for embedding of adata for similar-cell mixup.
        :param n_pcs: N PCs used for embedding of adata for similar-cell mixup.
        :param k: N nearest neighbours used for determining similar cells in similar-cell mixup.
        :return: Processed and registered adata.
        """
        # TODO add option to specify custom species and categories order

        # Adatas for merging
        adatas = []

        # Prepare original data
        # Make X a dense np array
        if sparse.issparse(adata.X):
            adata = sc.AnnData(X=adata.X.A,
                               obs=adata.obs.copy(deep=True),
                               var=adata.var.copy(deep=True),
                               uns=adata.uns.copy()
                               )
        adata_pp = sc.AnnData(adata.X, var=adata.var)

        # Prepare species and covariates metadata
        species_parsed, cov_data_parsed, orders_dict, cov_dict = _prepare_metadata(
            meta_data=adata.obs, species_key=species_key,
            cov_cat_keys=cov_cat_keys, cov_cont_keys=cov_cont_keys, orders=orders
        )
        # Species encoding
        species_order = orders_dict['species'].copy()
        species = adata.obs[species_key].values
        adata_pp.obsm['species_ratio'] = species_parsed
        # Covariate encoding
        adata_pp.obsm['cov_species'] = cov_data_parsed

        # Whether to eval only orthologues - always false here
        adata_pp.obsm['eval_o'] = np.array([0] * adata_pp.shape[0]).reshape(-1, 1)
        # Add orthologue info
        adata_pp.uns['orthologues'] = adata.uns[orthologues_map_key].copy()
        adatas.append(adata_pp)

        # Mixup preparation
        # Random corss-species mixup
        if random_mixup_ratio > 0:
            adata_random_species_mixup = _random_crosspecies_mixup(
                random_mixup_ratio=random_mixup_ratio, adata=adata_pp,
                species=species, species_order=species_order,
                alpha=alpha, seed=seed)
            adatas.append(adata_random_species_mixup)
        # Similar cross-species mixup
        if similar_mixup_ratio > 0:
            adata_similar_species_mixup = _similar_crosspecies_mixup(
                similar_mixup_ratio=similar_mixup_ratio, adata=adata_pp,
                species=species, species_order=species_order, alpha=alpha,
                n_hvgs=n_hvgs, k=k, n_pcs=n_pcs, seed=seed)
            adatas.append(adata_similar_species_mixup)

        # Combine adatas
        adata_combined = sc.concat(adatas)
        adata_combined.obs_names_make_unique()
        # Add extra info from processed adata
        adata_combined.var = adata_pp.var
        adata_combined.uns = adata_pp.uns
        adata_combined.uns['orders'] = orders_dict
        adata_combined.uns['covariates'] = cov_dict

        # Adata registration
        _setup_anndata(
            adata_combined)
        register_tensor_from_anndata(
            adata=adata_combined,
            adata_attr_name="obsm",
            adata_key_name='species_ratio',
            registry_key="species_ratio",
            is_categorical=False,
        )
        register_tensor_from_anndata(
            adata=adata_combined,
            adata_attr_name="obsm",
            adata_key_name='eval_o',
            registry_key="eval_o",
            is_categorical=False,
        )
        register_tensor_from_anndata(
            adata=adata_combined,
            adata_attr_name="obsm",
            adata_key_name='cov_species',
            registry_key="cov_species",
            is_categorical=False,
        )

        return adata_combined


def _prepare_metadata(meta_data: pd.DataFrame,
                      species_key,
                      cov_cat_keys: list = [],
                      cov_cont_keys: list = [],
                      orders=None):
    """

    :param meta_data: Dataframe containing species and covariate info, e.g. from non-registered adata.obs
    :param species_key: Name of species column in meta_data
    :param cov_cat_keys: List of categorical covariates column names.
    :param cov_cont_keys: List of continuous covariates column names.
    :param orders: Defined orders for species or categorical covariates. Dict with keys being
    'species' or categorical covariates names and values being lists of categories. May contain more/less
    categories than data.
    :return:
    """

    def dummies_categories(values: np.array, categories: Union[List, None] = None):
        """
        Make dummies of categorical covariates. Use specified order of categories.
        :param values: Categories for each observation.
        :param categories: Order of categories to use.
        :return: dummies, categories. Dummies - one-hot encoding of categories in same order as categories.
        """
        raise ValueError('TODO Falsely implemented categories - need adhere to the order')
        if categories is None:
            categories = pd.Categorical(values).categories.values
        dummies = pd.DataFrame(pd.get_dummies(values), columns=categories).fillna(0).values
        return dummies, list(categories)

    # Species encoding
    species_dummies, species_order = dummies_categories(
        values=meta_data[species_key], categories=orders.get('species', None))
    # Covariate encoding
    # Save order of covariates and categories
    cov_dict = {'categorical': cov_cat_keys, 'continuous': cov_cont_keys}
    orders_dict = {'species': species_order}
    # One-hot encoding of categorical covariates
    cov_cat_data = []
    for cov_cat_key in cov_cat_keys:
        cat_dummies, cat_order = dummies_categories(
            values=meta_data[cov_cat_key].values, categories=orders.get(cov_cat_key, None))
        cov_cat_data.append(cat_dummies)
        orders_dict[cov_cat_key] = cat_order
    # Prepare single cov array for all covariates and in per-species format
    cov_data_parsed = np.concatenate(cov_cat_data + [meta_data[cov_cont_keys].values], axis=1)
    cov_data_parsed = np.broadcast_to(np.expand_dims(cov_data_parsed, axis=1),
                                      (cov_data_parsed.shape[0], len(species_order), cov_data_parsed.shape[1]))
    return species_dummies, cov_data_parsed, orders_dict, cov_dict


def _create_mixup(indices, adata, species: list, species_order: list, obs_prefix: str,
                  seed=0, alpha=0.1,

                  ):
    """
    Prepare mixup data from adata and mixup indices.
    :param indices: Iter (cell pairs form mixup) of iters (cells in pairs) - 2 cells
    for mixup should be specified for each pair. Indices are obs positions from adata.
    :param adata: Adata used for making the mixup. Should have
    col species in var and cov_species in obsm.
    :param species: List of species corresponding to adata obs
    :param species_order: Order of species, list of names
    :param obs_prefix: Prefix added to mixup obs names
    :param seed: Set seed for mixup ratio generation. If None does not set the seed.
    :param alpha: Alpha for mixup
    :return: Mixup adata
    """
    # Data for building adata
    xs = []
    covs = []
    species_ratios = []
    obs_names = []
    if seed is not None:
        np.random.seed(seed)
    species_genes = adata.var['species'].values
    for i, j in indices:
        mixup_ratio_i = np.random.beta(alpha, alpha)
        mixup_ratio_j = 1 - mixup_ratio_i
        species_i = species[i]
        species_j = species[j]
        # Get expression, expression of unused species genes will be set to 0
        x_i = adata[i, :].X.copy().ravel()
        x_i[species_genes != species_i] = 0
        x_j = adata[j, :].X.copy().ravel()
        x_j[species_genes != species_j] = 0
        xs.append(x_i + x_j)
        cov_i = adata.obsm['cov_species'][i, 0]
        cov_j = adata.obsm['cov_species'][j, 0]
        # For species that are not being validated just set cov to mixup ratio,
        # but this is not very relevant for the model as it is not being validated
        cov_ij = cov_i * mixup_ratio_i + cov_j * mixup_ratio_j
        covs.append(np.array([cov_i, cov_j] + [cov_ij] * (len(species_order) - 2)))
        species_ratio = np.zeros(len(species_order))
        species_ratio[np.array(species_order) == species_i] = mixup_ratio_i
        species_ratio[np.array(species_order) == species_j] = mixup_ratio_j
        species_ratios.append(species_ratio)
        obs_names.append('_'.join([obs_prefix, str(i), str(j)]))
    adata_mixup = sc.AnnData(
        X=pd.DataFrame(np.array(xs), index=obs_names, columns=adata.var_names),
        obsm={'cov_species': np.array(covs), 'species_ratio': np.array(species_ratios)}
    )

    return adata_mixup


def _count_species_pairs(adata, description: str, species_order: list):
    """
    Counts N species pairs per species combination.
    Some mixup cells may seem to be from single species as alpha was 0 or 1.
    :param adata: Adata with species_ratio in obsm
    :param description: For printing out, adata name/description
    :param species_order: Order of species, list of names
    """
    species_pairs = dict()
    for row_idx in range(adata.obsm['species_ratio'].shape[0]):
        species_idxs = np.argwhere(adata.obsm['species_ratio'][row_idx] > 0).ravel()
        pair_name = ' and '.join([species_order[idx] for idx in species_idxs])
        if pair_name not in species_pairs:
            species_pairs[pair_name] = 0
        species_pairs[pair_name] = species_pairs[pair_name] + 1
    print('N pairs per species combination for', description, ':', species_pairs)


def _random_crosspecies_mixup(random_mixup_ratio: float, adata,
                              species: list, species_order: list,
                              alpha: float, seed: int = 0):
    """
    Random cross-species mixup
    :param random_mixup_ratio: Size of mixup as ratio of cells from adata
    :param adata: Adata to use for mixup, see _create_mixup for what it needs to contain
    :param species: List of species matching adata obs
    :param species_order: Order of species, list of names
    :param alpha: Mixup param
    :param seed: random seed, If None not used
    :return: mixup adata
    """
    desired_n = int(random_mixup_ratio * adata.shape[0])
    random_mixup_idx = set()
    idxs = list(range(adata.shape[0]))
    if seed is not None:
        np.random.seed(seed)
    # Try to generate N random corss-species pairs
    # TODO may be problematic as could run indefinitely, for now a quick fix is added
    tries = 0
    while len(random_mixup_idx) < desired_n and \
            tries < adata.shape[0] * 10:  # Quick fix to stop if can not find combionations
        # Randomly sample cells and make sure that species differ
        # TODO could make quicker by selecting random cell pairs from pairs
        # of species directly
        i = np.random.choice(idxs)
        j = np.random.choice(idxs)
        species_i = species[i]
        species_j = species[j]
        # TODO could add check not to use same cell 2x, but maybe less important as
        #  mixup ratio will differ
        if species_i != species_j:
            # Could be used for checking that same cell par was not used before
            random_mixup_idx.add(frozenset((i, j)))
        tries += 1
    if len(random_mixup_idx) < desired_n:
        warnings.warn('Found less than desired  number of random mixup samples.')
        print('Found %i/%i random mixup samples' % (
            len(random_mixup_idx), desired_n))
    # Make adata from selected mixup cells
    adata_random_species_mixup = _create_mixup(indices=random_mixup_idx,
                                               adata=adata,
                                               species=species, species_order=species_order,
                                               obs_prefix='mixup_species_random',
                                               seed=seed, alpha=alpha)
    _count_species_pairs(adata_random_species_mixup, species_order=species_order,
                         description='random corss-species mixup')
    adata_random_species_mixup.obsm['eval_o'] = np.array([1] * adata_random_species_mixup.shape[0]
                                                         ).reshape(-1, 1)
    return adata_random_species_mixup


def _similar_crosspecies_mixup(similar_mixup_ratio: float, adata,
                               species: list, species_order: list, alpha: float,
                               n_hvgs: int = 2000, k: int = 30, n_pcs: int = 15,
                               seed: int = 0):
    """
    Similar cells cross-species mixup.
    Find similar cells by embedding adata into shared PCA space and then for each pair of species finding
    mutual nearest neighbours (contained within neighbourhood).
    :param similar_mixup_ratio:
    :param adata: Adata, should contain orthologues mapping in uns and fields specified in _create_mixup
    :param species: List of species matching adata obs
    :param species_order: Order of species, list of names
    :param alpha: Mixup param
    :param k: N neighbours
    :param n_pcs: N PCs
    :param seed: random seed, If None not used
    :return: mixup adata
    """
    # Map for transofming input epxression to orthologues only
    # matmul(X,otm) = expression of orthologues only, combined by orthologuesacross species
    # dim otm = n_genes * n_orthologues (summarized across species)
    orthologues_transform_map = np.zeros((adata.shape[1], adata.uns['orthologues'].shape[0]))
    for idx, (name, data) in enumerate(adata.uns['orthologues'].iterrows()):
        for gene in data:
            orthologues_transform_map[np.argwhere(adata.var_names == gene), idx] = 1

    # Similar cells cross-species mixup
    # For all pairs of species find shared neighbours and then pick randomly
    # cell pairs from all possible cell pairs
    desired_n = int(similar_mixup_ratio * adata.shape[0])

    # Adata mapped to orthologues summarised across species
    adata_orthologues = sc.AnnData(np.matmul(adata.X, orthologues_transform_map),
                                   obs=pd.DataFrame({'species': species}, index=adata.obs_names))
    # Compute embedding
    sc.pp.highly_variable_genes(adata_orthologues,
                                # Can only compute as many HVGs as there are genes
                                n_top_genes=min([n_hvgs, adata_orthologues.shape[1]]),
                                flavor='cell_ranger', subset=True, inplace=True,
                                batch_key='species')
    sc.pp.scale(adata_orthologues)
    sc.pp.pca(adata_orthologues, n_comps=n_pcs, zero_center=True)

    # Neighbours for all species pairs
    pairs_all = []
    for i in range(len(species_order) - 1):
        for j in range(i + 1, len(species_order)):
            # Prepare species data
            s_i = species_order[i]
            s_j = species_order[j]
            e_i = adata_orthologues[adata_orthologues.obs['species'] == s_i, :].obsm['X_pca']
            e_j = adata_orthologues[adata_orthologues.obs['species'] == s_j, :].obsm['X_pca']
            # On which position was originally each cell
            idxs_i = np.argwhere(adata_orthologues.obs['species'].values == s_i).ravel()
            idxs_j = np.argwhere(adata_orthologues.obs['species'].values == s_j).ravel()
            # KNN
            index_i = NNDescent(e_i, metric='correlation', n_jobs=-1)
            neighbours_j, distances_j = index_i.query(e_j, k=k)
            index_j = NNDescent(e_j, metric='correlation', n_jobs=-1)
            neighbours_i, distances_i = index_j.query(e_i, k=k)
            neighbours = np.zeros((e_i.shape[0], e_j.shape[0]))
            # Parse KNN - count if presnet for both directions
            pairs = {}
            for cell_j in range(neighbours_j.shape[0]):
                for idx_i in range(k):
                    cell_i = neighbours_j[cell_j][idx_i]
                    pair = str(cell_i) + '_' + str(cell_j)
                    if pair not in pairs:
                        pairs[pair] = 0
                    pairs[pair] = pairs[pair] + 1
            for cell_i in range(neighbours_i.shape[0]):
                for idx_j in range(k):
                    cell_j = neighbours_i[cell_i][idx_j]
                    pair = str(cell_i) + '_' + str(cell_j)
                    if pair not in pairs:
                        pairs[pair] = 0
                    pairs[pair] = pairs[pair] + 1
            # Get shared neighbors based on counts of directions
            for pair, n in pairs.items():
                if n == 2:
                    idx_i = int(pair.split('_')[0])
                    idx_j = int(pair.split('_')[1])
                    # Map neighbors to original indices
                    pairs_all.append((idxs_i[idx_i], idxs_j[idx_j]))

    # Subset to desired N of pairs
    if seed is not None:
        np.random.seed(seed)
    pairs_all = np.array(pairs_all)
    print('Found %i similar cell pairs across species' % pairs_all.shape[0])
    pairs_idx = np.random.choice(range(pairs_all.shape[0]), size=desired_n, replace=True)
    pairs_all = pairs_all[pairs_idx]

    # Create adata
    adata_similar_species_mixup = _create_mixup(
        indices=pairs_all,
        adata=adata,
        species=species, species_order=species_order,
        obs_prefix='mixup_species_similar',
        seed=seed, alpha=alpha
    )
    _count_species_pairs(adata_similar_species_mixup, species_order=species_order,
                         description='similar corss-species mixup')
    adata_similar_species_mixup.obsm['eval_o'] = np.array([0] * adata_similar_species_mixup.shape[0]
                                                          ).reshape(-1, 1)
    return adata_similar_species_mixup
