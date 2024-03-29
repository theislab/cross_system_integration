import logging
from typing import List, Optional, Union, Sequence, Dict
from typing_extensions import Literal
import pandas as pd
import numpy as np
import torch

from anndata import AnnData
from scvi import REGISTRY_KEYS
from scvi.data import AnnDataManager
from scvi.data.fields import (
    LayerField, ObsmField,
)
from scvi.model.base import BaseModelClass, VAEMixin
from scvi.utils import setup_anndata_dsp

from cross_system_integration.model._training import TrainingMixin
from cross_system_integration.module._priors import VampPrior, GaussianMixtureModelPrior
from cross_system_integration.module._xxjointmodule import XXJointModule
from cross_system_integration.model._gene_maps import GeneMapInput
from cross_system_integration.model._utils import prepare_metadata

logger = logging.getLogger(__name__)


class XXJointModel(VAEMixin, TrainingMixin, BaseModelClass):

    def __init__(
            self,
            adata: AnnData,
            mixup_alpha: Optional[float] = None,
            system_decoders: bool = False,
            prior: Literal["standard_normal", "vamp", "gmm"] = 'standard_normal',
            n_prior_components=100,
            trainable_priors=True,
            encode_pseudoinputs_on_eval_mode=False,
            pseudoinputs_data_init: bool = True,
            pseudoinputs_data_indices: Optional[np.array] = None,
            adata_eval: Optional[AnnData] = None,
            **model_kwargs,
    ):
        """
        Architecture with a single encoder and decoder for two systems

        Parameters
        ----------
        adata
            AnnData object that has been registered via :meth:`setup_anndata`.
        mixup_alpha
            Alpha for mixup. If None dont use mixup. Should be set to None.
        system_decoders
            Separate decoders for each system . Should be False
        prior
            The prior model to be used. You can choose between "standard_normal", "vamp", and "gmm".
        n_prior_components
            Number of prior components in multimodal prior. Only used if you use multimodal prior
        trainable_priors
            Whether the priors in VAMP are trainable or not.
        encode_pseudoinputs_on_eval_mode
            Pseudoinputs are in the data space and are transferred to the latent spase on each iteration.
            Set this to True if you want this transfer to be made in eval mode (no dropout and batchnorm).
        pseudoinputs_data_init
            Set to True if you want pseudoinputs to be chosen randomly from the input adata.
            Otherwise they will be initiated randomly.
            Always set to true for VAMP
        pseudoinputs_data_indices
            By default (based on pseudoinputs_data_init), pseudoinputs are randomly selected from data.
            Alternatively, one can feed pseudoinputs using this parameter. The input should be a numpy array.
        adata_eval
            Adata used for eval metric computation.
            Should be set up as the main adata and have eval info in uns['eval_info'] given as
            dict(metric_name:dict(cells_in, switch_ system, cells_target, genes)) where metric_name is name of metric
            cells_in are obs names
            to be used for prediction (list), switch_system determines if system is switched in prediction (dict),
            cells_target are obs names to  evaluate prediction against (list),
            genes - genes to use for evaluation (list/set).
            List specifies multiple eval settings. Eval metric is correlation between mean of predicted and target cells
            on the specified genes. Uses null cov for predicted cells.
        model_kwargs
            Passed to model
            TODO add here params required by model
        """
        super(XXJointModel, self).__init__(adata)

        use_group = 'group' in adata.obsm

        if pseudoinputs_data_init:
            if pseudoinputs_data_indices is None:
                pseudoinputs_data_indices = np.random.randint(0, adata.shape[0], n_prior_components)
            pseudoinput_data = next(iter(self._make_data_loader(
                adata=adata, indices=pseudoinputs_data_indices, batch_size=n_prior_components, shuffle=False)))
        else:
            pseudoinput_data = None

        # self.summary_stats provides information about anndata dimensions and other tensor info
        self.module = XXJointModule(
            n_input=adata.var['input'].sum(),
            n_output=adata.shape[1],
            system_decoders=system_decoders,
            gene_map=GeneMapInput(adata=adata),
            n_cov=adata.obsm['covariates'].shape[1],
            n_system=adata.obsm['system'].shape[1],
            use_group=use_group,
            mixup_alpha=mixup_alpha,
            prior=prior,
            n_prior_components=n_prior_components,
            trainable_priors=trainable_priors,
            encode_pseudoinputs_on_eval_mode=encode_pseudoinputs_on_eval_mode,
            pseudoinput_data=pseudoinput_data,
            data_eval=self._prepare_eval_data(adata_eval) if adata_eval is not None else None,
            **model_kwargs)

        self._model_summary_string = "Overwrite this attribute to get an informative representation for your model"
        # necessary line to get params that will be used for saving/loading
        self.init_params_ = self._get_init_params(locals())
        self.init_params_['use_group'] = use_group

        logger.info("The model has been initialized")

    def _prepare_eval_data(self, adata: AnnData):
        """
        Prepare evaluation inputs
        :param adata: set up as training adata and with eval_info in uns
        :return:
        """
        adata = self._validate_anndata(adata)
        eval_data = {}
        for metric_name, eval_case_info in adata.uns['eval_info'].items():
            eval_case_data = {}
            obs_idx = dict(zip(adata.obs_names, range(adata.shape[0])))
            indices_in = [obs_idx[o] for o in eval_case_info['cells_in']]
            n_in = len(indices_in)
            eval_case_data['inference_tensors'] = next(iter(self._make_data_loader(
                adata=adata, indices=indices_in, batch_size=n_in, shuffle=False)))
            eval_case_data['generative_cov'] = torch.zeros(n_in, adata.obsm['covariates'].shape[1])
            x_x, x_y, pred_key = (False, True, 'y') if eval_case_info['switch_system'] else (True, False, 'x')
            eval_case_data['generative_kwargs'] = {'x_x': x_x, 'x_y': x_y, 'pred_key': pred_key}
            # Note: ordering of selected genes will match adata not eval_info gene list
            eval_genes = set(eval_case_info['genes'])
            eval_genes_bool = torch.tensor([g in eval_genes for g in adata.var_names])
            eval_case_data['genes'] = eval_genes_bool
            eval_case_data['target_x_m'] = torch.tensor(adata[eval_case_info['cells_target'], :].to_df().values
                                                        )[:, eval_genes_bool].mean(axis=0)
            eval_case_data['target_x_std'] = torch.tensor(adata[eval_case_info['cells_target'], :].to_df().values
                                                          )[:, eval_genes_bool].std(axis=0)
            eval_data[metric_name] = eval_case_data
        return eval_data

    @torch.no_grad()
    def translate(
            self,
            adata: AnnData,
            switch_system: bool = True,
            indices: Optional[Sequence[int]] = None,
            give_mean: bool = True,
            give_var: bool = False,
            covariates: Optional[Union[pd.Series, pd.DataFrame]] = None,
            batch_size: Optional[int] = None,
            as_numpy: bool = True
    ) -> (Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]):
        # TODo this is mainly the same as evaluation metrics data prep and computation so maybe combine
        """
        Translate expression - based on input expression and metadata
        predict expression of data as if it had given metadata.
        expression, metadata (from adata) -> latent
        latent, new metadata -> translated expression
        :param adata: Input adata based on which latent representation is obtained. Covariates are not used from here,
        see covariates param.
        :param switch_system: Should in translation system be switched or not
        :param indices: Adata indices to use
        :param give_mean: In latent and expression prediction use mean rather than samples
        :param give_var: Also return var besides mean/sampled
        :param covariates: Covariates to be used for data generation. Can be None (uses all-0 covariates) or a series
        with covariate metadata (same for all predicted samples, dim=n_cov*1) or dataframe (matching adata).
        :param batch_size: batch size for computation
        :param as_numpy:  Move output tensor to cpu and convert to numpy
        :return: mean/sample and optional var
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
        predicted_var = []
        x_x, x_y, pred_key = (False, True, 'y') if switch_system else (True, False, 'x')
        # Specify cov to use, determine cov size below as here not all batches are of equal size since we predict
        # each sample 1x
        if isinstance(covariates, pd.DataFrame):
            covariates = covariates.iloc[indices, :]
        cov_replace = self._make_covariates(adata=adata, batch_size=indices.shape[0], cov_template=covariates)
        idx_previous = 0
        for tensors in tensors_fwd:
            # Inference
            idx_next = idx_previous + tensors['covariates'].shape[0]
            inference_inputs = self.module._get_inference_input(tensors)
            generative_inputs = self.module._get_generative_input(
                tensors=tensors,
                inference_outputs=self.module.inference(**inference_inputs),
                cov_replace=cov_replace[idx_previous:idx_next, :])
            generative_outputs = self.module.generative(**generative_inputs, x_x=x_x, x_y=x_y)
            if give_mean:
                pred_sub = generative_outputs[pred_key + "_m"]
            else:
                pred_sub = generative_outputs[pred_key]
            predicted += [pred_sub]
            if give_var:
                predicted_var += [generative_outputs[pred_key + "_v"]]
            idx_previous = idx_next
        predicted = torch.cat(predicted)
        if give_var:
            predicted_var = torch.cat(predicted_var)

        if as_numpy:
            predicted = predicted.cpu().numpy()
            if give_var:
                predicted_var = predicted_var.cpu().numpy()

        if give_var:
            return predicted, predicted_var
        else:
            return predicted

    def _make_covariates(self, adata, batch_size,
                         cov_template: Optional[Union[pd.Series, pd.DataFrame]] = None) -> torch.Tensor:
        """
        Make covariate tensor corresponding to covariate features of an already registered adata
        Base covariates on a metadata series or empty (zeros).
        :param adata: Adata (already registered/validated) to which covariates should correspond
        :param batch_size: If cov_template is series, Number of samples, all will have same values
        :param cov_template: Series (same for all samples) or DF with metadata for covariates,
        if None creates all-0 covariates
        :return: Covariates tensor corresponding to covariates features in adata
        """
        if cov_template is None:
            cov = torch.zeros((batch_size, adata.obsm['covariates'].shape[1]), device=self.device)
        else:
            cov, _, _ = prepare_metadata(
                meta_data=cov_template.to_frame().T if isinstance(cov_template, pd.Series) else cov_template,
                cov_cat_keys=adata.uns['covariates_dict']['categorical'],
                cov_cont_keys=adata.uns['covariates_dict']['continuous'],
                orders=adata.uns['covariate_orders'])
            cov = torch.tensor(cov.values.astype(np.float32), device=self.device)
            if isinstance(cov_template, pd.Series):
                cov = cov.expand(batch_size, -1)
        return cov

    @torch.no_grad()
    def embed(
            self,
            adata: AnnData,
            indices: Optional[Sequence[int]] = None,
            cycle: bool = False,
            give_mean: bool = True,
            batch_size: Optional[int] = None,
            as_numpy: bool = True
    ) -> (Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]):
        """
        Produce latent embedding
        expression, metadata (from adata) -> latent
        :param adata: Input adata based on which latent representation is obtained.
        :param indices: Adata indices to use
        :param cycle: Return latent from cycle
        :param give_mean: In latent prediction use mean rather than samples
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
            inference_outputs = self.module.inference(**inference_inputs)
            if cycle:
                generative_inputs = self.module._get_generative_input(tensors, inference_outputs)
                generative_outputs = self.module.generative(**generative_inputs, x_x=False, x_y=True)
                inference_cycle_inputs = self.module._get_inference_cycle_input(
                    tensors=tensors, generative_outputs=generative_outputs)
                inference_outputs = self.module.inference(**inference_cycle_inputs)
            if give_mean:
                predicted += [inference_outputs['z_m']]
            else:
                predicted += [inference_outputs['z']]

        predicted = torch.cat(predicted)

        if as_numpy:
            predicted = predicted.cpu().numpy()
        return predicted

    @torch.no_grad()
    def get_prior_probs(
            self,
            adata: AnnData,
            indices: Optional[Sequence[int]] = None,
            batch_size: Optional[int] = None,
            as_numpy: bool = True
    ) -> (Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]):
        """
        Compute the prior probabilities for a given set of cells in an AnnData object.
        Calculates the prior probabilities for each observation based on the model's prior distribution.
    
        Args:
            adata: AnnData containing the expression of the cells to compute prior probabilities for.
            indices: Indices of observations to consider (default is all observations).
            batch_size: The batch size for processing data (default is None).
            as_numpy: If True, return the result as a NumPy array; otherwise, return a PyTorch tensor (default is True).
    
        Returns:
            Union[np.ndarray, torch.Tensor]: Array of size n_obs x n_prior containing a distribution over priors for each observation.
            Note that prior probabilities also depend on the learned weight of each prior.
            If `as_numpy` is True, the result is a NumPy array; otherwise, it's a PyTorch tensor.
        """
        # Check model and adata
        assert isinstance(self.module.prior, VampPrior) or isinstance(self.module.prior, GaussianMixtureModelPrior)
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
            inference_outputs = self.module.inference(**inference_inputs)
            z = inference_outputs['z_m']
            log_p_c_by_z = self.module.prior.log_prob_per_mod(z, mixture_weights=True)  # K x N x L
            p_c_by_z = torch.exp(log_p_c_by_z.sum(axis=2).permute([1, 0]))
            predicted.append(p_c_by_z)

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
            group_key: Optional[str] = None,
            input_gene_key: Optional[str] = None,
            layer: Optional[str] = None,
            categorical_covariate_keys: Optional[List[str]] = None,
            continuous_covariate_keys: Optional[List[str]] = None,
            covariate_orders: Optional[Dict] = None,
            **kwargs,
    ) -> Optional[AnnData]:
        """
        Setup anndata for model
        Parameters
        ----------
        adata
            Adata to setup
        system_key
            obs col of system
        group_key
            obs col of cell type (used for contrastive learning)
        input_gene_key
            var col used to define which genes should be used (should be 0/1 for F/T). If None use all genes
        layer
            X or layer with expression
        categorical_covariate_keys
            Categorical covariates to correct for besides system, given as obs columns
        continuous_covariate_keys
            Continous covariates to correct for besides system, given as obs columns
        covariate_orders
            Information on covariate order in the parsed covariate data, if None will be created during setup
        kwargs
            For AnnDataManager.register_fields
        Returns
        -------

        """
        setup_method_args = cls._get_setup_method_args(**locals())

        # Make sure var names are unique
        if adata.shape[1] != len(set(adata.var_names)):
            raise ValueError('Adata var_names are not unique')

        # User should copy its anndata itself if does not telerate changes.
        # adata = adata.copy()

        # Make system to categorical for cov
        if adata.obs[system_key].nunique() != 2:
            raise ValueError('There must be exactly two systems')
        # TODO add option to specify systems dict/order
        system_order = sorted(adata.obs[system_key].unique())
        systems_dict = dict(zip(system_order, [0.0, 1.0]))
        adata.uns['system_order'] = system_order
        adata.obsm['system'] = adata.obs[system_key].map(systems_dict).to_frame()

        # Remove any "group" column from obs (if group_key is None) as this is used to determine
        # if group info will be used
        if group_key is None and 'group' in adata.obsm:
            del adata.obsm['group']
        # Maps groups to numerical (int) as else data loading cant make tensors
        if group_key is not None:
            group_order = sorted(adata.obs[group_key].dropna().unique())
            group_dict = dict(zip(group_order, list(range(len(group_order)))))
            adata.uns['group_order'] = group_order
            adata.obsm['group'] = adata.obs[group_key].map(group_dict).to_frame()

        # Which genes to use as input
        if input_gene_key is None:
            adata.var['input'] = 1
        else:
            adata.var['input'] = adata.var[input_gene_key]

        # Set up covariates
        # TODO this could be handled by specific field type in registry
        if covariate_orders is None:
            covariate_orders = {}

        # System and group must not be in cov
        if categorical_covariate_keys is not None:
            if group_key in categorical_covariate_keys or system_key in categorical_covariate_keys:
                raise ValueError('group_key or system_key should not be within covariate keys')
        if continuous_covariate_keys is not None:
            if group_key in continuous_covariate_keys or system_key in continuous_covariate_keys:
                raise ValueError('group_key or system_key should not be within covariate keys')

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
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=False),
            ObsmField('covariates', 'covariates'),
            ObsmField('system', 'system'),
        ]
        if group_key is not None:
            anndata_fields.append(
                ObsmField('group', 'group'),
            )
        adata_manager = AnnDataManager(
            fields=anndata_fields, setup_method_args=setup_method_args
        )
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

        return adata
