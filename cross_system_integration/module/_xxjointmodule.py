import itertools
from collections import defaultdict
from typing import Optional, Union, Tuple, Dict
from typing_extensions import Literal
import numpy as np

import torch
from scvi import REGISTRY_KEYS
from scvi.module.base import BaseModuleClass, auto_move_data
from torch.distributions import Normal
from torch.distributions import kl_divergence

from cross_system_integration.model._gene_maps import GeneMapInput
from cross_system_integration.nn._base_components import EncoderDecoder
from cross_system_integration.module._loss_recorder import LossRecorder
from cross_system_integration.module._utils import *
from cross_system_integration.module._priors import StandardPrior, VampPrior, GaussianMixtureModelPrior

torch.backends.cudnn.benchmark = True


class XXJointModule(BaseModuleClass):
    """
    Skeleton Variational auto-encoder model.

    Here we implement a basic version of scVI's underlying VAE [Lopez18]_.
    This implementation is for instructional purposes only.

    Parameters
    ----------
    n_input
        Number of input genes
    library_log_means
        1 x n_batch array of means of the log library sizes. Parameterizes prior on library size if
        not using observed library size.
    library_log_vars
        1 x n_batch array of variances of the log library sizes. Parameterizes prior on library size if
        not using observed library size.
    n_batch
        Number of batches, if 0, no batch correction is performed.
    n_hidden
        Number of nodes per hidden layer
    n_latent
        Dimensionality of the latent space
    n_layers
        Number of hidden layers used for encoder and decoder NNs
    dropout_rate
        Dropout rate for neural networks
    """

    def __init__(
            self,
            n_input: int,
            n_output: int,
            gene_map: GeneMapInput,
            n_cov: int,
            n_system: int,  
            prior: Literal["standard_normal", "vamp", "gmm"] = 'vamp',
            n_prior_components=5,
            trainable_priors=True,
            encode_pseudoinputs_on_eval_mode=True,
            pseudoinput_data=None,
            z_dist_metric: str = 'MSE_standard',
            n_latent: int = 15,
            n_hidden: int = 256,
            n_layers: int = 2,
            dropout_rate: float = 0.1,
            out_var_mode: str = 'feature',
            **kwargs
    ):
        super().__init__()

        # self.gene_map = gene_map
        # TODO transfer functionality of gene maps to not need the class itself needed anymore -
        #  it was used only to return tensors on correct device for this specific model type
        self.register_buffer('gm_input_filter', gene_map.input_filter(), persistent=False)
        self.n_output = n_output
        self.z_dist_metric = z_dist_metric
        self.data_eval = None

        n_cov_encoder = n_cov + n_system

        self.encoder = EncoderDecoder(
            n_input=n_input,
            n_output=n_latent,
            n_cov=n_cov_encoder,
            n_hidden=n_hidden,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            sample=True,
            var_mode='sample_feature',
            **kwargs
        )

        self.decoder = EncoderDecoder(
            n_input=n_latent,
            n_output=n_output,
            n_cov=n_cov_encoder,
            n_hidden=n_hidden,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            sample=True,
            var_mode=out_var_mode,
            **kwargs
            )
        
        if prior == 'standard_normal':
            self.prior = StandardPrior()
        elif prior == 'vamp':
            if pseudoinput_data is not None:
                pseudoinput_data = self._get_inference_input(pseudoinput_data)
            self.prior = VampPrior(n_components=n_prior_components, n_input=n_input, n_cov=n_cov_encoder,
                                   encoder=self.encoder,
                                   data=(pseudoinput_data['expr'],
                                         self._merge_cov(cov=pseudoinput_data['cov'],
                                                         system=pseudoinput_data['system'])),
                                   trainable_priors=trainable_priors,
                                   encode_pseudoinputs_on_eval_mode=encode_pseudoinputs_on_eval_mode,
                                   )
        elif prior == 'gmm':
            if pseudoinput_data is not None:
                pseudoinput_data = self._get_inference_input(pseudoinput_data)
                original_mode = self.encoder.training
                self.encoder.train(False)
                encoded_pseudoinput_data = self.encoder(
                    x=pseudoinput_data['expr'],
                    cov=self._merge_cov(cov=pseudoinput_data['cov'], system=pseudoinput_data['system'])
                )
                self.encoder.train(original_mode)
                encoded_pseudoinput_data = encoded_pseudoinput_data['y_m'], encoded_pseudoinput_data['y_v']
            else:
                encoded_pseudoinput_data = None
            self.prior = GaussianMixtureModelPrior(
                n_components=n_prior_components, n_latent=n_latent,
                data=encoded_pseudoinput_data,
                trainable_priors=trainable_priors,
            )
        else:
            raise ValueError('Prior not recognised')

    @auto_move_data
    def _get_inference_input(self, tensors, **kwargs):
        """Parse the dictionary to get appropriate args"""
        input_features = torch.ravel(torch.nonzero(self.gm_input_filter))
        expr = tensors[REGISTRY_KEYS.X_KEY][:, input_features]
        cov = tensors['covariates']
        system = tensors['system']
        input_dict = dict(expr=expr, cov=cov, system=system)
        return input_dict

    @auto_move_data
    def _get_inference_cycle_input(self, tensors, generative_outputs, selected_system, **kwargs):
        """Parse the dictionary to get appropriate args"""
        input_features = torch.ravel(torch.nonzero(self.gm_input_filter))
        expr = generative_outputs['y_m'][:, input_features]
        cov = self._mock_cov(tensors['covariates'])
        system = selected_system
        input_dict = dict(expr=expr, cov=cov, system=system)
        return input_dict

    @auto_move_data
    def _get_generative_input(self, tensors, inference_outputs, selected_system, cov_replace: torch.Tensor = None,  **kwargs):
        """
        Parse the dictionary to get appropriate args
        :param cov_replace: Replace cov from tensors with this covariate vector
        """
        z = inference_outputs["z"]
        if cov_replace is None:
            cov = {'x': tensors['covariates'], 'y': self._mock_cov(tensors['covariates'])}
        else:
            cov = {'x': cov_replace, 'y': cov_replace}

        system = {'x': tensors['system'], 'y': selected_system}
        input_dict = dict(z=z, cov=cov, system=system)
        return input_dict
 
    @auto_move_data
    def _get_generative_cycle_input(self, tensors, inference_cycle_outputs, selected_system, **kwargs):
        raise NotImplementedError

    @staticmethod
    def _merge_cov(cov, system):
        return torch.cat([cov, system], dim=1)

    @staticmethod
    def _mock_cov(cov):
        return torch.zeros_like(cov)

    @auto_move_data
    def inference(self, expr, cov, system):
        """
        expression & cov -> latent representation
        All inputs (expr, cov, train) are split by x and y (two modalities)
        :param expr:
        :param cov:
        :param train: Should samples be used for training
        :param mask_subset: Do prediction only for samples that are supposed to be used for training (train!=0)
        :param pad_output: Pad (zeros) the predicted samples to get back to the non-masked number of samples
        :return:
        """

        z = self.encoder(x=expr, cov=self._merge_cov(cov=cov, system=system))
        return dict(z=z['y'], z_m=z['y_m'], z_v=z['y_v'])

    @auto_move_data
    def generative(self, z, cov, system, x_x=True, x_y=True):
        """
        latent representation & convariates -> expression
        :param z:
        :param cov:
        :param train: Should samples be used for training
        :param z_masked: Was z already predicted for only a subset (masked)
        :param mask_subset: Do prediction only for samples that are supposed to be used for training (train!=0)
        :param pad_output: Pad (zeros) the predicted samples to get back to the non-masked number of samples
        :param x_x: Should this prediction be computed
        :param x_y: Should this prediction be computed
        :param y_y: Should this prediction be computed
        :param y_x: Should this prediction be computed
        :return:
        """

        def outputs(compute, name, res, x, cov, system):
            if compute:
                res_sub = self.decoder(x=x, cov=self._merge_cov(cov=cov, system=system))
                res[name] = res_sub['y']
                res[name + '_m'] = res_sub['y_m']
                res[name + '_v'] = res_sub['y_v']

        res = {}
        outputs(compute=x_x, name='x', res=res, x=z, cov=cov['x'], system=system['x'])
        outputs(compute=x_y, name='y', res=res, x=z, cov=cov['y'], system=system['y'])
        return res

    @auto_move_data
    def forward(
            self,
            tensors,
            get_inference_input_kwargs: Optional[dict] = None,
            get_generative_input_kwargs: Optional[dict] = None,
            inference_kwargs: Optional[dict] = None,
            generative_kwargs: Optional[dict] = None,
            loss_kwargs: Optional[dict] = None,
            compute_loss=True,
    ) -> Union[
        Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
        Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], LossRecorder],
    ]:
        """
        Forward pass through the network.
        Parameters
        ----------
        tensors
            tensors to pass through
        get_inference_input_kwargs
            Keyword args for ``_get_inference_input()``
        get_generative_input_kwargs
            Keyword args for ``_get_generative_input()``
        inference_kwargs
            Keyword args for ``inference()``
        generative_kwargs
            Keyword args for ``generative()``
        loss_kwargs
            Keyword args for ``loss()``
        compute_loss
            Whether to compute loss on forward pass. This adds
            another return value.
        """
        """Core of the forward call shared by PyTorch- and Jax-based modules."""

        # TODO currently some forward paths are computed despite potentially having loss weight=0 -
        #  don't compute if not needed
        inference_kwargs = _get_dict_if_none(inference_kwargs)
        generative_kwargs = _get_dict_if_none(generative_kwargs)
        loss_kwargs = _get_dict_if_none(loss_kwargs)
        get_inference_input_kwargs = _get_dict_if_none(get_inference_input_kwargs)
        get_generative_input_kwargs = _get_dict_if_none(get_generative_input_kwargs)

        # Inference
        inference_inputs = self._get_inference_input(
            tensors, **get_inference_input_kwargs
        )
        inference_outputs = self.inference(**inference_inputs, **inference_kwargs)
        # Generative
        selected_system =  self.random_select_systems(tensors['system'])
        generative_inputs = self._get_generative_input(
            tensors, inference_outputs, selected_system=selected_system, **get_generative_input_kwargs
        )
        generative_outputs = self.generative(**generative_inputs, x_x=True, x_y=True, **generative_kwargs)    
        # Inference cycle
        inference_cycle_inputs = self._get_inference_cycle_input(
            tensors=tensors, generative_outputs=generative_outputs, selected_system=selected_system, **get_inference_input_kwargs)
        inference_cycle_outputs = self.inference(**inference_cycle_inputs, **inference_kwargs)
        # Combine outputs of all forward passes
        inference_outputs_merged = dict(**inference_outputs)
        inference_outputs_merged.update(
            **{k.replace('z', 'z_cyc'): v for k, v in inference_cycle_outputs.items()})
        generative_outputs_merged = dict(**generative_outputs)
       

        if compute_loss:
            losses = self.loss(
                tensors=tensors,
                inference_outputs=inference_outputs_merged,
                generative_outputs=generative_outputs_merged,
                **loss_kwargs
            )
            return inference_outputs_merged, generative_outputs_merged, losses
        else:
            return inference_outputs_merged, generative_outputs_merged

    def loss(
            self,
            tensors,
            inference_outputs,
            generative_outputs,
            kl_weight: float = 1.0,
            reconstruction_weight: float = 1.0,
            z_distance_cycle_weight: float = 2.0,
    ):

        x_true = tensors[REGISTRY_KEYS.X_KEY]

        # Reconstruction loss

        def reconst_loss_part(x_m, x, x_v):
            """
            Compute reconstruction loss
            :param x_m:
            :param x:
            :param x_v:
            :return:
            """
            return torch.nn.GaussianNLLLoss(reduction='none')(x_m, x, x_v).sum(dim=1)

        # Reconstruction loss
        reconst_loss_x = reconst_loss_part(x_m=generative_outputs['x_m'], x=x_true, x_v=generative_outputs['x_v'])
        reconst_loss = reconst_loss_x

        # Kl divergence on latent space
        kl_divergence_z = self.prior.kl(m_q=inference_outputs['z_m'], v_q=inference_outputs['z_v'],
                                        z=inference_outputs['z'])
        
        # Distance between modality latent space embeddings
        def z_dist(z_x_m, z_y_m, z_x_v, z_y_v):
            """
            If z_dist_metric is KL then KL(z_y|z_x) is computed
            If NLL then z_y_m is compared to N(z_x_m, z_x_v)
            :param z_x_m:
            :param z_y_m:
            :param z_x_v:
            :param z_y_v:
            :return:
            """

            if self.z_dist_metric == 'MSE':
                return torch.nn.MSELoss(reduction='none')(z_x_m, z_y_m).sum(dim=1)

            elif self.z_dist_metric == 'MSE_standard':
                # Standardise data (jointly both z-s) before MSE calculation
                z = torch.concat([z_x_m, z_y_m])
                means = z.mean(dim=0, keepdim=True)
                stds = z.std(dim=0, keepdim=True)

                def standardize(x):
                    return (x - means) / stds

                return torch.nn.MSELoss(reduction='none')(standardize(z_x_m), standardize(z_y_m)).sum(dim=1)

            elif self.z_dist_metric == 'KL':
                return kl_divergence(Normal(z_y_m, z_y_v.sqrt()), Normal(z_x_m, z_x_v.sqrt())).sum(dim=1)

            elif self.z_dist_metric == 'NLL':
                return reconst_loss_part(x_m=z_x_m, x=z_y_m, x_v=z_x_v)

            # elif self.z_dist_metric == 'cosine':
            #    return 1 - torch.nn.CosineSimilarity()(z_x, z_y)

            else:
                raise ValueError('z distance loss metric not recognised')

        # TODO one issue with z distance loss in cycle could be that the encoders and decoders learn to cheat
        #  and encode all cells from one species to one z region, even if they are from the cycle (e.g.
        #  z_x and z_x_y have more similar embeddings and z_y and z_y_x as well)
        z_distance_cyc = z_dist(z_x_m=inference_outputs['z_m'], z_y_m=inference_outputs['z_cyc_m'],
                                z_x_v=inference_outputs['z_v'], z_y_v=inference_outputs['z_cyc_v'])

        # Overall loss
        loss = (reconst_loss * reconstruction_weight +
                kl_divergence_z * kl_weight +
                z_distance_cyc * z_distance_cycle_weight)

        return LossRecorder(
            n_obs=loss.shape[0],
            loss=loss.mean(),
            loss_sum=loss.sum(),
            reconstruction_loss=reconst_loss.sum(),
            kl_local=kl_divergence_z.sum(),
            z_distance_cycle=z_distance_cyc.sum(), 
        )

    @torch.no_grad()
    def eval_metrics(self):
        if self.data_eval is None:
            return None
        else:
            return {metric_name + '_' + metric: val for metric, data in self.data_eval.items()
                    for metric_name, val in self._compute_eval_metrics(**data).items()}

    @auto_move_data
    def _compute_eval_metrics(self, inference_tensors, generative_cov, generative_kwargs, genes, target_x_m,
                              target_x_std):
        raise NotImplementedError

    def random_select_systems(self, tensors):
        """
        For every cell randomly selects a new system that is different from the original system

        Parameters:
        - tensors: torch.Tensor, tensor containing system information for each cell

        Returns:
        - new_tensor:  torch.Tensor, new tensor with the same shape as the input tensors, specifying the newly selected systems for each cell
        """
        #get available systems
        available_systems = 1 - tensors
        #Get indices for each cell
        row_indices, col_indices = torch.nonzero(available_systems, as_tuple=True)
        #Gather cols for a single row
        col_pairs = col_indices.view(-1, tensors.shape[1]-1)
        #Select system for every cell from available systems
        randomly_selected_indices = col_pairs.gather(1, torch.randint(0, tensors.shape[1]-1, size=(col_pairs.size(0), 1), 
                                                                      device=col_pairs.device, dtype=col_pairs.dtype))
        new_tensor = torch.zeros_like(available_systems)
        #generate system covariate tensor
        new_tensor.scatter_(1, randomly_selected_indices, 1)    

        return new_tensor

def _get_dict_if_none(param):
    param = {} if not isinstance(param, dict) else param
    return param
