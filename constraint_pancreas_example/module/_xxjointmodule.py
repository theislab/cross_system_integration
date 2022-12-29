from typing import Optional, Union, Tuple, Dict

import torch
from scvi import REGISTRY_KEYS
from scvi.module.base import BaseModuleClass, auto_move_data
from torch.distributions import Normal
from torch.distributions import kl_divergence

from constraint_pancreas_example.model._gene_maps import GeneMapInput
from constraint_pancreas_example.nn._base_components import EncoderDecoder
from constraint_pancreas_example.module._loss_recorder import LossRecorder

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
            n_latent: int = 15,
            n_hidden: int = 256,
            n_layers: int = 2,
            dropout_rate: float = 0.1,
            **kwargs
    ):
        super().__init__()

        self.gene_map = gene_map

        self.encoder = EncoderDecoder(
            n_input=n_input,
            n_output=n_latent,
            n_cov=n_cov,
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
            n_cov=n_cov,
            n_hidden=n_hidden,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            sample=True,
            var_mode='feature',
            **kwargs
        )

    def _get_inference_input(self, tensors, **kwargs):
        """Parse the dictionary to get appropriate args"""
        input_features = torch.ravel(torch.nonzero(self.gene_map.input_filter(device=self.device)))
        expr = tensors[REGISTRY_KEYS.X_KEY][:, input_features]
        cov = tensors['covariates']
        system = tensors['system']
        input_dict = dict(expr=expr, cov=cov, system=system)
        return input_dict

    def _get_inference_cycle_input(self, tensors, generative_outputs, **kwargs):
        """Parse the dictionary to get appropriate args"""
        input_features = torch.ravel(torch.nonzero(self.gene_map.input_filter(device=self.device)))
        expr = generative_outputs['y_m'][:, input_features]
        cov = self._mock_cov(tensors['covariates'])
        system = self._negate_zero_one(tensors['system'])
        input_dict = dict(expr=expr, cov=cov, system=system)
        return input_dict

    def _get_generative_input(self, tensors, inference_outputs, cov_replace: torch.Tensor = None, **kwargs):
        """
        Parse the dictionary to get appropriate args
        :param cov_replace: Replace cov from tensors with this covariate vector
        """
        z = inference_outputs["z"]
        if cov_replace is None:
            cov = {'x': tensors['covariates'], 'y': self._mock_cov(tensors['covariates'])}
        else:
            cov = {'x': cov_replace, 'y': cov_replace}
        system = {'x': tensors['system'], 'y': self._negate_zero_one(tensors['system'])}

        input_dict = dict(z=z, cov=cov, system=system)
        return input_dict

    def _get_generative_cycle_input(self, tensors, inference_cycle_outputs, **kwargs):
        """Parse the dictionary to get appropriate args"""
        z = inference_cycle_outputs["z"]
        cov = {'x': self._mock_cov(tensors['covariates']), 'y': tensors['covariates']}
        system = {'x': self._negate_zero_one(tensors['system']), 'y': tensors['system']}

        input_dict = dict(z=z, cov=cov, system=system)
        return input_dict

    @staticmethod
    def _negate_zero_one(x):
        # return torch.negative(x - 1)
        return torch.logical_not(x).float()

    @staticmethod
    def _merge_cov(covs):
        return torch.cat(covs, dim=1)

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

        z = self.encoder(x=expr, cov=self._merge_cov([cov, system]))
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

        def outputs(compute, name, res, x, cov):
            if compute:
                res_sub = self.decoder(x=x, cov=cov)
                res[name] = res_sub['y']
                res[name + '_m'] = res_sub['y_m']
                res[name + '_v'] = res_sub['y_v']

        res = {}
        outputs(compute=x_x, name='x', res=res, x=z, cov=self._merge_cov([cov['x'], system['x']]))
        outputs(compute=x_y, name='y', res=res, x=z, cov=self._merge_cov([cov['y'], system['y']]))
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
        generative_inputs = self._get_generative_input(
            tensors, inference_outputs, **get_generative_input_kwargs
        )
        generative_outputs = self.generative(**generative_inputs, x_x=True, x_y=True, **generative_kwargs)
        # Inference cycle
        inference_cycle_inputs = self._get_inference_cycle_input(
            tensors=tensors, generative_outputs=generative_outputs, **get_inference_input_kwargs)
        inference_cycle_outputs = self.inference(**inference_cycle_inputs, **inference_kwargs)
        # Generative cycle
        generative_cycle_inputs = self._get_generative_cycle_input(
            tensors=tensors, inference_cycle_outputs=inference_cycle_outputs, **get_generative_input_kwargs)
        generative_cycle_outputs = self.generative(**generative_cycle_inputs, x_x=False, x_y=True, **generative_kwargs)

        inference_outputs_merged = dict(**inference_outputs)
        inference_outputs_merged.update(
            **{k.replace('z_', 'z_cyc_'): v for k, v in inference_cycle_outputs.items()})
        generative_outputs_merged = dict(**generative_outputs)
        generative_outputs_merged.update(
            **{k.replace('x_', 'y_cyc_').replace('y_', 'x_cyc_'): v for k, v in generative_cycle_outputs.items()})

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

    @staticmethod
    def sample_expression(x_m, x_v):
        """
        Draw expression samples from mean and variance in generative outputs.
        :param x_m: Expression mean
        :param x_v: Expression variance
        :return: samples
        """
        return Normal(x_m, x_v.sqrt()).sample()

    def loss(
            self,
            tensors,
            inference_outputs,
            generative_outputs,
            kl_weight: float = 1.0,
            kl_cycle_weight: float = 1,
            reconstruction_weight: float = 1,
            reconstruction_cycle_weight: float = 1,
            # z_distance_paired_weight: float = 1,
            z_distance_cycle_weight: float = 1,
            # corr_cycle_weight: float = 1,

    ):
        x = tensors[REGISTRY_KEYS.X_KEY]

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
        reconst_loss_x = reconst_loss_part(x_m=generative_outputs['x_m'], x=x, x_v=generative_outputs['x_v'])
        reconst_loss = reconst_loss_x

        # Reconstruction loss in cycle
        reconst_loss_x_cyc = reconst_loss_part(x_m=generative_outputs['x_cyc_m'], x=x,
                                               x_v=generative_outputs['x_cyc_v'])
        reconst_loss_cyc = reconst_loss_x_cyc

        # Kl divergence on latent space
        def kl_loss_part(m, v):
            return kl_divergence(Normal(m, v.sqrt()), Normal(torch.zeros_like(m), torch.ones_like(v))).sum(dim=1)

        kl_divergence_z = kl_loss_part(m=inference_outputs['z_m'], v=inference_outputs['z_v'])

        # KL on the cycle z
        kl_divergence_z_cyc = kl_loss_part(m=inference_outputs['z_cyc_m'], v=inference_outputs['z_cyc_v'])

        # Distance between modality latent space embeddings
        def z_loss(z_x, z_y):
            return torch.nn.MSELoss(reduction='none')(z_x, z_y).sum(dim=1)

        # TODO one issue with z distance loss in cycle could be that the encoders and decoders learn to cheat
        #  and encode all cells from one species to one z region, even if they are from the cycle (e.g.
        #  z_x and z_x_y have more similar embeddings and z_y and z_y_x as well)
        z_distance_cyc = z_loss(z_x=inference_outputs['z_m'], z_y=inference_outputs['z_cyc_m'])

        # Correlation between both decodings within cycle - TODO

        # Overall loss
        loss = (reconst_loss * reconstruction_weight + reconst_loss_cyc * reconstruction_cycle_weight +
                kl_divergence_z * kl_weight + kl_divergence_z_cyc * kl_cycle_weight +
                z_distance_cyc * z_distance_cycle_weight)

        # TODO Currently this does not account for a different number of samples per batch due to masking
        return LossRecorder(
            n_obs=loss.shape[0], loss=loss.mean(), loss_sum=loss.sum(),
            reconstruction_loss=reconst_loss.sum(), kl_local=kl_divergence_z.sum(),
            reconstruction_loss_cycle=reconst_loss_cyc.sum(), kl_local_cycle=kl_divergence_z_cyc.sum(),
            z_distance_cycle=z_distance_cyc.sum(),
        )


def _get_dict_if_none(param):
    param = {} if not isinstance(param, dict) else param
    return param
