from typing import Optional, Union, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scvi import REGISTRY_KEYS
from scvi.distributions import ZeroInflatedNegativeBinomial
from scvi.module.base import BaseModuleClass, LossRecorder, auto_move_data
from scvi.nn import DecoderSCVI, Encoder, one_hot
from torch.distributions import Normal
from torch.distributions import kl_divergence

from cross_system_integration.model._gene_maps import GeneMapEmbedding
from cross_system_integration.nn._base_components import DecoderLin, EncoderDecoder
from cross_system_integration.nn._base_components import Layers

torch.backends.cudnn.benchmark = True


class XYLinModule(BaseModuleClass):
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
            n_cov_x: int,
            gene_map: GeneMapEmbedding,
            n_hidden: int = 256,
            n_layers: int = 3,
            dropout_rate: float = 0.1,
            **kwargs
    ):
        super().__init__()

        self.gene_map = gene_map

        # setup the parameters of your generative model, as well as your inference model
        # z encoder goes from the n_input-dimensional data to an n_latent-d
        # latent space representation
        self.encoder = EncoderDecoder(
            n_input=n_input,
            n_output=self.gene_map.n_embed,
            n_cov=n_cov_x,
            n_hidden=n_hidden,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            sample=True,
            **kwargs
        )
        self.decoder_y = DecoderLin(
            n_latent=self.gene_map.n_embed,
            n_output=n_output,
        )
        self.decoder_x = EncoderDecoder(
            n_input=self.gene_map.n_embed,
            n_output=n_input,
            n_cov=n_cov_x,
            n_hidden=n_hidden,
            n_layers=1,
            dropout_rate=dropout_rate,
            **kwargs
        )

    def _get_fwd_input(self, tensors):
        """Parse the dictionary to get appropriate args"""
        x = tensors[REGISTRY_KEYS.X_KEY][:, :self.gene_map.xsplit]
        constraints = self.gene_map.constraints(device=self.device)
        cov_x = tensors['covariates_x']

        input_dict = dict(x=x, cov_x=cov_x, embed=constraints['embed'], mean=constraints['mean'],
                          std=constraints['std'])
        return input_dict

    @auto_move_data
    def fwd_embed(self, x, cov_x):
        """
        High level inference method.

        Runs the inference (encoder) model.
        """
        # get embedding via the encoder networks
        outputs = self.encoder(x=x, cov=cov_x)
        return outputs['y_m'], outputs['y_v'], outputs['y']

    @auto_move_data
    def fwd_decode_y(self, z, embed, mean, std):
        """
        High level inference method.

        Runs the inference (encoder) model.
        """
        # TODO add y covariates - one layer z+cov_y to get to final z and then lin decode
        # get variational parameters via the encoder networks
        y_m, y_v = self.decoder_y(x=z, embed=embed, mean=mean, std=std)

        return y_m, y_v

    @auto_move_data
    def fwd_decode_x(self, z, cov_x):
        """
        High level inference method.

        Runs the inference (encoder) model.
        """
        # get variational parameters via the encoder networks
        outputs = self.decoder_x(x=z, cov=cov_x)

        return outputs['y_m'], outputs['y_v']

    @auto_move_data
    def fwd(self, x, cov_x, embed, mean, std):
        """
        High level inference method.

        Runs the inference (encoder) model.
        """
        # get variational parameters via the encoder networks
        z_m, z_v, z = self.fwd_embed(x=x, cov_x=cov_x)
        y_m, y_v = self.fwd_decode_y(z=z, embed=embed, mean=mean, std=std)
        x_m, x_v = self.fwd_decode_x(z=z, cov_x=cov_x)
        outputs = dict(y_m=y_m, y_v=y_v, x_m=x_m, x_v=x_v, z_m=z_m, z_v=z_v, z=z)
        return outputs

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
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, LossRecorder],
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

        fwd_inputs = self._get_fwd_input(tensors)
        fwd_outputs = self.fwd(**fwd_inputs)
        if compute_loss:
            losses = self.loss(
                tensors, fwd_outputs
            )
            # TODO remove the zeros mock here
            return None, fwd_outputs, losses
        else:
            return None, fwd_outputs

    def sample_expression(self, x_m, x_v):
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
            fwd_outputs,
    ):
        x = tensors[REGISTRY_KEYS.X_KEY][:, :self.gene_map.xsplit]
        y = tensors[REGISTRY_KEYS.X_KEY][:, self.gene_map.xsplit:]
        train_y = tensors['train_y']
        y_m = fwd_outputs["y_m"]
        y_v = fwd_outputs["y_v"]
        x_m = fwd_outputs["x_m"]
        x_v = fwd_outputs["x_v"]
        z_m = fwd_outputs["z_m"]
        z_v = fwd_outputs["z_v"]

        # Reconstruction loss
        # Reconstruction y loss is set to 0 for some examples that do not have meaningful expression
        reconst_loss_y = torch.nn.GaussianNLLLoss(reduction='none')(y_m, y, y_v).sum(dim=1) * torch.ravel(train_y)
        reconst_loss_x = torch.nn.GaussianNLLLoss(reduction='none')(x_m, x, x_v).sum(dim=1)

        # Kl divergence on latent space
        kl_divergence_z = kl_divergence(Normal(z_m, z_v.sqrt()),
                                        Normal(torch.zeros_like(z_m), torch.ones_like(z_v))
                                        ).sum(dim=1)

        loss = torch.mean(reconst_loss_y + reconst_loss_x + kl_divergence_z)

        # TODO maybe later adapt scvi/train/_trainingplans.py to keep both recon losses
        return LossRecorder(loss=loss,
                            reconstruction_loss=reconst_loss_y,
                            kl_local=kl_divergence_z,
                            reconstruction_loss_x=reconst_loss_x.sum())
