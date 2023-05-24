from typing import Optional, Union, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scvi import REGISTRY_KEYS
from scvi.distributions import ZeroInflatedNegativeBinomial
from scvi.module.base import BaseModuleClass, LossRecorder, auto_move_data
from scvi.nn import DecoderSCVI, Encoder, one_hot
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl

from constraint_pancreas_example.model._gene_maps import GeneMapConstraint
from constraint_pancreas_example.nn._base_components import EncoderDecoder

torch.backends.cudnn.benchmark = True


class XYModule(BaseModuleClass):
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
            gene_map: GeneMapConstraint,
            n_hidden: int = 256,
            n_layers: int = 3,
            dropout_rate: float = 0.1,
            constraint_weight: float = 1,
            constraint_regression_loss: bool = False
    ):
        super().__init__()

        self.gene_map = gene_map
        self.constraint_weight = constraint_weight
        self.constraint_regression_loss = constraint_regression_loss

        # setup the parameters of your generative model, as well as your inference model
        # z encoder goes from the n_input-dimensional data to an n_latent-d
        # latent space representation
        self.fwd_nn = EncoderDecoder(
            n_input=n_input,
            n_output=n_output,
            n_cov=0,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
        )

    def _get_fwd_input(self, tensors):
        """Parse the dictionary to get appropriate args"""
        x = tensors[REGISTRY_KEYS.X_KEY][:, :self.gene_map.xsplit]

        input_dict = dict(x=x)
        return input_dict

    @auto_move_data
    def fwd(self, x):
        """
        High level inference method.

        Runs the inference (encoder) model.
        """
        # get variational parameters via the encoder networks
        outputs = self.fwd_nn(x)

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
        y = tensors[REGISTRY_KEYS.X_KEY][:, self.gene_map.xsplit:]
        y_m = fwd_outputs["y_m"]
        y_v = fwd_outputs["y_v"]
        constraints = self.gene_map.constraints(device=self.device)

        #  TODO get in detail trhough this, sums etc

        # Reconstruction loss
        reconst_loss = torch.nn.GaussianNLLLoss(reduction='none')(y_m, y, y_v).sum(dim=1)

        # Constraint loss
        if self.constraint_regression_loss:
            loss_constraint = torch.abs(y_m[:, constraints['gx']] * constraints['coef'] + constraints['intercept'] -
                                        y_m[:, constraints['gx']]).sum(dim=1)
        else:
            loss_constraint = 0

        loss = (reconst_loss + self.constraint_weight * loss_constraint).sum()

        return LossRecorder(loss=loss, reconstruction_loss=reconst_loss)
