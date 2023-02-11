from typing import Union, List

import torch
from collections import OrderedDict

from torch.nn import Linear, Sequential, ReLU, BatchNorm1d, LayerNorm, Dropout, Parameter, Module

from scvi.nn._base_components import reparameterize_gaussian


class MultiEncoder(Module):

    def __init__(self,
                 n_input: int,
                 n_cov: int,
                 n_output: int,
                 n_encoders: int,
                 n_hidden: int = 128,
                 var_eps: float = 1e-4,
                 **kwargs
                 ):
        super().__init__()

        self.encoders = [Layers(n_in=n_input, n_cov=n_cov, n_out=n_hidden, n_hidden=n_hidden, **kwargs)
                         for i in range(n_encoders)]

        self.mean_encoder = Linear(n_hidden, n_output)
        self.var_encoder = VarEncoder(n_hidden, n_output, mode='sample_feature', eps=var_eps)

    def forward(self,
                x_species: torch.Tensor,
                cov_species: torch.Tensor,
                species_ratio: torch.Tensor,
                ):
        # Latent distribution
        q = [e(x=x_species[:, idx], cov=cov_species[:, idx]) *  # Per-species encoding
             species_ratio[:, idx].reshape(-1, 1)  # Contribution of each component
             for idx, e in enumerate(self.encoders)]
        q = torch.stack(q, dim=0).sum(dim=0)  # Combine per-species encodings
        q_m = self.mean_encoder(q)
        q_v = self.var_encoder(q)
        # Sample from latent distribution
        z = reparameterize_gaussian(q_m, q_v)
        return dict(qz_m=q_m, qz_v=q_v, z=z)


class VarEncoder(Module):
    """
    Encode variance (strictly positive).
    """

    def __init__(self, n_hidden, n_output, mode: str, eps: float = 1e-6):
        # NOTE: Changed eps
        """

        :param n_hidden:
        :param n_output:
        :param mode: How to compute var
        'sample_feature' - learn per sample and feature
        'feature' - learn per feature, constant across samples
        :param eps:
        """
        super().__init__()

        self.eps = eps
        self.mode = mode
        if self.mode == 'sample_feature':
            self.encoder = Linear(n_hidden, n_output)
        elif self.mode == 'feature':
            self.var_param = Parameter(torch.zeros(1, n_output))
        elif self.mode == 'linear':
            self.var_param_a1 = Parameter(torch.tensor([1.0]))
            self.var_param_a0 = Parameter(torch.tensor([self.eps]))
        else:
            raise ValueError('Mode not recognised.')
        self.activation = torch.exp

    def forward(self, x: torch.Tensor, x_m: torch.Tensor):
        """

        :param x: Used to encode var if mode is sample_feature
        :param x_m: Used to predict var instead of x if mode is linear
        :return:
        """
        # var on ln scale
        if self.mode == 'sample_feature':
            v = self.encoder(x)
            v = self.activation(v) + self.eps  # Ensure that var is strictly positive
        elif self.mode == 'feature':
            v = self.var_param.expand(x.shape[0], -1)  # Broadcast to input size
            v = self.activation(v) + self.eps  # Ensure that var is strictly positive
        elif self.mode == 'linear':
            v = self.var_param_a1 * x_m + self.var_param_a0
            # TODO come up with a better way to constrain this to positive while having lin  relationship
            v = torch.clamp(v, min=self.eps)
        return v


class Layers(Module):
    """
    A helper class to build fully-connected layers for a neural network.
    Adapted from scVI FCLayers to use non-categorical covariates

    Parameters
    ----------
    n_in
        The dimensionality of the main input
    n_out
        The dimensionality of the output
    n_cov
        Number of covariates that can be injected into each layer. If there are no cov this should be set to None -
        in this case cov will not be used.
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    dropout_rate
        Dropout rate to apply to each of the hidden layers
    use_batch_norm
        Whether to have `BatchNorm` layers or not
    use_layer_norm
        Whether to have `LayerNorm` layers or not
    use_activation
        Whether to have layer activation or not
    bias
        Whether to learn bias in linear layers or not
    inject_covariates
        Whether to inject covariates in each layer, or just the first.
    activation_fn
        Which activation function to use
    """

    def __init__(
            self,
            n_in: int,
            n_out: int,
            n_cov: Union[int, None] = None,
            n_layers: int = 1,
            n_hidden: int = 128,
            dropout_rate: float = 0.1,
            use_batch_norm: bool = True,
            use_layer_norm: bool = False,
            use_activation: bool = True,
            bias: bool = True,
            inject_covariates: bool = True,
            activation_fn: Module = ReLU,
    ):
        super().__init__()

        self.inject_covariates = inject_covariates
        self.n_cov = n_cov if n_cov is not None else 0

        layers_dim = [n_in] + (n_layers - 1) * [n_hidden] + [n_out]

        self.fc_layers = Sequential(
            OrderedDict(
                [
                    (
                        "Layer {}".format(i),
                        Sequential(
                            Linear(
                                n_in + self.n_cov * self.inject_into_layer(i),
                                n_out,
                                bias=bias,
                            ),
                            # non-default params come from defaults in original Tensorflow implementation
                            BatchNorm1d(n_out, momentum=0.01, eps=0.001)
                            if use_batch_norm
                            else None,
                            LayerNorm(n_out, elementwise_affine=False)
                            if use_layer_norm
                            else None,
                            activation_fn() if use_activation else None,
                            Dropout(p=dropout_rate) if dropout_rate > 0 else None,
                        ),
                    )
                    for i, (n_in, n_out) in enumerate(zip(layers_dim[:-1], layers_dim[1:]))
                ]
            )
        )

    def inject_into_layer(self, layer_num) -> bool:
        """Helper to determine if covariates should be injected."""
        user_cond = layer_num == 0 or (layer_num > 0 and self.inject_covariates)
        return user_cond

    def set_online_update_hooks(self, hook_first_layer=True):

        self.hooks = []

        def _hook_fn_weight(grad):
            new_grad = torch.zeros_like(grad)
            if self.n_cov > 0:
                new_grad[:, -self.n_cov:] = grad[:, -self.n_cov:]
            return new_grad

        def _hook_fn_zero_out(grad):
            return grad * 0

        for i, layers in enumerate(self.fc_layers):
            for layer in layers:
                if i == 0 and not hook_first_layer:
                    continue
                if isinstance(layer, Linear):
                    if self.inject_into_layer(i):
                        w = layer.weight.register_hook(_hook_fn_weight)
                    else:
                        w = layer.weight.register_hook(_hook_fn_zero_out)
                    self.hooks.append(w)
                    b = layer.bias.register_hook(_hook_fn_zero_out)
                    self.hooks.append(b)

    def forward(self, x: torch.Tensor, cov: Union[torch.Tensor, None] = None):
        """
        Forward computation on ``x``.

        Parameters
        ----------
        x
            tensor of values with shape ``(n_in,)``
        cov
            tensor of covariate values with shape ``(n_cov,)`` or None

        Returns
        -------
        py:class:`torch.Tensor`
            tensor of shape ``(n_out,)``

        """
        for i, layers in enumerate(self.fc_layers):
            for layer in layers:
                if layer is not None:
                    if isinstance(layer, BatchNorm1d):
                        if x.dim() == 3:
                            x = torch.cat(
                                [(layer(slice_x)).unsqueeze(0) for slice_x in x], dim=0
                            )
                        else:
                            x = layer(x)
                    else:
                        # Injection of covariates
                        if self.n_cov > 0 and isinstance(layer, Linear) and self.inject_into_layer(i):
                            x = torch.cat((x, cov), dim=-1)
                        x = layer(x)
        return x


class MultiDecoder(Module):

    def __init__(self,
                 n_input: int,
                 n_cov_species: int,
                 n_cov_shared: int,
                 n_output_species: List[int],
                 n_output_shared: int,
                 shared_first: bool,
                 n_hidden: int = 128,
                 var_eps: float = 1e-4,
                 var_mode: str = 'feature',
                 **kwargs
                 ):
        """

        :param n_input:
        :param n_cov_species:
        :param n_cov_shared:
        :param n_output_species:
        :param n_output_shared:
        :param shared_first:
        :param n_hidden:
        :param var_eps:
        :param var_mode: Single var per feature - "feature" or predict var from embedding - "sample_feature".
        :param kwargs:
        """
        super().__init__()

        self.shared_first = shared_first
        self.var_eps = var_eps
        n_output = sum(n_output_species + [n_output_shared])

        self.decoder_z = Layers(n_in=n_input, n_cov=0, n_out=n_hidden, n_hidden=n_hidden, **kwargs)

        if 'n_layers' in kwargs:
            del kwargs['n_layers']
        self.decoders_x_species = [
            Layers(n_in=n_hidden, n_cov=n_cov_species, n_out=n_out, n_hidden=n_hidden, n_layers=1, **kwargs)
            for n_out in n_output_species]
        self.decoder_x_shared = Layers(n_in=n_hidden, n_cov=n_cov_shared, n_out=n_output_shared, n_hidden=n_hidden,
                                       n_layers=1, **kwargs)

        self.mean_encoder = Linear(n_output, n_output)
        self.var_encoder = VarEncoder(n_output, n_output, mode=var_mode, eps=var_eps)

    def forward(self, x, cov_species, cov_shared):
        z = self.decoder_z(x=x)
        x_species = [decoder(x=z, cov=cov_species[:, idx]) for idx, decoder in enumerate(self.decoders_x_species)]
        x_shared = self.decoder_x_shared(x=z, cov=cov_shared)
        if self.shared_first:
            x = [x_shared] + x_species
        else:
            x = x_species + [x_shared]
        x = torch.cat(x, dim=-1)

        x_m = self.mean_encoder(x)
        x_v = self.var_encoder(x)

        return dict(x_m=x_m, x_v=x_v)
