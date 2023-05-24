import torch
from typing import Union
from collections import OrderedDict

from scvi.nn._base_components import reparameterize_gaussian

from torch.nn import Linear, Sequential, ReLU, BatchNorm1d, LayerNorm, Dropout, Parameter, Module



class EncoderDecoder(Module):

    def __init__(self,
                 n_input: int,
                 n_output: int,
                 n_cov:int,
                 n_hidden: int = 256,
                 n_layers: int = 3,
                 var_eps: float = 1e-4,
                 var_mode: str = 'feature',
                 sample: bool = False,
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
        self.sample = sample

        self.var_eps = var_eps

        self.decoder_y = Layers(n_in=n_input, n_cov=n_cov, n_out=n_hidden, n_hidden=n_hidden, n_layers=n_layers,
                                **kwargs)

        self.mean_encoder = Linear(n_hidden, n_output)
        self.var_encoder = VarEncoder(n_hidden, n_output, mode=var_mode, eps=var_eps)

    def forward(self, x, cov: Union[torch.Tensor, None] = None):
        y = self.decoder_y(x=x, cov=cov)
        # The nan_to_num should be temporary solution until figured out what is happening
        # TODO Here var eps was added so results would need to be re-run
        y_m = torch.nan_to_num(self.mean_encoder(y))
        y_v = self.var_encoder(y, x_m=y_m)

        outputs = dict(y_m=y_m, y_v=y_v)

        # Sample from latent distribution
        if self.sample:
            y = reparameterize_gaussian(y_m, y_v)
            outputs['y'] = y

        return outputs


class DecoderLin(Module):

    def __init__(self,
                 n_latent: int,
                 n_output: int,
                 var_eps: float = 1e-4,
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

        self.var_eps = var_eps

        # Var mode is feature so that var is not take from other genes. Another option would be to estimate based on
        # mean with 1d conv with kernel and stride of 1
        self.var_encoder = VarEncoder(n_hidden=n_latent, n_output=n_output, mode='feature', eps=var_eps)

    def forward(self, x, embed, mean, std):
        y = torch.matmul(x, embed.transpose(0, 1))
        y_m = (y * std.reshape(-1, std.shape[0])) + mean.reshape(-1, mean.shape[0])
        y_v = self.var_encoder(y)

        return y_m, y_v


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


class VarEncoder(Module):
    """
    Encode variance (strictly positive).
    """

    def __init__(self, n_hidden, n_output, mode: str, eps: float = 1e-4):
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
        # Force to be non nan - TODO come up with better way to do so
        if self.mode == 'sample_feature':
            v = self.encoder(x)
            v = torch.nan_to_num(self.activation(v)) + self.eps  # Ensure that var is strictly positive
        elif self.mode == 'feature':
            v = self.var_param.expand(x.shape[0], -1)  # Broadcast to input size
            v = torch.nan_to_num(self.activation(v)) + self.eps  # Ensure that var is strictly positive
        elif self.mode == 'linear':
            v = self.var_param_a1 * x_m.detach().clone() + self.var_param_a0
            # TODO come up with a better way to constrain this to positive while having lin  relationship
            v = torch.clamp(torch.nan_to_num(v), min=self.eps)
        return v

