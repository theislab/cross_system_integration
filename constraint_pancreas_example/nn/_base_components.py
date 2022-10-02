import torch
from torch.nn import Linear, Module
from typing import List

from cross_species_prediction.nn._base_components import Layers, VarEncoder


class EncoderDecoder(Module):

    def __init__(self,
                 n_input: int,
                 n_output: int,
                 n_hidden: int = 256,
                 n_layers: int = 3,
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

        self.var_eps = var_eps

        self.decoder_y = Layers(n_in=n_input, n_cov=0, n_out=n_hidden, n_hidden=n_hidden, n_layers=n_layers, **kwargs)

        self.mean_encoder = Linear(n_hidden, n_output)
        self.var_encoder = VarEncoder(n_hidden, n_output, mode=var_mode, eps=var_eps)

    def forward(self, x):
        y = self.decoder_y(x=x)
        y_m = self.mean_encoder(y)
        y_v = self.var_encoder(y)

        return y_m, y_v


class EncoderDecoderLin(Module):

    def __init__(self,
                 n_input: int,
                 n_latent: int,
                 n_output: int,
                 n_hidden: int = 256,
                 n_layers: int = 3,
                 var_eps: float = 1e-4,
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

        self.var_eps = var_eps

        self.decoder_y = Layers(n_in=n_input, n_cov=0, n_out=n_latent, n_hidden=n_hidden, n_layers=n_layers, **kwargs)
        # Var mode is feature so that var is not take from other genes. Another option would be to estimate based on
        # mean with 1d conv with kernel and stride of 1
        self.var_encoder = VarEncoder(n_hidden=n_latent, n_output=n_output, mode='feature', eps=var_eps)

    def forward(self, x, embed, mean, std):
        y = self.decoder_y(x=x)
        y = torch.matmul(y, embed.transpose(0, 1))
        y_m = (y * std.reshape(-1, std.shape[0])) + mean.reshape(-1, mean.shape[0])
        y_v = self.var_encoder(y)

        return y_m, y_v
