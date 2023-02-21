import torch
from torch.nn import Linear, Module
from typing import List, Union

from scvi.nn._base_components import reparameterize_gaussian

from cross_species_prediction.nn._base_components import Layers, VarEncoder


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
        y_v = torch.nan_to_num(self.var_encoder(y, x_m=y_m))+self.var_eps

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
