from typing import Optional, Union, Tuple

import torch
from scvi import REGISTRY_KEYS
from scvi.module.base import BaseModuleClass, LossRecorder, auto_move_data
from torch.distributions import Normal
from torch.distributions import kl_divergence

from constraint_pancreas_example.model._gene_maps import GeneMapInput
from constraint_pancreas_example.nn._base_components import EncoderDecoder

torch.backends.cudnn.benchmark = True


class XYBiModule(BaseModuleClass):
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
            n_input_x: int,
            n_input_y: int,
            n_output_x: int,
            n_output_y: int,
            gene_map: GeneMapInput,
            n_cov_x: int,
            n_cov_y: int,
            n_latent: int = 15,
            n_hidden: int = 256,
            n_layers: int = 2,
            dropout_rate: float = 0.1,
            **kwargs
    ):
        super().__init__()

        self.gene_map = gene_map

        # setup the parameters of your generative model, as well as your inference model
        # z encoder goes from the n_input-dimensional data to an n_latent-d
        # latent space representation
        self.encoder_x = EncoderDecoder(
            n_input=n_input_x,
            n_output=n_latent,
            n_cov=n_cov_x,
            n_hidden=n_hidden,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            sample=True,
            var_mode='sample_feature',
            **kwargs
        )
        self.encoder_y = EncoderDecoder(
            n_input=n_input_y,
            n_output=n_latent,
            n_cov=n_cov_y,
            n_hidden=n_hidden,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            sample=True,
            var_mode='sample_feature',
            **kwargs
        )
        self.decoder_x = EncoderDecoder(
            n_input=n_latent,
            n_output=n_output_x,
            n_cov=n_cov_x,
            n_hidden=n_hidden,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            sample=True,
            var_mode='feature',
            **kwargs
        )
        self.decoder_y = EncoderDecoder(
            n_input=n_latent,
            n_output=n_output_y,
            n_cov=n_cov_y,
            n_hidden=n_hidden,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            sample=True,
            var_mode='feature',
            **kwargs
        )

    def _get_inference_input(self, tensors, **kwargs):
        """Parse the dictionary to get appropriate args"""
        input_x = torch.ravel(torch.nonzero(self.gene_map.input_filter(device=self.device)[:self.gene_map.xsplit]))
        input_y = torch.ravel(torch.nonzero(self.gene_map.input_filter(device=self.device)[self.gene_map.xsplit:]))
        expr = {'x': tensors[REGISTRY_KEYS.X_KEY][:, :self.gene_map.xsplit][:, input_x],
                'y': tensors[REGISTRY_KEYS.X_KEY][:, self.gene_map.xsplit:][:, input_y]}
        cov = {'x': tensors['covariates_x'], 'y': tensors['covariates_y']}
        train = {'x': tensors['train_x'], 'y': tensors['train_y']}

        input_dict = dict(expr=expr, cov=cov, train=train)
        return input_dict

    def _get_generative_input(self, tensors, inference_outputs, **kwargs):
        """Parse the dictionary to get appropriate args"""
        z = {'x': inference_outputs["z_x"], 'y': inference_outputs["z_y"]}
        cov = {'x': tensors['covariates_x'], 'y': tensors['covariates_y']}
        train = {'x': tensors['train_x'], 'y': tensors['train_y']}

        input_dict = dict(z=z, cov=cov, train=train)
        return input_dict

    @auto_move_data
    def inference(self, expr, cov, train, mask_subset=True, pad_output=False):
        def fwd(network, x, c, mask):
            if mask_subset:
                mask_idx = torch.ravel(torch.nonzero(torch.ravel(mask)))
                x = x[mask_idx, :]
                c = c[mask_idx, :]
            res = network(x=x, cov=c)
            if pad_output and mask_subset:
                for k, dat in res.items():
                    out = torch.zeros((mask.shape[0], dat.shape[1]), device=self.device)
                    out[mask_idx, :] = dat
                    res[k] = out
            return res

        z_x = fwd(network=self.encoder_x, x=expr['x'], c=cov['x'], mask=train['x'])
        z_y = fwd(network=self.encoder_y, x=expr['y'], c=cov['y'], mask=train['y'])
        return dict(z_x=z_x['y'], z_x_m=z_x['y_m'], z_x_v=z_x['y_v'],
                    z_y=z_y['y'], z_y_m=z_y['y_m'], z_y_v=z_y['y_v'])

    @auto_move_data
    def generative(self, z, cov, train, z_masked=True, mask_subset=True, pad_output=False,
                   x_x=True, x_y=True, y_y=True, y_x=True):
        def fwd(network, x, c, mask):
            if mask_subset:
                mask_idx = torch.ravel(torch.nonzero(torch.ravel(mask)))
                if not z_masked:
                    x = x[mask_idx, :]
                c = c[mask_idx, :]
            res = network(x=x, cov=c)
            if pad_output and mask_subset:
                for k, dat in res.items():
                    out = torch.zeros((mask.shape[0], dat.shape[1]), device=self.device)
                    out[mask_idx, :] = dat
                    res[k] = out
            return res

        def outputs(compute, name, res, network, x, c, mask):
            if compute:
                res_sub = fwd(network=network, x=x, c=c, mask=mask)
                res[name] = res_sub['y']
                res[name + '_m'] = res_sub['y_m']
                res[name + '_v'] = res_sub['y_v']

        res = {}
        outputs(compute=x_x, name='x_x', res=res, network=self.decoder_x, x=z['x'], c=cov['x'], mask=train['x'])
        outputs(compute=x_y, name='x_y', res=res, network=self.decoder_y, x=z['x'], c=cov['y'], mask=train['x'])
        outputs(compute=y_y, name='y_y', res=res, network=self.decoder_y, x=z['y'], c=cov['y'], mask=train['y'])
        outputs(compute=y_x, name='y_x', res=res, network=self.decoder_x, x=z['y'], c=cov['x'], mask=train['y'])
        return res

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
            inference_outputs,
            generative_outputs,
            kl_weight: float = 1.0,
    ):
        x, y = (tensors[REGISTRY_KEYS.X_KEY][:, :self.gene_map.xsplit],
                tensors[REGISTRY_KEYS.X_KEY][:, self.gene_map.xsplit:])
        train_x, train_y = (tensors['train_x'], tensors['train_y'])

        # Reconstruction loss

        def reconst_loss_part(x_m, x, x_v, mask_x, mask_loss):
            """
            Compute reconstruction loss, do not compute for samples without matched x, cast to shape of original samples
            :param x_m:
            :param x:
            :param x_v:
            :param mask_x: Masking that was used for computing x_m and x_v subset
            :param mask_loss: Masking of loss since no expression is available
            :return:
            """
            out = torch.zeros(mask_x.shape[0], device = self.device)
            mask = torch.ravel(torch.nonzero(torch.ravel(mask_x * mask_loss)))
            mask_x = torch.ravel(torch.nonzero(torch.ravel(mask_x)))
            mask_sub = torch.ravel(torch.nonzero(torch.ravel(mask_loss[mask_x, :])))
            out[mask] = torch.nn.GaussianNLLLoss(reduction='none')(
                x_m[mask_sub, :], x[mask, :], x_v[mask_sub, :]).sum(dim=1)
            return out

        reconst_loss_x_x = reconst_loss_part(
            x_m=generative_outputs['x_x_m'], x=x, x_v=generative_outputs['x_x_v'], mask_x=train_x, mask_loss=train_x)
        reconst_loss_x_y = reconst_loss_part(
            x_m=generative_outputs['x_y_m'], x=y, x_v=generative_outputs['x_y_v'], mask_x=train_x, mask_loss=train_y)
        reconst_loss_y_y = reconst_loss_part(
            x_m=generative_outputs['y_y_m'], x=y, x_v=generative_outputs['y_y_v'], mask_x=train_y, mask_loss=train_y)
        reconst_loss_y_x = reconst_loss_part(
            x_m=generative_outputs['y_x_m'], x=x, x_v=generative_outputs['y_x_v'], mask_x=train_y, mask_loss=train_x)

        reconst_loss = reconst_loss_x_x + reconst_loss_x_y + reconst_loss_y_y + reconst_loss_y_x

        # Kl divergence on latent space
        def kl_loss_part(m, v, mask):
            out = torch.zeros(mask.shape[0], device = self.device)
            mask = torch.ravel(torch.nonzero(torch.ravel(mask)))
            out[mask] = kl_divergence(Normal(m, v.sqrt()), Normal(torch.zeros_like(m), torch.ones_like(v))).sum(dim=1)
            return out

        kl_divergence_z_x = kl_loss_part(m=inference_outputs['z_x_m'], v=inference_outputs['z_x_v'], mask=train_x)
        kl_divergence_z_y = kl_loss_part(m=inference_outputs['z_y_m'], v=inference_outputs['z_y_v'], mask=train_y)
        kl_divergence_z = kl_divergence_z_x + kl_divergence_z_y

        loss = torch.mean(reconst_loss + kl_divergence_z)

        # TODO maybe later adapt scvi/train/_trainingplans.py to keep both recon losses
        return LossRecorder(loss=loss, reconstruction_loss=reconst_loss, kl_local=kl_divergence_z,
                            reconst_loss_x_x=reconst_loss_x_x.sum(), reconst_loss_x_y=reconst_loss_x_y.sum(),
                            reconst_loss_y_y=reconst_loss_y_y.sum(), reconst_loss_y_x=reconst_loss_y_x.sum(),
                            kl_divergence_z_x=kl_divergence_z_x.sum(), kl_divergence_z_y=kl_divergence_z_y.sum(),
                            )
