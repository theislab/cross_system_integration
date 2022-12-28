from typing import Optional, Union, Tuple

import torch
from scvi import REGISTRY_KEYS
from scvi.module.base import BaseModuleClass, auto_move_data
from torch.distributions import Normal
from torch.distributions import kl_divergence

from constraint_pancreas_example.model._gene_maps import GeneMapXYBimodel
from constraint_pancreas_example.nn._base_components import EncoderDecoder
from constraint_pancreas_example.module._loss_recorder import LossRecorder

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
            gene_map: GeneMapXYBimodel,
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

    def _get_inference_cycle_input(self, tensors, generative_outputs, **kwargs):
        """Parse the dictionary to get appropriate args"""
        input_x = torch.ravel(torch.nonzero(self.gene_map.input_filter(device=self.device)[:self.gene_map.xsplit]))
        input_y = torch.ravel(torch.nonzero(self.gene_map.input_filter(device=self.device)[self.gene_map.xsplit:]))
        expr = {'x': generative_outputs['y_x_m'][:, input_x],
                'y': generative_outputs['x_y_m'][:, input_y]}
        # Cov must stay the same as they are modality specific and also should not affect final result after the
        # whole cycle, as long as the same cov are used to go z->y and y->z
        cov = {'x': tensors['covariates_x'], 'y': tensors['covariates_y']}
        # Here which samples are to be trained depends on the input samples from the 1st half of the cycle,
        # hence samples to be trained for x actually depend on samples that were trained for y in the first pass
        train = {'x': tensors['train_y'], 'y': tensors['train_x']}

        input_dict = dict(expr=expr, cov=cov, train=train)
        return input_dict

    def _get_generative_input(self, tensors, inference_outputs, **kwargs):
        """Parse the dictionary to get appropriate args"""
        z = {'x': inference_outputs["z_x"], 'y': inference_outputs["z_y"]}
        cov = {'x': tensors['covariates_x'], 'y': tensors['covariates_y']}
        train = {'x': tensors['train_x'], 'y': tensors['train_y']}

        input_dict = dict(z=z, cov=cov, train=train)
        return input_dict

    def _get_generative_cycle_input(self, tensors, inference_outputs, **kwargs):
        """Parse the dictionary to get appropriate args"""
        z = {'x': inference_outputs["z_x"], 'y': inference_outputs["z_y"]}
        # Cov must stay the same as they are modality specific and also should not affect final result after the
        # whole cycle, as long as the same cov are used to go z->y and y->z
        cov = {'x': tensors['covariates_x'], 'y': tensors['covariates_y']}
        # Here which samples are to be trained depends on the input samples from the 1st half of the cycle,
        # hence samples to be trained for x actually depend on samples that were trained for y in the first pass
        train = {'x': tensors['train_y'], 'y': tensors['train_x']}

        input_dict = dict(z=z, cov=cov, train=train)
        return input_dict

    @auto_move_data
    def inference(self, expr, cov, train, expr_masked=False, mask_subset=True, pad_output=False):
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

        def fwd(network, x, c, mask):
            mask_idx = torch.ravel(torch.nonzero(torch.ravel(mask)))
            if mask_subset:
                if not expr_masked:
                    x = x[mask_idx, :]
                c = c[mask_idx, :]
            res = network(x=x, cov=c)
            if pad_output:
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

        def fwd(network, x, c, mask):
            mask_idx = torch.ravel(torch.nonzero(torch.ravel(mask)))
            if mask_subset:
                if not z_masked:
                    x = x[mask_idx, :]
                c = c[mask_idx, :]
            res = network(x=x, cov=c)
            if pad_output:
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
        inference_outputs = self.inference(
            **inference_inputs, expr_masked=False, mask_subset=True, pad_output=False, **inference_kwargs)
        # Generative
        generative_inputs = self._get_generative_input(
            tensors, inference_outputs, **get_generative_input_kwargs
        )
        generative_outputs = self.generative(**generative_inputs, z_masked=True, mask_subset=True, pad_output=False,
                                             x_x=True, x_y=True, y_y=True, y_x=True, **generative_kwargs)
        # Inference cycle
        inference_cycle_inputs = self._get_inference_cycle_input(
            tensors=tensors, generative_outputs=generative_outputs, **get_inference_input_kwargs)
        inference_cycle_outputs = self.inference(
            **inference_cycle_inputs, expr_masked=True, mask_subset=True, pad_output=False, **inference_kwargs)
        # Generative cycle
        generative_cycle_inputs = self._get_generative_cycle_input(
            tensors=tensors, inference_outputs=inference_cycle_outputs, **get_generative_input_kwargs)
        generative_cycle_outputs = self.generative(
            **generative_cycle_inputs, z_masked=True, mask_subset=True, pad_output=False,
            x_x=False, x_y=True, y_y=False, y_x=True, **generative_kwargs)

        # TODO could make special TrainingPlan that expects more outputs and replace it in the
        #  model UnsupervisedTrainingMixin
        inference_outputs_merged = dict(**inference_outputs)
        inference_outputs_merged.update(
            **{k.replace('z_x', 'z_y_x') if k.startswith('z_x') else k.replace('z_y', 'z_x_y'): v
               for k, v in inference_cycle_outputs.items()})
        generative_outputs_merged = dict(**generative_outputs)
        generative_outputs_merged.update(
            # TODO could also do re.sub to just replace starting occurrences
            #  instead of checking that it starts with one of these
            **{k.replace('x_y', 'y_x_y') if k.startswith('x_y') else k.replace('y_x', 'x_y_x'): v
               for k, v in generative_cycle_outputs.items() if k.startswith('x_y') or k.startswith('y_x')})

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
            kl_cycle_weight: float = 1,
            reconstruction_weight: float = 1,
            reconstruction_cycle_weight: float = 1,
            z_distance_paired_weight: float = 1,
            z_distance_cycle_weight: float = 1,
            corr_cycle_weight: float = 1,

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
            out = torch.zeros(mask_x.shape[0], device=self.device)
            mask = torch.ravel(torch.nonzero(torch.ravel(mask_x * mask_loss)))
            mask_x = torch.ravel(torch.nonzero(torch.ravel(mask_x)))
            mask_sub = torch.ravel(torch.nonzero(torch.ravel(mask_loss[mask_x, :])))
            out[mask] = torch.nn.GaussianNLLLoss(reduction='none')(
                x_m[mask_sub, :], x[mask, :], x_v[mask_sub, :]).sum(dim=1)
            return out

        # Reconstruction loss
        reconst_loss_x_x = reconst_loss_part(
            x_m=generative_outputs['x_x_m'], x=x, x_v=generative_outputs['x_x_v'], mask_x=train_x, mask_loss=train_x)
        reconst_loss_x_y = reconst_loss_part(
            x_m=generative_outputs['x_y_m'], x=y, x_v=generative_outputs['x_y_v'], mask_x=train_x, mask_loss=train_y)
        reconst_loss_y_y = reconst_loss_part(
            x_m=generative_outputs['y_y_m'], x=y, x_v=generative_outputs['y_y_v'], mask_x=train_y, mask_loss=train_y)
        reconst_loss_y_x = reconst_loss_part(
            x_m=generative_outputs['y_x_m'], x=x, x_v=generative_outputs['y_x_v'], mask_x=train_y, mask_loss=train_x)

        reconst_loss = reconst_loss_x_x + reconst_loss_x_y + reconst_loss_y_y + reconst_loss_y_x

        # Reconstruction loss in cycle
        reconst_loss_x_y_x = reconst_loss_part(
            x_m=generative_outputs['x_y_x_m'], x=x, x_v=generative_outputs['x_y_x_v'],
            mask_x=train_x, mask_loss=train_x)
        reconst_loss_y_x_y = reconst_loss_part(
            x_m=generative_outputs['y_x_y_m'], x=y, x_v=generative_outputs['y_x_y_v'],
            mask_x=train_y, mask_loss=train_y)
        reconst_loss_cycle = reconst_loss_x_y_x + reconst_loss_y_x_y

        # Kl divergence on latent space
        def kl_loss_part(m, v, mask):
            out = torch.zeros(mask.shape[0], device=self.device)
            mask = torch.ravel(torch.nonzero(torch.ravel(mask)))
            out[mask] = kl_divergence(Normal(m, v.sqrt()), Normal(torch.zeros_like(m), torch.ones_like(v))).sum(dim=1)
            return out

        kl_divergence_z_x = kl_loss_part(m=inference_outputs['z_x_m'], v=inference_outputs['z_x_v'], mask=train_x)
        kl_divergence_z_y = kl_loss_part(m=inference_outputs['z_y_m'], v=inference_outputs['z_y_v'], mask=train_y)
        kl_divergence_z = kl_divergence_z_x + kl_divergence_z_y
        # KL on the cycle z
        kl_divergence_z_x_y = kl_loss_part(
            m=inference_outputs['z_x_y_m'], v=inference_outputs['z_x_y_v'], mask=train_x)
        kl_divergence_z_y_x = kl_loss_part(
            m=inference_outputs['z_y_x_m'], v=inference_outputs['z_y_x_v'], mask=train_y)
        kl_divergence_z_cycle = kl_divergence_z_x_y + kl_divergence_z_y_x

        # Distance between modality latent space embeddings
        def z_loss(z_x, z_y, mask_x, mask_y):
            out = torch.zeros(mask_x.shape[0], device=self.device)
            mask = torch.ravel(torch.nonzero(torch.ravel(mask_x * mask_y)))
            mask_x_idx = torch.ravel(torch.nonzero(torch.ravel(mask_x)))
            mask_y_idx = torch.ravel(torch.nonzero(torch.ravel(mask_y)))
            mask_x_sub = torch.ravel(torch.nonzero(torch.ravel(mask_y[mask_x_idx, :])))
            mask_y_sub = torch.ravel(torch.nonzero(torch.ravel(mask_x[mask_y_idx, :])))
            out[mask] = torch.nn.MSELoss(reduction='none')(
                z_x[mask_x_sub, :], z_y[mask_y_sub, :]).sum(dim=1)
            return out

        z_distance_paired = z_loss(z_x=inference_outputs['z_x_m'], z_y=inference_outputs['z_y_m'],
                                   mask_x=train_x, mask_y=train_y)
        # TODO one issue with z distance loss in cycle could be that the encoders and decoders learn to cheat
        #  and encode all cells from one species to one z region, even if they are from the cycle (e.g.
        #  z_x and z_x_y have more similar embeddings and z_y and z_y_x as well)
        z_distance_cycle_x = z_loss(z_x=inference_outputs['z_x_m'], z_y=inference_outputs['z_x_y_m'],
                                    mask_x=train_x, mask_y=train_x)
        z_distance_cycle_y = z_loss(z_x=inference_outputs['z_y_m'], z_y=inference_outputs['z_y_x_m'],
                                    mask_x=train_y, mask_y=train_y)

        z_distance_cycle = z_distance_cycle_x + z_distance_cycle_y

        # Correlation between both decodings within cycle for orthologous genes
        # TODO This could be also NegLL of one prediction against the other, although unsure how well
        #  this would fit wrt normalisation used in both species (e.g. gene means may be different due to
        #  different expression of otehr genes)
        # TODO Could be decayed towards the end after the matched cell types were primed
        #  to enable species-specific flexibility as not all orthologues are functionally related
        #   Alternatively: Could do weighted correlation where sum of all weights is constant but can be learned to be
        #   distributed differently across genes
        def expr_correlation_loss(x_m, y_m, mask_x, mask_y):
            # Consideration for correlation loss:
            # If x_m and y_m have no features we need to set the loss to 0 the loss woul be 1 (1-corr, corr=0),
            # when we have features we need to set loss to 1-corr to have loss of 0 when correlation==1
            out = torch.zeros(mask_x.shape[0], device=self.device)
            if x_m.shape[1] > 0 and y_m.shape[1] > 0:
                mask = torch.ravel(torch.nonzero(torch.ravel(mask_x * mask_y)))
                mask_x_idx = torch.ravel(torch.nonzero(torch.ravel(mask_x)))
                mask_y_idx = torch.ravel(torch.nonzero(torch.ravel(mask_y)))
                mask_x_sub = torch.ravel(torch.nonzero(torch.ravel(mask_y[mask_x_idx, :])))
                mask_y_sub = torch.ravel(torch.nonzero(torch.ravel(mask_x[mask_y_idx, :])))
                x_m = x_m[mask_x_sub, :]
                y_m = y_m[mask_y_sub, :]

                def center(x):
                    return x - x.mean(dim=1, keepdim=True)

                out[mask] = 1 - torch.nn.CosineSimilarity()(center(x_m), center(y_m))
            return out

        corr_cycle_x = expr_correlation_loss(
            x_m=generative_outputs['x_y_x_m'][:, self.gene_map.orthology_output(modality='x', device=self.device)],
            y_m=generative_outputs['x_y_m'][:, self.gene_map.orthology_output(modality='y', device=self.device)],
            mask_x=train_x, mask_y=train_x)
        corr_cycle_y = expr_correlation_loss(
            x_m=generative_outputs['y_x_y_m'][:, self.gene_map.orthology_output(modality='y', device=self.device)],
            y_m=generative_outputs['y_x_m'][:, self.gene_map.orthology_output(modality='x', device=self.device)],
            mask_x=train_y, mask_y=train_y)
        corr_cycle = corr_cycle_x + corr_cycle_y

        # Overall loss
        loss = (reconst_loss * reconstruction_weight + reconst_loss_cycle * reconstruction_cycle_weight +
                kl_divergence_z * kl_weight + kl_divergence_z_cycle * kl_cycle_weight +
                z_distance_paired * z_distance_paired_weight + z_distance_cycle * z_distance_cycle_weight +
                corr_cycle * corr_cycle_weight)

        # TODO Currently this does not account for a different number of samples per batch due to masking
        return LossRecorder(
            n_obs=loss.shape[0], loss=loss.mean(), loss_sum=loss.sum(),
            reconstruction_loss=reconst_loss.sum(), kl_local=kl_divergence_z.sum(),
            reconstruction_loss_cycle=reconst_loss_cycle.sum(), kl_local_cycle=kl_divergence_z_cycle.sum(),
            z_distance_paired=z_distance_paired.sum(), z_distance_cycle=z_distance_cycle.sum(),
            corr_cycle=corr_cycle.sum(),
            reconst_loss_x_x=reconst_loss_x_x.sum(), reconst_loss_x_y=reconst_loss_x_y.sum(),
            reconst_loss_y_y=reconst_loss_y_y.sum(), reconst_loss_y_x=reconst_loss_y_x.sum(),
            reconst_loss_x_y_x=reconst_loss_x_y_x.sum(), reconst_loss_y_x_y=reconst_loss_y_x_y.sum(),
            kl_divergence_z_x=kl_divergence_z_x.sum(), kl_divergence_z_y=kl_divergence_z_y.sum(),
            kl_divergence_z_x_y=kl_divergence_z_x_y.sum(), kl_divergence_z_y_x=kl_divergence_z_y_x.sum(),
            z_distance_cycle_x=z_distance_cycle_x.sum(), z_distance_cycle_y=z_distance_cycle_y.sum(),
            corr_cycle_x=corr_cycle_x.sum(), corr_cycle_y=corr_cycle_y.sum(),
        )


def _get_dict_if_none(param):
    param = {} if not isinstance(param, dict) else param
    return param
