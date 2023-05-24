import itertools
from collections import defaultdict
from typing import Optional, Union, Tuple, Dict, Literal
import numpy as np

import torch
from scvi import REGISTRY_KEYS
from scvi.module.base import BaseModuleClass, auto_move_data
from torch.distributions import Normal
from torch.distributions import kl_divergence

from constraint_pancreas_example.model._gene_maps import GeneMapInput
from constraint_pancreas_example.nn._base_components import EncoderDecoder
from constraint_pancreas_example.module._loss_recorder import LossRecorder
from constraint_pancreas_example.module._utils import *
from constraint_pancreas_example.module._priors import StandardPrior, VampPrior

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
            system_decoders: bool,
            gene_map: GeneMapInput,
            n_cov: int,
            n_system: int,  # This should be one anyways
            use_group: bool,
            mixup_alpha: Optional[float] = None,
            prior: Literal["standard_normal", "vamp"] = 'standard_normal',
            n_prior_components=100,
            pseudoinput_data=None,
            z_dist_metric: str = 'MSE',
            n_latent: int = 15,
            n_hidden: int = 256,
            n_layers: int = 2,
            dropout_rate: float = 0.1,
            data_eval=None,
            out_var_mode: str = 'feature',
            **kwargs
    ):
        super().__init__()

        # self.gene_map = gene_map
        # TODO transfer functionality of gene maps to not need the class itself needed anymore -
        #  it was used only to return tensors on correct device for this specific model type
        self.register_buffer('gm_input_filter', gene_map.input_filter(), persistent=False)
        self.use_group = use_group
        self.mixup_alpha = mixup_alpha
        self.system_decoders = system_decoders
        self.n_output = n_output
        self.z_dist_metric = z_dist_metric
        self.data_eval = data_eval

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

        if not self.system_decoders:
            self.decoder = EncoderDecoder(
                n_input=n_latent,
                n_output=n_output,
                n_cov=n_cov + n_system,
                n_hidden=n_hidden,
                n_layers=n_layers,
                dropout_rate=dropout_rate,
                sample=True,
                var_mode=out_var_mode,
                **kwargs
            )
        else:
            # Must first assign decoders to self, as in the super base model only the params that belong to self
            # are moved to the correct device
            self.decoder_0 = EncoderDecoder(
                n_input=n_latent,
                n_output=n_output,
                n_cov=n_cov,
                n_hidden=n_hidden,
                n_layers=n_layers,
                dropout_rate=dropout_rate,
                sample=True,
                var_mode=out_var_mode,
                **kwargs
            )
            self.decoder_1 = EncoderDecoder(
                n_input=n_latent,
                n_output=n_output,
                n_cov=n_cov,
                n_hidden=n_hidden,
                n_layers=n_layers,
                dropout_rate=dropout_rate,
                sample=True,
                var_mode=out_var_mode,
                **kwargs
            )
            # Which decoder belongs to which system
            self.decoder = {0: self.decoder_0, 1: self.decoder_1}

        if prior == 'standard_normal':
            self.prior = StandardPrior()
        elif prior == 'vamp':
            if pseudoinput_data is not None:
                pseudoinput_data = self._get_inference_input(pseudoinput_data)
            self.prior = VampPrior(n_components=n_prior_components, n_input=n_input, n_cov=n_cov_encoder,
                                   encoder=self.encoder,
                                   data=(pseudoinput_data['expr'],
                                         self._merge_cov(cov=pseudoinput_data['cov'],
                                                         system=pseudoinput_data['system']))
                                   )
        else:
            raise ValueError('Prior not recognised')

    def _get_inference_input(self, tensors, **kwargs):
        """Parse the dictionary to get appropriate args"""
        input_features = torch.ravel(torch.nonzero(self.gm_input_filter))
        expr = tensors[REGISTRY_KEYS.X_KEY][:, input_features]
        cov = tensors['covariates']
        system = tensors['system']
        input_dict = dict(expr=expr, cov=cov, system=system)
        return input_dict

    def _get_inference_cycle_input(self, tensors, generative_outputs, **kwargs):
        """Parse the dictionary to get appropriate args"""
        input_features = torch.ravel(torch.nonzero(self.gm_input_filter))
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

    def _get_generative_mixup_input(self, tensors, inference_outputs, mixup_setting):
        z = mixup_data(x=inference_outputs["z"], **mixup_setting)
        cov = {'x': mixup_data(x=tensors['covariates'], **mixup_setting),
               # This wouldn't really need mixup as currently use all-0 cov, but added for safety if mock covar change
               'y': mixup_data(x=self._mock_cov(tensors['covariates']), **mixup_setting)}
        system = {'x': mixup_data(x=tensors['system'], **mixup_setting),
                  'y': mixup_data(x=self._negate_zero_one(tensors['system']), **mixup_setting)}

        input_dict = dict(z=z, cov=cov, system=system)
        return input_dict

    @staticmethod
    def _negate_zero_one(x):
        # return torch.negative(x - 1)
        return torch.logical_not(x).float()

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
                if not self.system_decoders:
                    res_sub = self.decoder(x=x, cov=self._merge_cov(cov=cov, system=system))
                else:
                    res_sub = {k: torch.zeros((x.shape[0], self.n_output), device=self.device)
                               for k in ['y', 'y_m', 'y_v']}
                    system_idx = group_indices(system, return_tensors=True, device=self.device)
                    for group, idxs in system_idx.items():
                        res_sub_parts = self.decoder[group](x=x[idxs, :], cov=cov[idxs, :])
                        for k, v in res_sub_parts.items():
                            res_sub[k][idxs, :] = v
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
        generative_inputs = self._get_generative_input(
            tensors, inference_outputs, **get_generative_input_kwargs
        )
        generative_outputs = self.generative(**generative_inputs, x_x=True, x_y=True, **generative_kwargs)

        # Generative mixup
        if self.mixup_alpha is not None:
            mixup_setting = mixup_setting_generator(
                alpha=self.mixup_alpha, device=self.device, within_group=tensors['system'])
            generative_mixup_inputs = self._get_generative_mixup_input(
                tensors=tensors, inference_outputs=inference_outputs, mixup_setting=mixup_setting)
            generative_mixup_outputs = self.generative(
                **generative_mixup_inputs, x_x=True, x_y=False, **generative_kwargs)

        # Inference cycle
        inference_cycle_inputs = self._get_inference_cycle_input(
            tensors=tensors, generative_outputs=generative_outputs, **get_inference_input_kwargs)
        inference_cycle_outputs = self.inference(**inference_cycle_inputs, **inference_kwargs)
        # Generative cycle
        generative_cycle_inputs = self._get_generative_cycle_input(
            tensors=tensors, inference_cycle_outputs=inference_cycle_outputs, **get_generative_input_kwargs)
        generative_cycle_outputs = self.generative(**generative_cycle_inputs, x_x=False, x_y=True, **generative_kwargs)

        # Combine outputs of all forward passes
        inference_outputs_merged = dict(**inference_outputs)
        inference_outputs_merged.update(
            **{k.replace('z', 'z_cyc'): v for k, v in inference_cycle_outputs.items()})
        generative_outputs_merged = dict(**generative_outputs)
        if self.mixup_alpha is not None:
            generative_outputs_merged.update(
                **{k.replace('x', 'x_mixup'): v for k, v in generative_mixup_outputs.items()})
            generative_outputs_merged['x_true_mixup'] = mixup_data(tensors[REGISTRY_KEYS.X_KEY], **mixup_setting)
        generative_outputs_merged.update(
            # y_cyc (from output x) won't be present as we don't predict x in the cycle
            **{k.replace('y', 'x_cyc'): v for k, v in generative_cycle_outputs.items()})

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
            kl_cycle_weight: float = 1,
            reconstruction_weight: float = 1,
            reconstruction_mixup_weight: float = 1,
            reconstruction_cycle_weight: float = 1,
            z_distance_cycle_weight: float = 1,
            translation_corr_weight: float = 1,
            z_contrastive_weight: float = 1,
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

        # Reconstruction loss in mixup
        if self.mixup_alpha is not None:
            reconst_loss_x_mixup = reconst_loss_part(x_m=generative_outputs['x_mixup_m'],
                                                     x=generative_outputs['x_true_mixup'],
                                                     x_v=generative_outputs['x_mixup_v'])
            reconst_loss_mixup = reconst_loss_x_mixup
        else:
            reconst_loss_mixup = torch.zeros_like(reconst_loss)

        # Reconstruction loss in cycle
        reconst_loss_x_cyc = reconst_loss_part(x_m=generative_outputs['x_cyc_m'], x=x_true,
                                               x_v=generative_outputs['x_cyc_v'])
        reconst_loss_cyc = reconst_loss_x_cyc

        # Kl divergence on latent space
        kl_divergence_z = self.prior.kl(m_q=inference_outputs['z_m'], v_q=inference_outputs['z_v'],
                                        z=inference_outputs['z'])

        # KL on the cycle z
        kl_divergence_z_cyc = self.prior.kl(m_q=inference_outputs['z_cyc_m'], v_q=inference_outputs['z_cyc_v'],
                                            z=inference_outputs['z_cyc'])

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

        # Correlation between both decoded expression reconstructions
        # TODO This could be also NegLL of one prediction against the other, although unsure how well
        #  this would fit wrt normalisation used in both species (e.g. gene means may be different due to
        #  different expression of otehr genes)
        # TODO Could be decayed towards the end after the matched cell types were primed
        #  to enable species-specific flexibility as not all orthologues are functionally related
        #   Alternatively: Could do weighted correlation where sum of all weights is constant but can be learned to be
        #   distributed differently across genes

        def center_samples(x):
            return x - x.mean(dim=1, keepdim=True)

        def expr_correlation_loss(x, y):
            return 1 - torch.nn.CosineSimilarity()(center_samples(x), center_samples(y))

        transl_corr = expr_correlation_loss(
            x=generative_outputs['y_m'],
            y=generative_outputs['x_m'])

        # Contrastive loss

        def product_cosine(x, y, eps=1e-8):
            """
            Cosine similarity between all pairs of samples in x and y
            :param x:
            :param y:
            :param eps:
            :return:
            """
            # Center to get correlation
            # if corr:
            #    x = center_samples(x)
            #    y = center_samples(y)

            # Cosine between all pairs of samples
            x_n, y_n = torch.linalg.norm(x, dim=1)[:, None], torch.linalg.norm(y, dim=1)[:, None]
            x_norm = x / torch.clamp(x_n, min=eps)
            y_norm = y / torch.clamp(y_n, min=eps)
            cos = torch.mm(x_norm, y_norm.transpose(0, 1))

            return cos

        if self.use_group:
            # Contrastive loss between samples of the same group across systems,
            # based on cosine similarity between latent embeddings
            # sum_over_groups(mean(-log(cos(within_group))) + mean(-log(1-cos(between_groups))))

            # Cosine similarity between samples from the two systems
            system_idx = group_indices(tensors['system'], return_tensors=True, device=self.device)
            # TODO BUG this fails if only 1 system is present -
            #  system_idx does not have 2nd element (index out of shape)
            idx_i = system_idx[0]
            idx_j = system_idx[1]
            sim = product_cosine(inference_outputs['z_m'][idx_i, :], inference_outputs['z_m'][idx_j, :])
            eps = 1e-8
            # Precompute loss components used for positive and negative pairs
            pos_l_parts = -torch.log(torch.clamp(sim, min=eps))
            neg_l_parts = -torch.log(torch.clamp(1 - sim, min=eps))
            # Sample indices in similarity matrix belonging to each group
            group_idx_i = group_indices(tensors['group'][idx_i, :], return_tensors=False)
            group_idx_j = group_indices(tensors['group'][idx_j, :], return_tensors=False)
            n_pairs = sim.shape[0] * sim.shape[1]
            z_contrastive_pos = 0
            z_contrastive_neg = 0
            # For each group compute positive and negative loss components
            # Potential refs (similar): https://arxiv.org/pdf/1902.01889.pdf , https://arxiv.org/pdf/1511.06452.pdf
            for group, idx_group_i in group_idx_i.items():
                idx_group_j = group_idx_j.get(group, None)
                if idx_group_j is not None:
                    # Which pairs are from the same/different class
                    indices = torch.tensor(list(itertools.product(idx_group_i, idx_group_j)), device=self.device)
                    n_pos = indices.shape[0]
                    pos_pairs = torch.zeros_like(sim)
                    pos_pairs[indices[:, 0], indices[:, 1]] = 1.0
                    neg_pairs = self._negate_zero_one(pos_pairs)
                    # Similarity with positive and negative examples
                    # Up-weights bad examples (of negatives/positives),
                    # assuming we use similarity bounded by [0,1] (cosine)
                    z_contrastive_pos += (pos_l_parts * pos_pairs).sum() / n_pos
                    z_contrastive_neg += (neg_l_parts * neg_pairs).sum() / (n_pairs - n_pos)
        else:
            z_contrastive_pos, z_contrastive_neg = 0, 0
        z_contrastive = z_contrastive_pos + z_contrastive_neg
        # TODO due to class imbalance we may not get positive samples for some classes (very often).
        #  An alternative would be to construct data batches that contain more positive pairs:
        #  If batch size=n, sample n/2 points from system 1 and then find from system 2 one sample with matching class
        #  for every sampled point from system 1. To ensure that we dont introduce class imbalance bias from one system
        #  iterate between sampling first from system 1 or 2
        #  This could be done in ann_dataloader.BatchSampler; used in DataLoaderClass constructed
        #  in DataSplitter in training

        # Overall loss
        loss = (reconst_loss * reconstruction_weight +
                reconst_loss_mixup * reconstruction_mixup_weight +
                reconst_loss_cyc * reconstruction_cycle_weight +
                kl_divergence_z * kl_weight +
                kl_divergence_z_cyc * kl_cycle_weight +
                z_distance_cyc * z_distance_cycle_weight +
                transl_corr * translation_corr_weight +
                z_contrastive * z_contrastive_weight)

        return LossRecorder(
            n_obs=loss.shape[0],
            loss=loss.mean(),
            loss_sum=loss.sum(),
            reconstruction_loss=reconst_loss.sum(),
            kl_local=kl_divergence_z.sum(),
            reconstruction_loss_mixup=reconst_loss_mixup.sum(),
            reconstruction_loss_cycle=reconst_loss_cyc.sum(),
            kl_local_cycle=kl_divergence_z_cyc.sum(),
            z_distance_cycle=z_distance_cyc.sum(),
            translation_corr=transl_corr.sum(),
            z_contrastive=z_contrastive,
            z_contrastive_pos=z_contrastive_pos,
            z_contrastive_neg=z_contrastive_neg,
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
        inference_inputs = self._get_inference_input(inference_tensors)
        generative_inputs = self._get_generative_input(
            tensors=inference_tensors,
            inference_outputs=self.inference(**inference_inputs),
            cov_replace=generative_cov)
        generative_outputs = self.generative(
            **generative_inputs,
            x_x=generative_kwargs['x_x'],
            x_y=generative_kwargs['x_y'])
        pred_x = generative_outputs[generative_kwargs['pred_key'] + "_m"][:, genes]
        corr = torch.corrcoef(torch.concat([pred_x.mean(axis=0).unsqueeze(0), target_x_m.unsqueeze(0)]))[0, 1].item()
        # Dont use 0-std genes for ll
        std_filter = target_x_std > 0
        gll = torch.distributions.Normal(loc=target_x_m[std_filter], scale=target_x_std[std_filter]
                                         ).log_prob(pred_x[:, std_filter]).mean(axis=1).mean()
        return {'correlation': corr, 'GaussianLL': gll}


def _get_dict_if_none(param):
    param = {} if not isinstance(param, dict) else param
    return param
