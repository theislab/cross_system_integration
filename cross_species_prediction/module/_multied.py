import numpy as np
import torch

from torch.distributions import kl_divergence, Normal

from scvi import _CONSTANTS
from scvi.module.base import BaseModuleClass, LossRecorder, auto_move_data

from cross_species_prediction.nn._base_components import MultiDecoder, MultiEncoder
from cross_species_prediction.model._gene_maps import GeneMap
from cross_species_prediction.module.metrics import gaussian_nll_mask
from cross_species_prediction.constants import FLOAT_NN

torch.backends.cudnn.benchmark = True


class Multied(BaseModuleClass):
    """

    Parameters
    ----------
    n_input
        Number of input genes
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
            gene_maps: GeneMap,
            n_cov: int,
            n_species: int,
            n_latent: int = 10,
            encoder_params: dict = {},
            decoder_params: dict = {}
    ):
        """

        :param n_input:
        :param n_batch:
        :param n_species:
        :param n_hidden:
        :param n_latent:
        :param n_layers:
        :param loss: mse or corr
        """
        super().__init__()
        self.n_species = n_species
        self.gene_maps = gene_maps

        # z encoder goes from the n_input-dimensional data to an n_latent-d
        # latent space representation
        self.encoder = MultiEncoder(
            n_input=gene_maps.n_genes_mapped,
            n_cov=n_cov,
            n_output=n_latent,
            n_encoders=self.n_species,
            **encoder_params,
        )
        # decoder goes from n_latent-dimensional space to n_input-d data
        self.decoder = MultiDecoder(
            n_input=n_latent,
            n_cov_species=n_cov,
            n_cov_shared=n_cov + self.n_species,
            n_output_species=self.gene_maps.gene_numbers['species_specific'],
            n_output_shared=self.gene_maps.gene_numbers['orthologues'],
            shared_first=self.gene_maps.orthologues_first,
            **decoder_params
        )

    def _get_inference_input(self, tensors, **kwargs):
        """Parse the dictionary to get appropriate args"""
        x = tensors[_CONSTANTS.X_KEY]
        species_ratio = tensors['species_ratio']
        cov_species = tensors['cov_species']

        # Species-specific expression
        x_species = self.expression_species(x=x)

        input_dict = dict(
            x_species=x_species,
            cov_species=cov_species,
            species_ratio=species_ratio
        )
        return input_dict

    def _get_generative_input(self, tensors, inference_outputs, **kwargs):
        z = inference_outputs["z"]
        species_ratio = tensors['species_ratio']
        cov_species = tensors['cov_species']

        # Orthologue decoder covariates = cov_mixup & species_ratio
        cov_mixup = self.covariate_mixup(cov_species=cov_species, species_ratio=species_ratio)
        cov_shared = torch.cat((species_ratio, cov_mixup), dim=-1)

        input_dict = dict(
            z=z,
            cov_species=cov_species,
            cov_shared=cov_shared
        )
        return input_dict

    @auto_move_data
    def inference(self, x_species, cov_species, species_ratio):
        """
        High level inference method.

        Runs the inference (encoder) model.
        """
        outputs = self.encoder(x_species=x_species, cov_species=cov_species, species_ratio=species_ratio)
        return outputs

    @auto_move_data
    def generative(self, z, cov_species, cov_shared):
        """Runs the generative model."""
        outputs = self.decoder(x=z, cov_species=cov_species, cov_shared=cov_shared)
        return outputs

    def loss(
            self,
            tensors,
            inference_outputs,
            generative_outputs,
    ):
        x = tensors[_CONSTANTS.X_KEY]
        species_ratio = tensors['species_ratio']
        mixup_map = self.expression_mixup_map(species_ratio=species_ratio)
        eval_o = tensors['eval_o']

        qz_m = inference_outputs['qz_m']
        qz_v = inference_outputs['qz_v']
        x_m = generative_outputs["x_m"]
        x_v = generative_outputs["x_v"]

        # Kl divergence on latent space
        kl_divergence_z = kl_divergence(Normal(qz_m, qz_v.sqrt()),
                                        Normal(torch.zeros_like(qz_m), torch.ones_like(qz_v))
                                        ).sum(dim=1)

        # Reconstruction loss (assuming expression is normalized and log-transformed)
        # Expression mixup
        x_mixup = self.expression_mixup(x=x, mixup_map=mixup_map)
        # Loss mask
        x_eval_map = self.expression_eval_map(eval_o=eval_o, mixup_map=mixup_map)
        # Masked reconstruction loss
        reconst_loss = gaussian_nll_mask(m=x_m, x=x_mixup, v=x_v, mask=x_eval_map)

        # Combined loss
        loss = torch.mean(reconst_loss + kl_divergence_z)
        return LossRecorder(loss=loss, reconstruction_loss=reconst_loss, kl_local=kl_divergence_z)

    def expression_mixup_map(self, species_ratio):
        """
        Mixup map for expression.
        For each sample specify how much of each gene is used and
        map genes to joint cross-species orthologue representation
        dim = samples*species*genes*genes_mapped
        :param species_ratio: Ratio of species in sample/mixup, dim = samples*species
        :return:
        """
        # Mixup orthologues, dim = samples*species*genes
        orthologue_mixup = species_ratio.unsqueeze(2) * \
                           (self.gene_maps.orthologue_map.expand(species_ratio.shape[0], 1, -1)).to(FLOAT_NN)
        # Use species specific genes if species is present in mixup, dim = samples*species*genes
        non_orthologue_mixup = (species_ratio.unsqueeze(2) > 0).to(FLOAT_NN) * \
                               (torch.logical_not(self.gene_maps.orthologue_map.expand(species_ratio.shape[0], 1, -1))
                                ).to(FLOAT_NN)
        # Make combined orthologue and species-specific genes maps, dim = samples*species*genes
        gene_mixup_map = (orthologue_mixup + non_orthologue_mixup).unsqueeze(3)
        # Make gene maps species specific (keep only genes from the species) and summarized
        # in terms of orthologues, dim = samples*species*genes*genes_mapped
        mixup_map = gene_mixup_map * self.gene_maps.species_maps
        return mixup_map

    def expression_mixup(self, x, mixup_map):
        """
        Mixup expression across species, using genes with jointly mapped orthologues across species.
        dim = samples * genes_mapped
        :param x: Expression, dim = samples * genes
        :param mixup_map: See expression_mixup_map function.
        :return:
        """
        # Expression mixup across species, dim = samples * genes_mapped
        expr_mixup = (x.unsqueeze(1).expand(-1, mixup_map.shape[1], -1).unsqueeze(3) * mixup_map).sum(1).sum(1)
        return expr_mixup

    def expression_species(self, x):
        """
        Per-species expression of genes with orthologues jointly mapped across species and
        non-species-specific genes set to 0.
        dim = samples * species * genes_mapped
        :param x:
        :return:
        """
        # Per-species expression with non-species genes masked, dim = samples * species * genes_mapped
        expr_species = (x.unsqueeze(1).expand(-1, self.gene_maps.species_maps.shape[0], -1).unsqueeze(3) *
                        self.gene_maps.species_maps.unsqueeze(0).expand(x.shape[0], -1, -1, -1)).sum(axis=2)
        return expr_species

    def covariate_mixup(self, cov_species, species_ratio):
        """
        Mixup covariates across species.
        dim = samples * covariates
        :param cov_species: Per-species covariates of samples used for mixup, dim = samples * species * covariates
        :param species_ratio: Ratio of species in sample/mixup, dim = samples*species
        :return:
        """
        # Metadata/cov mixup
        cov_mixup = torch.matmul(species_ratio.unsqueeze(1), cov_species).squeeze(1)
        return cov_mixup

    def expression_eval_map(self, eval_o, mixup_map):
        """
        Mask for reconstruction loss computation on expression of genes with orthologues jointly mapped across species.
        dim = samples * genes_mapped
        :param eval_o: Whether only orthologues should be used for evaluation (1) or all genes (0), dim = samples * 1
        :param mixup_map: See expression_mixup_map function.
        :return:
        """
        # Genes to evaluate/compute loss on
        # Genes to eval in each sample based on orthologues/not
        eval_genes = torch.matmul(
            torch.logical_not(eval_o).to(FLOAT_NN),
            torch.logical_not(self.gene_maps.orthologue_map).to(FLOAT_NN).unsqueeze(0)  # Eval species specific genes or not
        ) + self.gene_maps.orthologue_map  # Orthologues - always evaluated
        # Genes to eval in each sample based on mixup species and orthologue/not
        eval_map = torch.matmul(eval_genes.unsqueeze(1), mixup_map.sum(axis=1).to(FLOAT_NN)).squeeze(1)
        return eval_map
