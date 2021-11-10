import numpy as np
import pandas as pd
import torch as nn

from cross_species_prediction.constants import FLOAT_NP


class GeneMap:
    ORTHOLOGUES = 'orthologues'
    SPECIES_SPECIFIC = 'species_specific'

    """
    Maps of genes across species
    """

    def __init__(self, adata):
        self._build_orthologues(adata=adata)
        self._build_species_order(adata=adata)
        self._build_species_maps(adata=adata)
        self._build_orthologues_map(adata=adata)

    def _build_orthologues(self, adata):
        """
        Get gene names of orthologues
        :param adata:
        :return:
        """
        orthologues = adata.uns['orthologues'].values.ravel()
        self._orthologues = orthologues

    def _build_species_order(self, adata):
        """
        Ordered list of species
        :param adata:
        :return:
        """
        self._species_order = adata.uns['species_order'].copy()

    def _build_species_maps(self, adata):
        """
        Build species-specific map of genes to genes summarised by orthologues, so that genes
        not present in species are set to 0.
        If there are k species and n genes with o orthologues the dimensions of gene-orthologue map are
        n*(n-(o*(k-1))) so that orthologues are summarised across species.
        Species specific maps are stacked together to give outer dimension of k, e.g k*n*(n-(o*(k-1)))
        Also build gene order info.
        :param adata: Should have fields:
        - uns['orthologues'] - df specifying 1-to-1 orthologues in rows with species in columns,
            filled with gene names used in var
        - var['species'] - which species the gene comes from
        - uns['species_order'] - order in which species should be used, list of species
        :return:
        """

        # Map genes to orthologues (not species specific)

        self._n_genes = adata.var.shape[0]
        self._n_genes_mapped = self._n_genes - self._orthologues.shape[0] + adata.uns['orthologues'].shape[0]
        gene_map = np.zeros((self._n_genes, self._n_genes_mapped))

        # Map index to integers for latter determining orthologue position
        orthologues_df = adata.uns['orthologues'].copy()
        orthologues_df.index = range(adata.uns['orthologues'].shape[0])

        # Fill gene to orthologue map

        # Specify/save order in which gene-types will be stored - first orthologues and then species-specific genes
        # ordered by species
        self._orthologues_first = True

        # Ensure orthologue and species ordering of genes
        # Orthologue order
        # Assumes that var names are unique across all species and each genes is in orthologues of species only 1x,
        # e.g. each gene name is present in orthologues_df only 1x # TODO check
        self._gene_order = {gene: idx
                            for idx, data in orthologues_df.iterrows()
                            for gene in data.values}

        # Number of genes per gene group, orthologues are counted only 1x
        # For orthologues specify number, for species-specific specify list with numbers per species
        self._gene_numbers = {self.ORTHOLOGUES: orthologues_df.shape[0],
                              self.SPECIES_SPECIFIC: []}

        # Species-specific order, starting at idx positions after orthologues
        idx = orthologues_df.shape[0]
        for species in self._species_order:
            n_species_specific = 0
            genes_species = adata.var.query('species==@species').index
            for gene in genes_species:
                if gene not in self._orthologues:
                    self._gene_order[gene] = idx
                    idx += 1
                    n_species_specific += 1
            self._gene_numbers[self.SPECIES_SPECIFIC].append(n_species_specific)

        # Map between original genes order and sorted genes with merged orthologues across species
        for gene_idx, gene in enumerate(adata.var.index):
            gene_map[gene_idx, self._gene_order[gene]] = 1

        # Make species-specific gene-orthologue maps
        # Modify general gene-orthologue map to contain only species specific genes
        species_maps = []
        for species in self._species_order:
            species_maps.append(gene_map * (adata.var['species'] == species).values.reshape(-1, 1))
        species_maps = np.array(species_maps, dtype=FLOAT_NP)
        self._species_maps = species_maps

    @property
    def species_maps(self):
        """
        :return:  Species maps as tensor
        """
        return nn.tensor(self._species_maps)

    @property
    def n_species(self):
        return len(self._species_maps)

    @property
    def gene_numbers(self):
        """
        :return:  Gene numbers as dict {orthologues:n, species_specific:[n-per-species]}
        """
        return self._gene_numbers.copy()

    @property
    def n_genes_mapped(self):
        return self._n_genes_mapped

    @property
    def orthologues_first(self):
        """
        :return:  Are orthologues saved before species-specific genes
        """
        return self._orthologues_first

    def _build_orthologues_map(self, adata):
        """
        Map specifying if gene is orthologue or not. Vector of size n genes.
        :param adata: Adata that contains
        - var
        - uns['orthologues'] - df specifying 1-to-1 orthologues in rows with species in columns,
            filled with gene names used in var
        :return:
        """
        orthologue_map = np.array([g in self._orthologues for g in adata.var.index], dtype=FLOAT_NP)
        self._orthologue_map = orthologue_map

    @property
    def orthologue_map(self):
        """
        :return: Orthologue map as tensor
        """
        return nn.tensor(self._orthologue_map)
