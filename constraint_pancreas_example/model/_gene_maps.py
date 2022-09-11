import numpy as np
import torch as nn

from cross_species_prediction.constants import FLOAT_NP, FLOAT_NN


class GeneMap:
    ORTHOLOGUES = 'orthologues'
    SPECIES_SPECIFIC = 'species_specific'
    device = None

    """
    Maps of genes across species
    """

    def __init__(self, adata):
        self._build_xsplit(adata=adata)

    def _build_xsplit(self, adata):
        self._xsplit = adata.uns['xsplit']

    @property
    def xsplit(self):
        return self._xsplit
