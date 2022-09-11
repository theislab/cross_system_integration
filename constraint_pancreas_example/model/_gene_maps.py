import numpy as np
import torch as nn

from cross_species_prediction.constants import INT_NN, FLOAT_NN


class GeneMap:
    ORTHOLOGUES = 'orthologues'
    SPECIES_SPECIFIC = 'species_specific'
    device = None

    """
    Maps of genes across species
    """

    def __init__(self, adata):
        self._build_xsplit(adata=adata)
        self._build_constraints(adata=adata)

    def _build_xsplit(self, adata):
        self._xsplit = adata.uns['xsplit']

    def _build_constraints(self, adata):
        constraints = adata.uns['y_corr'].copy()
        y = adata.var.query('xy=="y"').copy()
        y['index'] = range(y.shape[0])
        constraints['gx'] = y.loc[constraints['gx'], 'index'].values
        constraints['gy'] = y.loc[constraints['gy'], 'index'].values
        self._constraints = constraints

    @property
    def xsplit(self):
        return self._xsplit

    def constraints(self, device):
        return {'gx': nn.tensor(self._constraints['gx'], device=device, dtype=INT_NN),
                'gy': nn.tensor(self._constraints['gy'], device=device, dtype=INT_NN),
                'intercept': nn.tensor(self._constraints['intercept'], device=device, dtype=FLOAT_NN),
                'coef': nn.tensor(self._constraints['coef'], device=device, dtype=FLOAT_NN)
                }
