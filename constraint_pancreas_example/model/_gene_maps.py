import abc

import numpy as np
import torch as nn
from abc import ABC

from cross_species_prediction.constants import INT_NN, FLOAT_NN


class GeneMap(ABC):
    ORTHOLOGUES = 'orthologues'
    SPECIES_SPECIFIC = 'species_specific'
    device = None

    """
    Maps of genes across species
    """

    def __init__(self, adata, build_constraint: bool = True):
        self._build_xsplit(adata=adata)
        if build_constraint:
            self._build_constraints(adata=adata)

    def _build_xsplit(self, adata):
        self._xsplit = adata.uns['xsplit']

    def _build_constraints(self, adata):
        pass

    @property
    def xsplit(self):
        return self._xsplit

    def constraints(self, device):
        pass


class GeneMapRegression(GeneMap):
    ORTHOLOGUES = 'orthologues'
    SPECIES_SPECIFIC = 'species_specific'
    device = None

    """
    Maps of genes across species
    """

    def __init__(self, adata, build_constraint: bool = True):
        super(GeneMapRegression, self).__init__(adata, build_constraint)

    def _build_constraints(self, adata):
        constraints = adata.uns['y_corr'].copy()
        y = adata.var.query('xy=="y"').copy()
        y['index'] = range(y.shape[0])
        constraints['gx'] = y.loc[constraints['gx'], 'index'].values
        constraints['gy'] = y.loc[constraints['gy'], 'index'].values
        self._constraints = constraints

    def constraints(self, device):
        return {'gx': nn.tensor(self._constraints['gx'], device=device, dtype=INT_NN),
                'gy': nn.tensor(self._constraints['gy'], device=device, dtype=INT_NN),
                'intercept': nn.tensor(self._constraints['intercept'], device=device, dtype=FLOAT_NN),
                'coef': nn.tensor(self._constraints['coef'], device=device, dtype=FLOAT_NN)
                }


class GeneMapEmbedding(GeneMap):
    ORTHOLOGUES = 'orthologues'
    SPECIES_SPECIFIC = 'species_specific'
    device = None

    """
    Maps of genes across species
    """

    def __init__(self, adata, build_constraint: bool = True):
        super(GeneMapEmbedding, self).__init__(adata, build_constraint)

    def _build_constraints(self, adata):
        self._constraints = dict()
        # TODO Could be already transposed here
        is_y = (adata.var['xy'] == 'y').values
        self._constraints['embed'] = np.array(adata.varm['embed'][is_y])
        self._constraints['mean'] = adata.var['mean'].values[is_y]
        self._constraints['std'] = adata.var['std'].values[is_y]

    @property
    def n_embed(self):
        return self._constraints['embed'].shape[1]

    def constraints(self, device):
        return {
            'embed': nn.tensor(self._constraints['embed'], device=device, dtype=FLOAT_NN),
            'mean': nn.tensor(self._constraints['mean'], device=device, dtype=FLOAT_NN),
            'std': nn.tensor(self._constraints['std'], device=device, dtype=FLOAT_NN),
        }
