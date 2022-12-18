import numpy as np
import torch as nn

from cross_species_prediction.constants import INT_NN, FLOAT_NN


class GeneMap():
    ORTHOLOGUES = 'orthologues'
    SPECIES_SPECIFIC = 'species_specific'
    device = None

    """
    Maps of genes across species
    """

    def __init__(self, adata):
        self._build_xsplit(adata=adata)
        self._setup(adata=adata)

    def _build_xsplit(self, adata):
        self._xsplit = adata.uns['xsplit']

    def _setup(self, adata):
        pass

    @property
    def xsplit(self):
        return self._xsplit


class GeneMapInput(GeneMap):
    def __init__(self, adata):
        super(GeneMapInput, self).__init__(adata)

    def _setup(self, adata):
        self._build_input_filter(adata=adata)
        self._build_orthology_output(adata=adata)

    def _build_input_filter(self, adata):
        self._input_filter = adata.var['input'].values.ravel()

    def _build_orthology_output(self, adata):
        var_names_x = adata.var_names[:self.xsplit].values
        var_names_y = adata.var_names[self.xsplit:].values
        orthologs_x = []
        orthologs_y = []
        for i, (x, y) in adata.uns['orthology'][['x', 'y']].iterrows():
            orthologs_x.append(np.argwhere(var_names_x == x)[0][0])
            orthologs_y.append(np.argwhere(var_names_y == y)[0][0])
        self._orthologs_output = {'x': orthologs_x, 'y': orthologs_y}

    def input_filter(self, device):
        return nn.tensor(self._input_filter, device=device, dtype=INT_NN)

    def orthology_output(self, modality, device):
        return nn.tensor(self._orthologs_output[modality], device=device, dtype=INT_NN)


class GeneMapRegression(GeneMap):
    """
    Maps of genes across species
    """

    def __init__(self, adata):
        super(GeneMapRegression, self).__init__(adata)

    def _setup(self, adata):
        self._build_constraints(adata=adata)

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
    """
    Maps of genes across species
    """

    def __init__(self, adata):
        super(GeneMapEmbedding, self).__init__(adata)

    def _setup(self, adata):
        self._build_constraints(adata=adata)

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
