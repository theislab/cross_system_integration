import numpy as np
import torch as nn

from cross_system_integration.constants import INT_NN, FLOAT_NN


class GeneMapConstraint:

    def __init__(self, adata):
        self._build_constraints(adata=adata)

    def _build_constraints(self, adata):
        pass

    def constraints(self, device):
        pass


class GeneMapSplit:

    def __init__(self, adata):
        self._build_xsplit(adata=adata)

    def _build_xsplit(self, adata):
        self._xsplit = adata.uns['xsplit']

    @property
    def xsplit(self):
        return self._xsplit


class GeneMapInput:

    def __init__(self, adata):
        self._build_input_filter(adata=adata)

    def _build_input_filter(self, adata):
        self._input_filter = adata.var['input'].values.ravel()

    def input_filter(self, device=None):
        return nn.tensor(self._input_filter, device=device, dtype=INT_NN)


class GeneMapXYBimodel(GeneMapSplit, GeneMapInput):
    def __init__(self, adata):
        GeneMapSplit.__init__(self, adata=adata)
        GeneMapInput.__init__(self, adata=adata)
        self._build_orthology_output(adata=adata)

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


class GeneMapRegression(GeneMapSplit, GeneMapConstraint):
    """
    Maps of genes across species
    """

    def __init__(self, adata):
        GeneMapSplit.__init__(self,adata=adata)
        GeneMapConstraint.__init__(self, adata=adata)
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


class GeneMapEmbedding(GeneMapSplit, GeneMapConstraint):
    """
    Maps of genes across species
    """

    def __init__(self, adata):
        GeneMapSplit.__init__(self,adata=adata)
        GeneMapConstraint.__init__(self, adata=adata)
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
