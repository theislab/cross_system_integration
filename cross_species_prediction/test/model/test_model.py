import scanpy as sc
import numpy as np
import pandas as pd

from cross_species_prediction.model._multied import Model


def mock_adata():
    adata = sc.AnnData(np.ones((4, 5)) * [1, 2, 3, 4, 5],
                       obsm={'species_ratio': np.array([  # Species ratios
                           [1, 0],
                           [0, 1],
                           [0.5, 0.5],
                           [0.1, 0.9]]),
                           'eval_o': np.array([0, 0, 0, 1]).reshape(-1, 1),  # Eval only orthologues},
                           'cov_species': np.ones((4, 2, 1)) * [[1], [2]]},  # Species-metadata map
                       var=pd.DataFrame({'species': ['K'] * 3 + ['L'] * 2},
                                        index=['a', 'b', 'c', 'd', 'e']),  # Species of gene
                       uns={'orthologues': pd.DataFrame({'K': ['c'], 'L': ['d']}),  # Orthgologue map
                            'species_order': ['K', 'L']})  # Order of species
    return adata


def test_model():
    print('testing model')
    adata = mock_adata()
    Model.setup_anndata(adata)
    model = Model(adata=adata)
    model.train()