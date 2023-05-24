import scanpy as sc
import numpy as np
import pandas as pd

from scipy import sparse

from cross_system_integration_old.model._multied import Model


def mock_adata_processed():
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
                            'orders': {'species': ['K', 'L']}})  # Order of species
    return adata


def mock_adata():
    # Make large random data with different gene distns so that hvg selection and knn search works
    adata = sc.AnnData(sparse.csr_matrix(np.exp(np.concatenate([
        np.random.normal(1, 0.5, (200, 5)),
        np.random.normal(1.1, 0.00237, (200, 5)),
        np.random.normal(1.3, 0.35, (200, 5)),
        np.random.normal(2, 0.111, (200, 5)),
        np.random.normal(2.2, 0.3, (200, 5)),
        np.random.normal(2.7, 0.01, (200, 5)),
        np.random.normal(1, 0.001, (200, 5)),
        np.random.normal(0.00001, 0.4, (200, 5)),
        np.random.normal(0.2, 0.91, (200, 5)),
        np.random.normal(0.1, 0.0234, (200, 5)),
        np.random.normal(0.00005, 0.1, (200, 5)),
        np.random.normal(0.05, 0.001, (200, 5)),
        np.random.normal(0.023, 0.3, (200, 5)),
        np.random.normal(0.6, 0.13, (200, 5)),
        np.random.normal(0.9, 0.5, (200, 5)),
        np.random.normal(1, 0.0001, (200, 5)),
        np.random.normal(1.5, 0.05, (200, 5)),
        np.random.normal(2, 0.009, (200, 5)),
        np.random.normal(1, 0.0001, (200, 5)),
    ], axis=1))),
        obs={'species': ['K'] * 100 + ['L', 'M'] * 50,  # Species
             'cov_c': ['a', 'b'] * 100,  # Covariates - categorical example
             'cov_n': [0, 1] * 100},  # Covariates - continous (numerical) example
        var=pd.DataFrame({'species':
                              ['K', 'L', 'M'] * 30 + ['K'] * 1 + ['L'] * 2 + ['M'] * 2},
                         index=[str(i) for i in range(95)]),  # Make var names str to enable orthologue mapping
        # Species of gene
        uns={'orthologues': pd.DataFrame(np.array(range(90)).reshape(30, 3),
                                         columns=['K', 'L', 'M']).astype(str),  # Orthgologue map
             })
    return adata


def test_model():
    print('testing model')
    adata = mock_adata()
    adata_training = Model.setup_anndata(
        adata,
        species_key='species',
        cov_cat_keys=['cov_c'],
        cov_cont_keys=['cov_n'],
        orthologues_map_key='orthologues',
        seed=0,
        alpha=0.4,
        random_mixup_ratio=1,
        similar_mixup_ratio=1,
        n_hvgs=5,
        n_pcs=3,
        k=2)
    model = Model(adata=adata_training)
    model.train(max_epochs=2)
    adata_translation = Model.setup_anndata(
        adata,
        species_key='species',
        cov_cat_keys=['cov_c'],
        cov_cont_keys=['cov_n'],
        orthologues_map_key='orthologues',
        seed=0,
        alpha=0.4,
        random_mixup_ratio=0,
        similar_mixup_ratio=0,
        n_hvgs=5,
        n_pcs=3,
        k=2)
    translated = model.translate(
        prediction_metadata=adata.obs,
        species_key='species',
        adata=adata_translation,
        indices=None,
        give_mean=True,
        batch_size=None,
        as_numpy=True)
