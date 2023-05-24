import scanpy as sc
import numpy as np
import pandas as pd

from scipy import sparse

from constraint_pancreas_example.model._xybimodel import XYBiModel


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
        var=pd.DataFrame({'xy': ['y'] * 45 + ['x'] * 50},
                         index=[str(i) for i in range(95)]),  # Make var names str to enable orthologue mapping
    )
    adata.obs['covariate_x_cont'] = list(range(200))
    adata.obs['covariate_x_cat'] = ['a'] * 100 + ['b'] * 100
    adata.obs['covariate_y_cont'] = list(range(200))
    adata.obs['train_x'] = [1] * 100 + [0] * 100
    adata.obs['train_y'] = [0] * 50 + [1] * 150
    adata.var['input'] = [1] * 20 + [0] * 25 + [1] * 20 + [0] * 30
    adata.uns['orthology'] = pd.DataFrame({
        'x': np.array([45, 46, 46, 65, 65]).astype(str),
        'y': np.array([0, 0, 1, 20, 21]).astype(str),
    })

    return adata


def test_model():
    print('testing model')
    adata = mock_adata()
    adata_training = XYBiModel.setup_anndata(
        adata,
        xy_key='xy',
        train_x_key='train_x',
        train_y_key='train_y',
        orthology_key='orthology',
        input_gene_key='input',
        # TODO improve test for adata setup
        categorical_covariate_keys_x=['covariate_x_cat'],
        continuous_covariate_keys_x=['covariate_x_cont'],
        continuous_covariate_keys_y=['covariate_y_cont'],
    )
    model = XYBiModel(adata=adata_training)
    model.train(max_epochs=2)

    adata_translation = XYBiModel.setup_anndata(
        adata,
        xy_key='xy',
        train_x_key='train_x',
        train_y_key='train_y',
        input_gene_key='input',
        categorical_covariate_keys_x=['covariate_x_cat'],
        continuous_covariate_keys_x=['covariate_x_cont'],
        continuous_covariate_keys_y=['covariate_y_cont'],
    )
    embedding_x, embedding_y = model.embed(
        adata=adata_translation,
        indices=None,
        give_mean=True,
        batch_size=None,
        as_numpy=True
    )
    assert embedding_x.shape[0] == embedding_y.shape[0] == adata_translation.shape[0]

    translated_x, translated_y = model.translate(
        adata=adata_translation,
        indices=None,
        give_mean=True,
        batch_size=None,
        as_numpy=True
    )
    assert translated_x.shape[0] == translated_y.shape[0] == adata_translation.shape[0]
