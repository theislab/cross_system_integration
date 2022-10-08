import scanpy as sc
import numpy as np
import pandas as pd

from scipy import sparse

from constraint_pancreas_example.model._xylinmodel import XYLinModel


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
    adata.varm['embed'] = np.random.randn(95, 5)
    adata.var['mean'] = np.random.randn(95)
    adata.var['std'] = np.random.randn(95)

    return adata


def test_model():
    print('testing model')
    adata = mock_adata()
    adata_training = XYLinModel.setup_anndata(
        adata,
        xy_key='xy',
        gene_embed_key='embed',
        gene_mean_key='mean',
        gene_std_key='std')
    model = XYLinModel(adata=adata_training)
    model.train(max_epochs=2)
    adata_translation = XYLinModel.setup_anndata(
        adata,
        xy_key='xy',
        gene_embed_key='embed',
        gene_mean_key='mean',
        gene_std_key='std')
    translated = model.translate(
        adata=adata_translation,
        indices=None,
        give_mean=True,
        batch_size=None,
        as_numpy=True)
    embedding = model.embed(
        adata=adata_translation,
        indices=None,
        batch_size=None,
        as_numpy=True
    )
