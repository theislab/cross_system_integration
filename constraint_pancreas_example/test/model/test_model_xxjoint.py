import scanpy as sc
import numpy as np
import pandas as pd

from scipy import sparse

from constraint_pancreas_example.model._xxjointmodel import XXJointModel
from constraint_pancreas_example.train import WeightScaling


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
        var=pd.DataFrame(index=[str(i) for i in range(95)]),  # Make var names str to enable orthologue mapping
    )
    adata.obs['covariate_cont'] = list(range(200))
    adata.obs['covariate_cat'] = ['a'] * 50 + ['b'] * 50 + ['c'] * 50 + ['d'] * 50
    adata.obs['system'] = ['a'] * 100 + ['b'] * 100
    adata.obs['group'] = ((['a'] * 25 + ['b'] * 25) * 2) * 2
    adata.var['input'] = [1] * 20 + [0] * 25 + [1] * 20 + [0] * 30

    return adata


def test_model():
    print('testing model')
    adata = mock_adata()
    adata_training = XXJointModel.setup_anndata(
        adata,
        input_gene_key='input',
        system_key='system',
        group_key='group',
        # TODO improve test for adata setup
        categorical_covariate_keys=['covariate_cat'],
        continuous_covariate_keys=['covariate_cont'],
    )

    # Test both options with shared or per-system decoder
    model = XXJointModel(adata=adata_training, mixup_alpha=None, system_decoders=False)
    model.train(max_epochs=2,
                plan_kwargs={'loss_weights': {
                    'kl_weight': 2,
                    'kl_cycle_weight': WeightScaling(weight_start=0, weight_end=1,
                                                     point_start=0, point_end=2, update_on='step')
                }})
    model = XXJointModel(adata=adata_training, mixup_alpha=0.4, system_decoders=True)
    model.train(max_epochs=2)

    # TODO test registration of new adata

    embedding = model.embed(
        adata=adata_training,
        indices=None,
        give_mean=True,
        batch_size=None,
        as_numpy=True
    )
    assert embedding.shape[0] == adata_training.shape[0]

    translated_y = model.translate(
        adata=adata_training,
        indices=None,
        covariates=pd.Series({'covariate_cat': 'a', 'covariate_cont': 1}),
        give_mean=True,
        batch_size=None,
        as_numpy=True
    )
    assert translated_y.shape[0] == adata_training.shape[0]
