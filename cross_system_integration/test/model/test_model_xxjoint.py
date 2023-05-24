import math

import scanpy as sc
import numpy as np
import pandas as pd

from scipy import sparse

from cross_system_integration.model._xxjointmodel import XXJointModel
from cross_system_integration.train import WeightScaling


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
    # Also deal with missing group vals
    adata.obs['group'] = ((['a'] * 25 + ['b'] * 20 + [np.nan] * 5) * 2) * 2
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

    # Test single decoder and different losses (with w scale) and in z_distance_cycle metric MSE standard loss
    adata_training.uns['eval_info'] = {'evalMetric': {
        'cells_in': ['1', '4'],
        'switch_system': True,
        'cells_target': ['2', '4'],
        'genes': ['2', '1', '3']}}
    model = XXJointModel(adata=adata_training,
                         mixup_alpha=None,
                         system_decoders=False,
                         z_dist_metric='KL',
                         adata_eval=adata_training,
                         out_var_mode='linear',
                         )
    model.train(max_epochs=2,
                log_every_n_steps=1,
                batch_size=math.ceil(adata_training.n_obs / 2.0),
                check_val_every_n_epoch=1,
                val_check_interval=1,
                plan_kwargs={
                    'log_on_epoch': False,
                    'log_on_step': True,
                    'loss_weights': {
                        'kl_weight': 2,
                        'kl_cycle_weight': dict(weight_start=0, weight_end=1,
                                                point_start=1, point_end=3, update_on='step')
                    }})
    # Test double decoder and mixup
    model = XXJointModel(adata=adata_training, mixup_alpha=0.4, system_decoders=True)
    model.train(max_epochs=2)
    # Test vamp prior
    model = XXJointModel(adata=adata_training, prior='vamp')
    model.train(max_epochs=2)
    # Test training with different indices at different training runs
    model = XXJointModel(adata=adata_training)
    n_indices1 = int(adata_training.shape[0] * 0.5)
    indices_permuted = np.random.permutation(adata_training.shape[0])
    indices1 = indices_permuted[:n_indices1]
    indices2 = indices_permuted[n_indices1:]
    model.train(max_epochs=1, indices=indices1)
    np.testing.assert_array_equal(np.sort(indices1),
                                  np.sort(np.concatenate([model.train_indices, model.validation_indices])))
    model.train(max_epochs=1, indices=indices2)
    np.testing.assert_array_equal(np.sort(indices2),
                                  np.sort(np.concatenate([model.train_indices, model.validation_indices])))

    # TODO test registration of new adata

    embedding = model.embed(
        adata=adata_training,
        indices=None,
        give_mean=True,
        batch_size=None,
        as_numpy=True
    )
    assert embedding.shape[0] == adata_training.shape[0]
    embedding = model.embed(
        adata=adata_training,
        indices=None,
        cycle=True,
        give_mean=True,
        batch_size=None,
        as_numpy=True
    )

    translated_y = model.translate(
        adata=adata_training,
        indices=None,
        covariates=pd.Series({'covariate_cat': 'a', 'covariate_cont': 1}),
        give_mean=True,
        batch_size=None,
        as_numpy=True
    )
    assert translated_y.shape[0] == adata_training.shape[0]
    _, translated_y_v = model.translate(
        adata=adata_training,
        indices=None,
        covariates=pd.Series({'covariate_cat': 'a', 'covariate_cont': 1}),
        give_mean=True,
        give_var=True,
        batch_size=None,
        as_numpy=True
    )
    assert translated_y_v.shape[0] == adata_training.shape[0]
    translated_y = model.translate(
        adata=adata_training[[1, 2], :],
        indices=None,
        covariates=pd.DataFrame({'covariate_cat': ['a', 'b'], 'covariate_cont': [1] * 2}),
        give_mean=True,
        batch_size=None,
        as_numpy=True
    )
