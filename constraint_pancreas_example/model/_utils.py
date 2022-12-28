import pandas as pd
from typing import Optional, List, Union


def prepare_metadata(meta_data: pd.DataFrame,
                     cov_cat_keys: Optional[list] = None,
                     cov_cont_keys: Optional[list] = None,
                     orders=None):
    """

    :param meta_data: Dataframe containing species and covariate info, e.g. from non-registered adata.obs
    :param cov_cat_keys: List of categorical covariates column names.
    :param cov_cont_keys: List of continuous covariates column names.
    :param orders: Defined orders for species or categorical covariates. Dict with keys being
    'species' or categorical covariates names and values being lists of categories. May contain more/less
    categories than data.
    :return: covariate data, dict with order of categories per covariate, dict with keys categorical and continuous
    specifying lists of covariates
    """
    if cov_cat_keys is None:
        cov_cat_keys = []
    if cov_cont_keys is None:
        cov_cont_keys = []

    def dummies_categories(values: pd.Series, categories: Union[List, None] = None):
        """
        Make dummies of categorical covariates. Use specified order of categories.
        :param values: Categories for each observation.
        :param categories: Order of categories to use.
        :return: dummies, categories. Dummies - one-hot encoding of categories in same order as categories.
        """
        if categories is None:
            categories = pd.Categorical(values).categories.values
        values = pd.Series(pd.Categorical(values=values, categories=categories, ordered=True),
                           index=values.index, name=values.name)
        dummies = pd.get_dummies(values, prefix=values.name)
        return dummies, list(categories)

    # Covariate encoding
    # Save order of covariates and categories
    cov_dict = {'categorical': cov_cat_keys, 'continuous': cov_cont_keys}
    # One-hot encoding of categorical covariates
    orders_dict = {}
    cov_cat_data = []
    for cov_cat_key in cov_cat_keys:
        cat_dummies, cat_order = dummies_categories(
            values=meta_data[cov_cat_key], categories=orders.get(cov_cat_key, None))
        cov_cat_data.append(cat_dummies)
        orders_dict[cov_cat_key] = cat_order
    # Prepare single cov array for all covariates and in per-species format
    cov_data_parsed = pd.concat(cov_cat_data + [meta_data[cov_cont_keys]], axis=1)
    return cov_data_parsed, orders_dict, cov_dict
