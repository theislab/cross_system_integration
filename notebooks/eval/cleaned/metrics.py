import pandas as pd
import numpy as np

from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import jaccard_score

import scib_metrics as sm

def asw_label(X,labels):
    asw=sm.utils.silhouette_samples(
        X=X, 
        labels=labels, 
        chunk_size=256)
    asw=(asw + 1) / 2
    asw_micro= np.mean(asw)
    asw_label=pd.DataFrame({'asw':asw,'label':labels}).groupby('label')['asw'].mean()
    asw_macro=asw_label.mean()
    return asw_micro, asw_macro, asw_label

def asw_batch(X,batches,labels):
    asw_label={}
    asws=[]
    for label in np.unique(labels):
        labels_mask = labels == label
        X_sub = X[labels_mask]
        batches_sub = batches[labels_mask]
        n_batches = len(np.unique(batches_sub))

        if (n_batches == 1) or (n_batches == X_sub.shape[0]):
            continue
        asw=sm.utils.silhouette_samples(
            X=X_sub, 
            labels=batches_sub, 
            chunk_size=256)
        # ASW is negative when distance between batches is smaller than within batches
        # This deviates from scib implementation where abs of ASW is used 
        # (hence also different scaling)
        asw=(1-asw)/2
        asws.extend(asw)
        asw_label[label]=np.mean(asw)
    asw_label=pd.Series(asw_label)
    asw_macro=asw_label.mean()
    asw_micro=np.mean(asws)

    return asw_micro, asw_macro, asw_label

def clisi(X, labels):
    labels_code = np.asarray(pd.Categorical(labels).codes)
    clisi = sm.lisi_knn(X, labels_code, perplexity=None)
    nlabels = len(np.unique(labels))
    clisi = (nlabels - clisi) / (nlabels - 1)
    clisi_micro = np.nanmedian(clisi)
    clisi_label=pd.DataFrame({'clisi':clisi,'label':labels}).groupby('label').apply(
        lambda x: np.nanmedian(x))
    clisi_macro=clisi_label.mean()
    return clisi_micro, clisi_macro, clisi_label


def ilisi(X, labels, batches):
    batches_code = np.asarray(pd.Categorical(batches).codes)
    ilisi = sm.lisi_knn(X, batches_code, perplexity=None)
    nbatches = len(np.unique(batches))
    ilisi = (ilisi - 1) / (nbatches - 1)
    ilisi_micro = np.nanmedian(ilisi)
    ilisi_label=pd.DataFrame({'ilisi':ilisi,'label':labels}).groupby('label').apply(
        lambda x: np.nanmedian(x))
    ilisi_macro = ilisi_label.mean()
    return ilisi_micro, ilisi_macro, ilisi_label


def _cluster_classification_metrics(labels,clusters, jaccard:bool=True):
    
    labels_df=pd.DataFrame({'labels':labels,
                            'clusters':clusters})
    
    # Micro metrics computed with sklearn
    nmi = normalized_mutual_info_score(
        labels_df['labels'], labels_df['clusters'], average_method="arithmetic")
    ari = adjusted_rand_score(labels_df['labels'], labels_df['clusters'])
    
    if jaccard:
        jaccard_micro=jaccard_score(labels_df['labels'], labels_df['clusters'], average='micro')

        # Class-wise jaccard index implementation
        jaccard_label=dict()
        for label in labels_df['labels'].unique():
            is_label=labels_df[['labels','clusters']]==label
            jaccard_label[label]=is_label.all(axis=1).sum()/is_label.any(axis=1).sum()
        jaccard_label=pd.Series(jaccard_label)
        jaccard_macro=jaccard_label.mean() # Was cheched to match sklearn
    
        return nmi, ari, jaccard_micro, jaccard_macro, jaccard_label
    else:
        return nmi, ari


def _compute_cluster_clasification_leiden(X,labels,resolution):
    import random
    random.seed(0)
    labels_pred = sm._nmi_ari._compute_clustering_leiden(X, resolution)
    return _cluster_classification_metrics(labels=labels,clusters=labels_pred, jaccard=False)


def cluster_classification_optimized(X,labels):
    """
    reimplemented NMI ARI metrics to add random seed
    X: connectivity graph
    """
    from sklearn.utils import check_array
    
    X = check_array(X, accept_sparse=True, ensure_2d=True)
    sm.utils.check_square(X)
    n = 10
    resolutions = np.array([2 * x / n for x in range(1, n + 1)])
    try:
        from joblib import Parallel, delayed

        out = Parallel(n_jobs=-1)(delayed(_compute_cluster_clasification_leiden)(
            X=X, labels=labels,resolution=r) for r in resolutions)
    except ImportError:
        warnings.warn("Using for loop over clustering resolutions. `pip install joblib` for parallelization.")
        out = [_compute_cluster_clasification_leiden(
            X=X, labels=labels,resolution=r) for r in resolutions]
    nmi=[]
    ari=[]
    for data in out:
        for value,storage in zip(data,(nmi,ari)):
            storage.append(value)
    nmi=max(nmi)
    ari=max(ari)

    return nmi, ari

def cluster_classification(labels,clusters):

    # Map clusters to labels by mode-label assignment
    labels_df=pd.DataFrame({'labels':labels,'clusters':clusters})
    cluster_map=labels_df.groupby('clusters')['labels'].agg(lambda x: pd.Series.mode(x)[0]).to_dict()
    labels_df['labels_pred']=labels_df['clusters'].map(cluster_map)
    
    return _cluster_classification_metrics(labels=labels_df['labels'],clusters=labels_df['labels_pred'])

