import pandas as pd
import numpy as np

from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import jaccard_score

import scib_metrics as sm


# labels - cell types
# batches - batch info


def knn_purity(distances,labels, k=30):
    """
    How many of K nearest nieghbors are of the same label as the cell
    :parm distances: Anndata KNN distances, 
    assuming self is no longer included among elements with non-zero distance
    :param labels: Cell labels
    :param k: K nearest neighbours to use
    :retun: tuple of: macro-mean knn purity across labels, knn purity as series of label:purity
    """
    
    # Get k nearest neighbours of each cell
    
    def csr_row_argwhich_nonzero_min(matrix,n):
        """
        For csr matrix get from every row column indices on n smallest nonzero elements
        """
        top_n_idx=[]
        for le, ri in zip(matrix.indptr[:-1], matrix.indptr[1:]):
            non_zero=matrix.data[le:ri]
            n_non_zero=non_zero.shape[0]
            if n==n_non_zero:
                idxs=np.arange(n)
            elif n<n_non_zero:
                idxs=np.argpartition(non_zero, n)[:n]
            else:
                raise ValueError('Not enough elements in row')
            tops=matrix.indices[le + idxs]
            top_n_idx.append(tops)
        return np.array(top_n_idx)

    knn_arg=csr_row_argwhich_nonzero_min(matrix=distances,n=k)
    
    # Determine how many of the KNNs are own label
    label_map=dict(zip(range(labels.shape[0]),labels.values))
    knn_label=np.vectorize(label_map.get)(knn_arg)
    labels_temp=labels.copy()
    labels_temp.index=labels_temp.values
    knn_purity={}
    for group, indices in labels_temp.groupby(level=0).indices.items():
        knn_group=knn_label[indices,:].ravel()
        knn_purity[group]=(knn_group==group).sum()/knn_group.shape[0]
    knn_purity=pd.Series(knn_purity)
    
    return knn_purity.mean(),knn_purity


def asw_label(X,labels):
    """
    ASW label reimplementation with macro/micro scores
    :param X: Embedding
    """
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
    """
    ASW batch reimplementation with macro/micro scores
    :param X: Embedding 
    """
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
    """
    clisi with macro/micro scores
    :param X: Scanpy distance matrix
    """
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
    """
    ilisi with macro/micro scores
    :param X: Scanpy distance matrix
    """
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
    """
    Compute clusters classification metrics based on true labels and data-driven clusters
    Jaccard computed as micro/macro
    """
    
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
    """
    Cluster classification metrics for given leiden resolution
    """
    import random
    random.seed(0)
    labels_pred = sm._nmi_ari._compute_clustering_leiden(X, resolution)
    return _cluster_classification_metrics(labels=labels,clusters=labels_pred, jaccard=False)


def cluster_classification_optimized(X,labels):
    """
    Cluster classification metrics where clusters are selected based on optimal cluster resolution
    reimplemented scIB NMI ARI metrics to add random seed
    X: connectivity graph of Scanpy
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
    """
    Cluster classification metrics where clusters are mapped to labels based on majority vote
    """

    # Map clusters to labels by mode-label assignment
    labels_df=pd.DataFrame({'labels':labels,'clusters':clusters})
    cluster_map=labels_df.groupby('clusters')['labels'].agg(lambda x: pd.Series.mode(x)[0]).to_dict()
    labels_df['labels_pred']=labels_df['clusters'].map(cluster_map)
    
    return _cluster_classification_metrics(labels=labels_df['labels'],clusters=labels_df['labels_pred'])

