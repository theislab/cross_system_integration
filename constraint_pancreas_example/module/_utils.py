from typing import Union, Dict

import numpy as np
from collections import defaultdict

import torch


def group_indices(group: Union[np.ndarray, list, torch.Tensor], return_tensors=False, device=None) -> \
        Dict[object, Union[list, torch.Tensor]]:
    if isinstance(group, torch.Tensor):
        group = group.ravel().tolist()
    if isinstance(group, np.ndarray):
        group = group.ravel()
    else:
        group = np.array(group).ravel()

    group_idx = defaultdict(list)
    for idx, group in enumerate(group):
        group_idx[group].append(idx)

    if return_tensors:
        group_idx = {group: torch.tensor(idxs, device=device) for group, idxs in group_idx.items()}

    return group_idx


def mixup_setting_generator(alpha: float, device, n: int = None, within_group: torch.Tensor = None):
    """
    Prepare mixup settings: indices and proportions for combining
    :param alpha: Alpha for beta distn from which mixup ratios are sampled.
    If ==0 gets uniform, if <1 concave, if >1 convex. Symmetric as use same param 9alpha) for both alpha and beta.
    :param device: Device for output tensors
    :param n: Number of samples between which mixup should be created
    :param within_group: Tensor (shape=n*1) specifying groups within which mixup should be performed.
    If this is specified the param n is ignored and n of samples is determined based on within_group.
    :return: Indices and ratios for mixup pairs. Number of mixup samples equals the number of input samples.
    """

    # Params for performing mixup within a sample group or on all samples
    if within_group is None and n is None:
        raise ValueError('Either within_group or n must be specified')
    if within_group is None:
        within_group = np.ones(n)

    # Indices associated with each group within which mixup should be preformed
    group_idx = group_indices(group=within_group, return_tensors=False)

    # Generate mixup setting within every group
    idx_i = []  # 1-dim
    idx_j = []
    ratio_i = []  # 2-dim
    ratio_j = []
    for group, idxs in group_idx.items():
        ratio_i_sub = np.random.beta(alpha, alpha, size=(len(idxs), 1)).astype(np.float32)
        ratio_j_sub = 1 - ratio_i_sub
        idxs_shuffled = np.random.permutation(idxs)
        for lis, el in [(idx_i, idxs), (idx_j, idxs_shuffled), (ratio_i, ratio_i_sub), (ratio_j, ratio_j_sub)]:
            lis.append(el)

    out = {}
    for name, lis in [('idx_i', idx_i), ('idx_j', idx_j), ('ratio_i', ratio_i), ('ratio_j', ratio_j)]:
        out[name] = torch.tensor(np.concatenate(lis), device=device)

    return out


def mixup_data(x, idx_i, idx_j, ratio_i, ratio_j):
    """
    Perform mixup between samples of given tensor
    :param x: Tensor whose samples to mixup
    :param idx_i: Indices of the first sample in mixup pair
    :param idx_j: Indices of the second sample in mixup pair
    :param ratio_i: Proportion of the first sample in mixup pair
    :param ratio_j: Proportion of the second sample in mixup pair
    :return: Mixed up tensor
    """
    return x[idx_i] * ratio_i + x[idx_j] * ratio_j
