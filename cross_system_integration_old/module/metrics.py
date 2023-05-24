import torch as nn


def correlation(x, y, as_loss=False, reduce=True):
    """
    Correlation between samples.
    Note: Does not account for expression strength so could be combined with size_diff_ratio loss. But there are
    problems with masking anyways as number of 0 features would affect result.
    :param x:
    :param y:
    :param as_loss: Instead of correlation return correlation distance as it can function as loss
    (non negative, higher if more different), computed as 1-correlation
    :param reduce: Output mean instead of per-sample values
    :return:
    """
    vx = nn.sub(x, nn.mean(x, dim=1, keepdim=True))
    vy = nn.sub(y, nn.mean(y, dim=1, keepdim=True))
    res = nn.sum(vx * vy, dim=1, keepdim=True) * \
          (nn.rsqrt(nn.sum(vx ** 2, dim=1, keepdim=True)) * nn.rsqrt(nn.sum(vy ** 2, dim=1, keepdim=True)))
    if as_loss:
        res = 1 - res
    if reduce:
        res = nn.mean(res)
    return res


def size_diff_ratio(x, y, reduce=True):
    """
    Absolute difference of x and y sizes as ratio of x size
    :param x: ref vector
    :param y: other vector
    :param reduce: Output mean instead of per-sample values
    :return:
    """
    size_x = nn.sum(x, dim=1, keepdim=True)
    size_y = nn.sum(y, dim=1, keepdim=True)
    res = nn.divide(nn.abs(nn.sub(size_x, size_y)), size_x)
    if reduce:
        res = nn.mean(res)
    return res


def gaussian_nll_mask(m, x, v, mask):
    """
    Compute Gaussian negative log likelihood loss with sample-specific masked features.
    :param m: Predicted mean of target
    :param x: True target
    :param v: predicted v of target
    :param mask: Sample-specific feature mask of same shape as x specifying by 1/0 if
        sample-feature should be used for computing loss or not, respectively.
    :return: loss
    """
    l = nn.nn.GaussianNLLLoss(reduction='none')(m, x, v)
    l = l * mask  # Masking - set some sample-specific features to 0
    l = l.sum(dim=1) / mask.sum(dim=1)  # Normalise accounting for masking
    return l
