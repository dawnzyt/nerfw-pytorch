import torch
import torch.nn as nn


def mse(img_pred, img, mask=None, reduction='mean'):
    value = (img_pred - img) ** 2
    if mask is not None:
        value = value[mask]
    if reduction == 'mean':
        return torch.mean(value)
    return value


def psnr(img_pred, img, mask=None, reduction='mean'):
    """

    :param img_pred: torch.tensor (h,w,3)
    :param img: same
    :param mask:
    :param reduction:
    :return:
    """
    return -10 * torch.log10(mse(img_pred, img, mask, reduction))
