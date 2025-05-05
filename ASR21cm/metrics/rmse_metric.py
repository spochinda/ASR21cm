import torch

from basicsr.utils.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register()
def calculate_rmse(sr, hr, **kwargs):
    """Calculate rmse.

    Args:
        sr (Tensor): Super-resolution cube with shape (N, C, H, W, D)
        hr (Tensor): High-resolution cube with shape (N, C, H, W, D)

    Returns:
        float: rmse value
    """
    mean = kwargs.get('mean', 0.0)
    std = kwargs.get('std', 1.0)

    assert sr.shape == hr.shape, (f'Image shapes are different: {sr.shape}, {hr.shape}.')
    sr = sr * std + mean
    hr = hr * std + mean
    mse = torch.mean((sr - hr)**2, dim=[-1, -2, -3])
    rmse = torch.sqrt(mse)
    rmse = rmse.mean()
    rmse = rmse.item()
    return rmse
