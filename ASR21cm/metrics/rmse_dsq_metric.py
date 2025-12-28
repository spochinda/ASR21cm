import torch

from ASR21cm.utils import calculate_power_spectrum
from basicsr.utils.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register()
def calculate_rmse_dsq(sr, hr, **kwargs):
    """Calculate RMSE in Delta^2(k) power spectrum space.

    Computes mean in log space, then converts to real space before calculating RMSE.

    Args:
        sr (Tensor): Super-resolution cube with shape (N, C, H, W, D)
        hr (Tensor): High-resolution cube with shape (N, C, H, W, D)
        **kwargs: Additional arguments including mean and std for denormalization

    Returns:
        float: RMSE of Delta^2(k) values in real space
    """
    mean = kwargs.get('mean', 0.0)
    std = kwargs.get('std', 1.0)

    assert sr.shape == hr.shape, (f'Image shapes are different: {sr.shape}, {hr.shape}.')

    # Denormalize
    sr = sr * std + mean
    hr = hr * std + mean

    # Calculate power spectra
    k_hr, dsq_hr = calculate_power_spectrum(data_x=hr, Lpix=3, kbins=100, dsq=True, method='torch', device='cpu')
    k_sr, dsq_sr = calculate_power_spectrum(data_x=sr, Lpix=3, kbins=100, dsq=True, method='torch', device='cpu')

    # Calculate mean in log space
    log10dsq_hr = torch.log10(dsq_hr)
    log10dsq_sr = torch.log10(dsq_sr)

    log10dsq_hr_mean = torch.nanmean(log10dsq_hr)
    log10dsq_sr_mean = torch.nanmean(log10dsq_sr)

    # Convert back to real space
    dsq_hr_mean = 10 ** log10dsq_hr_mean
    dsq_sr_mean = 10 ** log10dsq_sr_mean

    # Compute MSE and RMSE in real space
    mse = (dsq_hr_mean - dsq_sr_mean) ** 2
    rmse = torch.sqrt(mse)
    rmse = rmse.item()

    return rmse
