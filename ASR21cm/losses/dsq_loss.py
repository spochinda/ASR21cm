import torch
from torch import nn as nn

from ASR21cm.utils import calculate_power_spectrum
from basicsr.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class dsq_loss(nn.Module):
    """Delta^2 (dimensionless power spectrum) loss.

    Args:
        loss_weight (float): Loss weight. Default: 1.0.
        device (str): Device for power spectrum computation. Default: 'cpu'.
    """

    def __init__(self, loss_weight=1.0, device='cpu'):
        super().__init__()
        self.loss_weight = loss_weight
        self.ps_device = device

    def forward(self, pred, target, **kwargs):
        k_hr, dsq_hr = calculate_power_spectrum(data_x=target, Lpix=3, kbins=100, dsq=True, method='torch', device=self.ps_device)
        k_sr, dsq_sr = calculate_power_spectrum(data_x=pred, Lpix=3, kbins=100, dsq=True, method='torch', device=self.ps_device)
        loss = (torch.log10(dsq_hr) - torch.log10(dsq_sr)).abs().nanmean().squeeze()
        return self.loss_weight * loss
