import torch
from ASR21cm.utils import calculate_power_spectrum
from torch import nn as nn

from basicsr.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class dsq_loss(nn.Module):
    """Delta^2 Loss.

    Args:
        loss_weight (float): Loss weight for Delta^2 loss. Default: 1.0.
    """

    def __init__(self, loss_weight=1.0):
        super(dsq_loss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, pred, target, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        k_hr, dsq_hr = calculate_power_spectrum(data_x=target, Lpix=3, kbins=100, dsq=True, method='torch', device='cpu')
        k_sr, dsq_sr = calculate_power_spectrum(data_x=pred, Lpix=3, kbins=100, dsq=True, method='torch', device='cpu')
        log10dsq_hr = torch.log10(dsq_hr)
        log10dsq_sr = torch.log10(dsq_sr)
        loss = (log10dsq_hr - log10dsq_sr).abs().nanmean().squeeze()
        return self.loss_weight * loss
