import torch

from basicsr.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class VPLoss(torch.nn.Module):

    def __init__(self):
        super(VPLoss, self).__init__()

    def forward(self, score, std, gt, gt_noisy):
        loss = torch.norm(score * std + (gt_noisy - gt) / std, p=2, dim=(1, 2, 3, 4))**2
        loss = torch.mean(loss)
        return loss


if __name__ == '__main__':
    loss_fn = VPLoss()
