import torch

from basicsr.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class VPLoss(torch.nn.Module):

    def __init__(self):
        super(VPLoss, self).__init__()
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, eps_theta, std, gt, gt_noisy, eps=None):
        b, c, h, w, d = eps_theta.shape
        #loss = torch.norm(score * std + (gt_noisy - gt) / std, p=2, dim=(1, 2, 3, 4))**2
        #loss = loss / (h * w * d)
        #loss = torch.mean(loss)

        #score = -score / std
        #loss = torch.square(score * std + eps)
        #loss = torch.sum(loss, dim=(1,2,3,4)) / (h * w * d)
        #loss = torch.mean(loss) * 0.5

        loss = self.loss_fn(eps, eps_theta)
        return loss


if __name__ == '__main__':
    loss_fn = VPLoss()
