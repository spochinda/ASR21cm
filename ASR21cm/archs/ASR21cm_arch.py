# from basicsr.archs.arch_util import default_init_weights
import torch
import torch.nn as nn

from ASR21cm.archs.arch_utils import RDN, MLP_decoder, make_coord  # MambaIR, MambaIREncoder
from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class ArSSR(nn.Module):

    def __init__(
            self,  # hr_size,
            dim,
            decoder_depth,
            decoder_width,
            use_checkpoint=False,
            **kwargs):
        super(ArSSR, self).__init__()
        self.multi_gpu = kwargs.get('multi_gpu', False)
        self.device = kwargs.get('device', torch.device('cpu'))

        self.use_checkpoint = use_checkpoint
        # self.hr_size = hr_size
        self.feature_dim = int(dim * 2**4)
        # self.encoder = MambaIREncoder(
        #     inp_channels=1,
        #     out_channels=1,
        #     dim=dim,
        #     num_blocks=[4, 4, 4, 4],
        #     num_refinement_blocks=4,
        #     mlp_ratio=1.,
        #     bias=False,
        #     )
        # self.encoder = MambaIR(in_chans=1, embed_dim=self.feature_dim, depths=(6, 6, 6, 6), drop_rate=0., d_state=16, mlp_ratio=1., drop_path_rate=0.1, norm_layer=nn.LayerNorm, patch_norm=True, use_checkpoint=self.use_checkpoint, upscale=None, **kwargs)
        self.encoder = RDN(feature_dim=self.feature_dim, num_features=64, growth_rate=64, num_blocks=8, num_layers=3)

        self.decoder = MLP_decoder(in_dim=self.feature_dim + 3, out_dim=1, depth=decoder_depth, width=decoder_width, **kwargs)

    def forward(self, img_lr, xyz_hr):
        """
        :param img_lr: N×1×h×w×d
        :param xyz_hr: N×K×3
        Note that,
            N: batch size  (N in Equ. 3)
            K: coordinate sample size (K in Equ. 3)
            {h,w,d}: dimensional size of LR input image
        """
        # extract feature map from LR image
        if self.use_checkpoint:
            feature_map = torch.utils.checkpoint.checkpoint(self.encoder, img_lr, use_reentrant=False)
        else:
            feature_map = self.encoder(img_lr)

        # generate feature vector for coordinate through trilinear interpolation (Equ. 4 & Fig. 3).
        coords = xyz_hr.flip(-1)
        coords = coords.unsqueeze(1).unsqueeze(1)
        feature_vector = nn.functional.grid_sample(feature_map, coords, mode='bilinear', align_corners=False)[:, :, 0, 0, :].permute(0, 2, 1)
        # concatenate coordinate with feature vector
        feature_vector_and_xyz_hr = torch.cat([feature_vector, xyz_hr], dim=-1)  # N×K×(3+feature_dim)
        # estimate the voxel intensity at the coordinate by using decoder.
        N, K = xyz_hr.shape[:2]
        feature_vector_and_xyz_hr = feature_vector_and_xyz_hr.view(N * K, -1)
        img_sr = self.decoder(feature_vector_and_xyz_hr).view(N, K, -1)

        h = w = d = int(round(K**(1 / 3)))
        img_sr = img_sr.view(N, 1, h, w, d)
        return img_sr


if __name__ == '__main__':
    import numpy as np
    h = 93
    b = 2
    T21 = torch.rand(b, 1, h, h, h)
    scale_max = 3.0
    scale_min = 2.0
    scale_factor = np.random.rand(1)[0] * (scale_max - scale_min) + scale_min
    while (round(h / scale_factor) / 4) % 2 != 0:
        scale_factor = np.random.rand(1)[0] * (scale_max - scale_min) + scale_min
    h_lr = round(h / scale_factor)
    T21_lr = torch.nn.functional.interpolate(T21, size=h_lr, mode='trilinear')
    print('T21_lr shape: ', T21_lr.shape, 'T21 shape: ', T21.shape, 'scale_factor: ', scale_factor)

    xyz_hr = make_coord([h, h, h], ranges=None, flatten=False)
    xyz_hr = xyz_hr.view(1, -1, 3)
    xyz_hr = xyz_hr.repeat(b, 1, 1)
    network = ArSSR(dim=4, decoder_depth=4, decoder_width=8)
    output = network(img_lr=T21_lr, xyz_hr=xyz_hr)
    print(output.shape)
