import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from ASR21cm.archs.unet_utils import PositionalEmbedding, GroupNorm, Linear

from basicsr.utils.registry import ARCH_REGISTRY


# @ARCH_REGISTRY.register(suffix='basicsr')
class UNetDiscriminatorSN3D(nn.Module):
    """Defines a U-Net discriminator with spectral normalization (SN)

    It is used in Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    Arg:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_feat (int): Channel number of base intermediate features. Default: 64.
        skip_connection (bool): Whether to use skip connections between U-Net. Default: True.
    """

    def __init__(self, num_in_ch, num_feat=64, skip_connection=True):
        super(UNetDiscriminatorSN3D, self).__init__()
        self.skip_connection = skip_connection
        norm = spectral_norm
        # the first convolution
        self.conv0 = nn.Conv3d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1)
        # downsample
        self.conv1 = norm(nn.Conv3d(num_feat, num_feat * 2, 4, 2, 1, bias=False))
        self.conv2 = norm(nn.Conv3d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False))
        self.conv3 = norm(nn.Conv3d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False))
        # upsample
        self.conv4 = norm(nn.Conv3d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False))
        self.conv5 = norm(nn.Conv3d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
        self.conv6 = norm(nn.Conv3d(num_feat * 2, num_feat, 3, 1, 1, bias=False))
        # extra convolutions
        self.conv7 = norm(nn.Conv3d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv8 = norm(nn.Conv3d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv9 = nn.Conv3d(num_feat, 1, 3, 1, 1)

    def forward(self, x):
        # downsample
        x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2, inplace=True)
        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=True)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=True)
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2, inplace=True)

        # upsample
        x3 = F.interpolate(x3, scale_factor=2, mode='trilinear', align_corners=False)
        x4 = F.leaky_relu(self.conv4(x3), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x4 = x4 + x2
        x4 = F.interpolate(x4, scale_factor=2, mode='trilinear', align_corners=False)
        x5 = F.leaky_relu(self.conv5(x4), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x5 = x5 + x1
        x5 = F.interpolate(x5, scale_factor=2, mode='trilinear', align_corners=False)
        x6 = F.leaky_relu(self.conv6(x5), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x6 = x6 + x0

        # extra convolutions
        out = F.leaky_relu(self.conv7(x6), negative_slope=0.2, inplace=True)
        out = F.leaky_relu(self.conv8(out), negative_slope=0.2, inplace=True)
        out = self.conv9(out)

        return out


@ARCH_REGISTRY.register()
class VGGStyleDiscriminator3D(nn.Module):
    """VGG style discriminator with input size 128 x 128 or 256 x 256.

    It is used to train SRGAN, ESRGAN, and VideoGAN.

    Args:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_feat (int): Channel number of base intermediate features.Default: 64.
    """

    def __init__(self, num_in_ch, num_feat, input_size=128, redshift_embedding=False, conditional_cubes=False):
        super(VGGStyleDiscriminator3D, self).__init__()
        self.conditional_cubes = conditional_cubes
        assert not (self.conditional_cubes and num_in_ch != 3), \
            'If conditional cubes are used, num_in_ch must be 3.'
        self.input_size = input_size
        assert self.input_size == 64 or self.input_size == 128 or self.input_size == 256, \
            f'Input size must be 64, 128, or 256, but received {self.input_size}.'

        self.redshift_embedding = redshift_embedding
        if self.redshift_embedding:
            self.z_emb = PositionalEmbedding(num_channels=num_feat*4, endpoint=True)
            self.map_layer0 = nn.Linear(in_features=4*num_feat, out_features=4*num_feat) # , **init)
            self.map_layer1 = nn.Linear(in_features=4*num_feat, out_features=num_feat) # , **init)

        # add film
        if self.redshift_embedding:
            self.affine_0 = Linear(in_features=num_feat, out_features=num_in_ch * 2) #, **init), in_features=num_feat to match embedding size
            self.norm_0 = GroupNorm(num_channels=num_in_ch, eps=1e-5)
        self.conv0_0 = nn.Conv3d(num_in_ch, num_feat, 3, 1, 1, bias=True)
        self.conv0_1 = nn.Conv3d(num_feat, num_feat, 4, 2, 1, bias=False)
        self.bn0_1 = nn.BatchNorm3d(num_feat, affine=True)

        # add film
        if self.redshift_embedding:
            self.affine_1 = Linear(in_features=num_feat, out_features=num_feat * 2)
            self.norm_1 = GroupNorm(num_channels=num_feat, eps=1e-5)
        self.conv1_0 = nn.Conv3d(num_feat, num_feat * 2, 3, 1, 1, bias=False)
        self.bn1_0 = nn.BatchNorm3d(num_feat * 2, affine=True)
        self.conv1_1 = nn.Conv3d(num_feat * 2, num_feat * 2, 4, 2, 1, bias=False)
        self.bn1_1 = nn.BatchNorm3d(num_feat * 2, affine=True)

        # add film
        if self.redshift_embedding:
            self.affine_2 = Linear(in_features=num_feat, out_features=num_feat * 2 * 2)
            self.norm_2 = GroupNorm(num_channels=num_feat * 2, eps=1e-5)
        self.conv2_0 = nn.Conv3d(num_feat * 2, num_feat * 4, 3, 1, 1, bias=False)
        self.bn2_0 = nn.BatchNorm3d(num_feat * 4, affine=True)
        self.conv2_1 = nn.Conv3d(num_feat * 4, num_feat * 4, 4, 2, 1, bias=False)
        self.bn2_1 = nn.BatchNorm3d(num_feat * 4, affine=True)

        # add film
        if self.redshift_embedding:
            self.affine_3 = Linear(in_features=num_feat, out_features=num_feat * 4 * 2)
            self.norm_3 = GroupNorm(num_channels=num_feat * 4, eps=1e-5)
        self.conv3_0 = nn.Conv3d(num_feat * 4, num_feat * 8, 3, 1, 1, bias=False)
        self.bn3_0 = nn.BatchNorm3d(num_feat * 8, affine=True)
        self.conv3_1 = nn.Conv3d(num_feat * 8, num_feat * 8, 4, 2, 1, bias=False)
        self.bn3_1 = nn.BatchNorm3d(num_feat * 8, affine=True)

        if self.input_size >= 128:
            # add film
            if self.redshift_embedding:
                self.affine_4 = Linear(in_features=num_feat, out_features=num_feat * 8 * 2)
                self.norm_4 = GroupNorm(num_channels=num_feat * 8, eps=1e-5)
            self.conv4_0 = nn.Conv3d(num_feat * 8, num_feat * 8, 3, 1, 1, bias=False)
            self.bn4_0 = nn.BatchNorm3d(num_feat * 8, affine=True)
            self.conv4_1 = nn.Conv3d(num_feat * 8, num_feat * 8, 4, 2, 1, bias=False)
            self.bn4_1 = nn.BatchNorm3d(num_feat * 8, affine=True)

            if self.input_size >= 256:
                # add film
                if self.redshift_embedding:
                    self.affine_5 = Linear(in_features=num_feat, out_features=num_feat * 8 * 2)
                    self.norm_5 = GroupNorm(num_channels=num_feat * 8, eps=1e-5)
                self.conv5_0 = nn.Conv3d(num_feat * 8, num_feat * 8, 3, 1, 1, bias=False)
                self.bn5_0 = nn.BatchNorm3d(num_feat * 8, affine=True)
                self.conv5_1 = nn.Conv3d(num_feat * 8, num_feat * 8, 4, 2, 1, bias=False)
                self.bn5_1 = nn.BatchNorm3d(num_feat * 8, affine=True)

        self.linear1 = nn.Linear(num_feat * 8 * 4 * 4 * 4, 100)
        self.linear2 = nn.Linear(100, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x, z=None, **kwargs):
        assert x.size(2) == self.input_size, (f'Input size must be identical to input_size {self.input_size}, but received {x.size()}. Consider checking if network_d[input_size] == datasets[train][gt_size] in the config file.')

        delta = kwargs.get('delta', None)
        vbv = kwargs.get('vbv', None)
        assert not (self.conditional_cubes and (delta is None or vbv is None)), \
            'If conditional cubes are used, delta and vbv must be provided.'
        if delta is not None:
            x = torch.cat([x, delta], dim=1)  # concatenate delta to input
        if vbv is not None:
            x = torch.cat([x, vbv], dim=1)

        if self.redshift_embedding:
            assert z is not None, "Redshift embedding is enabled, but no redshift value provided."
            emb = self.z_emb(z)
            emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape)  # swap sin/cos
            # class labels (astro parameters) to be added...
            emb = torch.nn.functional.silu(self.map_layer0(emb))
            emb = torch.nn.functional.silu(self.map_layer1(emb))
        else:
            assert z is None, "Redshift value provided, but redshift embedding is not enabled."
            emb = None

        if emb is not None:
            params = self.affine_0(emb).unsqueeze(2).unsqueeze(3).unsqueeze(4).to(x.dtype)
            scale, shift = params.chunk(chunks=2, dim=1)
            x = torch.nn.functional.silu(torch.addcmul(shift, self.norm_0(x), scale + 1))
        feat = self.lrelu(self.conv0_0(x))
        feat = self.lrelu(self.bn0_1(self.conv0_1(feat)))  # output spatial size: /2

        if emb is not None:
            params = self.affine_1(emb).unsqueeze(2).unsqueeze(3).unsqueeze(4).to(feat.dtype)
            scale, shift = params.chunk(chunks=2, dim=1)
            feat = torch.nn.functional.silu(torch.addcmul(shift, self.norm_1(feat), scale + 1))
        feat = self.lrelu(self.bn1_0(self.conv1_0(feat)))
        feat = self.lrelu(self.bn1_1(self.conv1_1(feat)))  # output spatial size: /4

        if emb is not None:
            params = self.affine_2(emb).unsqueeze(2).unsqueeze(3).unsqueeze(4).to(feat.dtype)
            scale, shift = params.chunk(chunks=2, dim=1)
            feat = torch.nn.functional.silu(torch.addcmul(shift, self.norm_2(feat), scale + 1))
        feat = self.lrelu(self.bn2_0(self.conv2_0(feat)))
        feat = self.lrelu(self.bn2_1(self.conv2_1(feat)))  # output spatial size: /8

        if emb is not None:
            params = self.affine_3(emb).unsqueeze(2).unsqueeze(3).unsqueeze(4).to(feat.dtype)
            scale, shift = params.chunk(chunks=2, dim=1)
            feat = torch.nn.functional.silu(torch.addcmul(shift, self.norm_3(feat), scale + 1))
        feat = self.lrelu(self.bn3_0(self.conv3_0(feat)))
        feat = self.lrelu(self.bn3_1(self.conv3_1(feat)))  # output spatial size: /16

        if self.input_size >= 128:
            if emb is not None:
                params = self.affine_4(emb).unsqueeze(2).unsqueeze(3).unsqueeze(4).to(feat.dtype)
                scale, shift = params.chunk(chunks=2, dim=1)
                feat = torch.nn.functional.silu(torch.addcmul(shift, self.norm_4(feat), scale + 1))
            feat = self.lrelu(self.bn4_0(self.conv4_0(feat)))
            feat = self.lrelu(self.bn4_1(self.conv4_1(feat)))  # output spatial size: /32

            if self.input_size == 256:
                if emb is not None:
                    params = self.affine_5(emb).unsqueeze(2).unsqueeze(3).unsqueeze(4).to(feat.dtype)
                    scale, shift = params.chunk(chunks=2, dim=1)
                    feat = torch.nn.functional.silu(torch.addcmul(shift, self.norm_5(feat), scale + 1))
                feat = self.lrelu(self.bn5_0(self.conv5_0(feat)))
                feat = self.lrelu(self.bn5_1(self.conv5_1(feat)))  # output spatial size: / 64

        # spatial size: (4, 4)
        feat = feat.view(feat.size(0), -1)
        feat = self.lrelu(self.linear1(feat))
        out = self.linear2(feat)
        return out


if __name__ == '__main__':
    # net = UNetDiscriminatorSN3D(1, 64, True)
    # print(net)
    # x = torch.randn(1, 1, 64, 64, 64)  # (batch_size, channels, depth, height, width)
    # y = net(x)
    # print(y.shape)
    network_d = {
        'num_in_ch': 3,
        'num_feat': 32,
        'input_size': 64,
        'redshift_embedding': True,
        'conditional_cubes': True
    }
    net = VGGStyleDiscriminator3D(**network_d)
    print(net)
    b,  c, h, w = 1, 1, network_d['input_size'], network_d['input_size']
    x = torch.randn(b, c, h, w, h)  # (batch_size, channels, depth, height, width)
    z = torch.ones(b)
    delta = torch.randn(b, 1, h, w, h)  # Example delta tensor
    vbv = torch.randn(b, 1, h, w, h)  # Example vbv tensor
    y = net(x, z=z, delta=delta, vbv=vbv)
    print(y.shape)
