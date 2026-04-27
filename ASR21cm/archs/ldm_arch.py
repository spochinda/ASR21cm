import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from basicsr.utils.registry import ARCH_REGISTRY


def Normalize(num_channels):
    # use the largest divisor of num_channels that is <= 32
    num_groups = next(g for g in range(min(32, num_channels), 0, -1) if num_channels % g == 0)
    return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels, eps=1e-6, affine=True)


@ARCH_REGISTRY.register()
class NLayerDiscriminator3D(nn.Module):
    """3-D PatchGAN discriminator (adapted from taming-transformers NLayerDiscriminator).

    Produces a spatial map of real/fake logits rather than a single scalar,
    so each output element covers a receptive-field 'patch' of the input.
    Uses GroupNorm (via Normalize()) instead of BatchNorm so it is stable
    at batch_size=1.

    Args:
        input_nc (int): Number of input channels.
        ndf      (int): Base number of discriminator filters.
        n_layers (int): Number of strided Conv3d downsampling layers.
    """

    def __init__(self, input_nc=1, ndf=64, n_layers=3, use_spectral_norm=False):
        super().__init__()

        kw = 4
        padw = 1
        wrap = spectral_norm if use_spectral_norm else (lambda x: x)

        def conv_block(in_ch, out_ch, stride):
            layers = [wrap(nn.Conv3d(in_ch, out_ch, kernel_size=kw, stride=stride, padding=padw, bias=use_spectral_norm))]
            if not use_spectral_norm:
                layers.append(Normalize(out_ch))
            layers.append(nn.LeakyReLU(0.2, True))
            return layers

        sequence = [
            wrap(nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw)),
            nn.LeakyReLU(0.2, True),
        ]

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += conv_block(ndf * nf_mult_prev, ndf * nf_mult, stride=2)

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += conv_block(ndf * nf_mult_prev, ndf * nf_mult, stride=1)

        sequence += [wrap(nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw))]
        self.main = nn.Sequential(*sequence)

    def forward(self, x):
        return self.main(x)


class Upsample(nn.Module):

    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode='nearest')
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):

    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode='constant', value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool3d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):

    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False, dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = F.silu(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(F.silu(temb)).view(temb.shape[0], -1, *([1] * (h.ndim - 2)))

        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class AttnBlock(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.norm = Normalize(in_channels)
        self.q = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv3d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        h_ = self.norm(x)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, *spatial = q.shape
        # reshape to (b, 1, n, c) for scaled_dot_product_attention
        q = q.flatten(2).unsqueeze(1).transpose(2, 3)
        k = k.flatten(2).unsqueeze(1).transpose(2, 3)
        v = v.flatten(2).unsqueeze(1).transpose(2, 3)

        h_ = F.scaled_dot_product_attention(q, k, v)
        h_ = h_.transpose(2, 3).reshape(b, c, *spatial)

        return x + self.proj_out(h_)


def make_attn(in_channels, attn_type='vanilla'):
    assert attn_type in ['vanilla', 'none'], f'attn_type {attn_type} unknown'
    if attn_type == 'vanilla':
        return AttnBlock(in_channels)
    else:
        return nn.Identity(in_channels)


class Encoder(nn.Module):

    def __init__(self, *, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks, attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels, resolution, z_channels, double_z=True, use_linear_attn=False, attn_type='vanilla', **ignore_kwargs):
        super().__init__()
        if use_linear_attn:
            attn_type = 'linear'
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = torch.nn.Conv3d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        curr_res = resolution
        in_ch_mult = (1, ) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv3d(block_in, 2 * z_channels if double_z else z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):

    def __init__(self, *, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks, attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels, resolution, z_channels, give_pre_end=False, tanh_out=False, use_linear_attn=False, attn_type='vanilla', **ignorekwargs):
        super().__init__()
        if use_linear_attn:
            attn_type = 'linear'
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out

        # start at the bottleneck channel count and current (lowest) resolution
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2**(self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res, curr_res)
        print('Working with z of shape {} = {} dimensions.'.format(self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv3d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv3d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        # assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        return h


@ARCH_REGISTRY.register()
class AutoencoderKL(nn.Module):

    def __init__(self, autoencoder_opt):
        super().__init__()
        embed_dim = autoencoder_opt['embed_dim']
        self.encoder = Encoder(**autoencoder_opt)
        self.decoder = Decoder(**autoencoder_opt)
        assert autoencoder_opt['double_z']
        self.quant_conv = torch.nn.Conv3d(2 * autoencoder_opt['z_channels'], 2 * embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv3d(embed_dim, autoencoder_opt['z_channels'], 1)
        self.embed_dim = embed_dim

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        # Return the raw moments tensor instead of the DiagonalGaussianDistribution
        # object so that DataParallel can gather outputs across GPUs.
        return dec, posterior.parameters

    def get_last_layer(self):
        return self.decoder.conv_out.weight


class DiagonalGaussianDistribution(object):

    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        return self.mean + self.std * torch.randn_like(self.mean)

    def kl(self, other=None):
        dims = list(range(1, self.mean.ndim))
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.sum(torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar, dim=dims)
            else:
                return 0.5 * torch.sum(torch.pow(self.mean - other.mean, 2) / other.var + self.var / other.var - 1.0 - self.logvar + other.logvar, dim=dims)

    def nll(self, sample, dims=None):
        if self.deterministic:
            return torch.Tensor([0.])
        if dims is None:
            dims = list(range(1, self.mean.ndim))
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var, dim=dims)

    def mode(self):
        return self.mean


if __name__ == '__main__':
    ddconfig = dict(
        double_z=True,
        z_channels=4,
        embed_dim=4,
        resolution=128,
        in_channels=1,
        out_ch=1,
        ch=8,
        ch_mult=(1, 2, 4, 4),
        num_res_blocks=2,
        attn_resolutions=[],
        dropout=0.0,
    )

    model = AutoencoderKL(autoencoder_opt=ddconfig)
    model.eval()

    shapes = {}

    def hook(name):

        def fn(module, input, output):
            shapes[name] = tuple(output.shape)

        return fn

    # register hooks on key submodules
    model.encoder.conv_in.register_forward_hook(hook('enc conv_in'))
    for i, down in enumerate(model.encoder.down):
        attn_tag = ' [attn]' if len(down.attn) > 0 else ''
        down.block[-1].register_forward_hook(hook(f'enc down[{i}]{attn_tag}'))
        if len(down.attn) > 0:
            down.attn[-1].register_forward_hook(hook(f'enc down[{i}] attn out'))
        if hasattr(down, 'downsample'):
            down.downsample.register_forward_hook(hook(f'enc downsample[{i}]'))
    model.encoder.mid.block_2.register_forward_hook(hook('enc mid [attn]'))
    model.encoder.mid.attn_1.register_forward_hook(hook('enc mid attn out'))
    model.encoder.conv_out.register_forward_hook(hook('enc conv_out (pre quant)'))
    model.quant_conv.register_forward_hook(hook('quant_conv (moments)'))
    model.post_quant_conv.register_forward_hook(hook('post_quant_conv'))
    model.decoder.mid.block_2.register_forward_hook(hook('dec mid [attn]'))
    model.decoder.mid.attn_1.register_forward_hook(hook('dec mid attn out'))
    for i, up in enumerate(model.decoder.up):
        attn_tag = ' [attn]' if len(up.attn) > 0 else ''
        up.block[-1].register_forward_hook(hook(f'dec up[{i}]{attn_tag}'))
        if len(up.attn) > 0:
            up.attn[-1].register_forward_hook(hook(f'dec up[{i}] attn out'))
        if hasattr(up, 'upsample'):
            up.upsample.register_forward_hook(hook(f'dec upsample[{i}]'))
    model.decoder.conv_out.register_forward_hook(hook('dec conv_out'))

    x = torch.randn(2, 1, 128, 128, 128)
    with torch.no_grad():
        posterior = model.encode(x)
        z = posterior.sample()
        dec = model.decode(z)

    print(f"{'input':<30} {tuple(x.shape)}")
    for name, shape in shapes.items():
        print(f'  {name:<28} {shape}')
    print(f"{'z (sampled)':<30} {tuple(z.shape)}")
    print(f"  {'--- decode ---':<28}")
    print(f"{'output':<30} {tuple(dec.shape)}")
    assert dec.shape == x.shape, 'Output shape mismatch'
    print('\nAutoencoderKL forward pass OK')
