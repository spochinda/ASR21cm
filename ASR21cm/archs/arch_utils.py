# The Code Implementatio of MambaIR model for Real Image Denoising task
import collections.abc
import math
import torch
import torch.nn as nn
import warnings
from einops import rearrange, repeat
from functools import partial
# from pdb import set_trace as stx
from timm.layers import DropPath  # , trunc_normal_
from typing import Callable

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn  # selective_scan_ref
    test_mode = False
except Exception as e:
    print(e)
    print('Selective scan not available, using reference implementation and enabling test mode')
    test_mode = True
    selective_scan_fn = None

NEG_INF = -1000000


def _ntuple(n):

    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple([x] * n)

    return parse


to_3tuple = _ntuple(3)

# leftover from medsegmamba (below) #


class VSSLayer3D(nn.Module):
    """ A basic layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self,
        dim,
        depth,
        attn_drop=0.,
        mlp_drop=0.,
        drop_path=0.,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
        d_state=64,
        version='v5',
        expansion_factor=1,
        scan_type='scan',
        orientation_order=None,
        size=12,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint
        if version is None:
            print('Vanilla VSS')
            self.blocks = nn.ModuleList([VSSBlock3D(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
                expansion_factor=expansion_factor,
            ) for i in range(depth)])
        else:
            raise Exception('define a valid VSS version')
        if True:

            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ['out_proj.weight']:
                        p = p.clone().detach_()  # fake init, just to keep the seed ....
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))

            self.apply(_init_weights)

        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


# for MambaIR models (below) ###
class SS3D(nn.Module):  # for the original Vanilla VSS block, worse as described in VMamba paper

    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=3,
        expand=1,
        dt_rank='auto',
        dt_min=0.001,
        dt_max=0.1,
        dt_init='random',
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        conv_bias=True,
        bias=False,
        device='cpu',
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.d_model = d_model  # channel dim, 512 or 1024, gets expanded
        self.d_state = d_state

        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == 'auto' else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv3d = nn.Conv3d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=8, N, inner) = (K=8, new_c = self.dt_rank + self.d_state * 2, C)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=8, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=8, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init='random', dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == 'constant':
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == 'random':
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=8, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            'n -> d n',
            d=d_inner,
        ).contiguous()
        # ('A', A.shape)
        A_log = torch.log(A)  # Keep A_log in fp32

        if copies > 1:
            A_log = repeat(A_log, 'd n -> r d n', r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=8, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, 'n1 -> r n1', r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        # 0,1, 2, 3, 4
        B, C, H, W, D = x.shape
        L = H * W * D
        K = 8

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L), torch.transpose(x, dim0=2, dim1=4).contiguous().view(B, -1, L), torch.transpose(x, dim0=3, dim1=4).contiguous().view(B, -1, L)], dim=1).view(B, 4, -1, L)
        # hwd, whd, dwh, hdw; reversed
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, c, l)
        # hwd b, 1, c, l >
        # whd b, 1, c, l >
        # dwh b, 1, c, l >
        # hdw b, 1, c, l >
        # hwd reversed l
        # whd reversed l
        # dwh reversed l
        # hdw reversed l

        x_dbl = torch.einsum('b k d l, k c d -> b k c l', xs.view(B, K, -1, L), self.x_proj_weight)

        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)

        dts = torch.einsum('b k r l, k d r -> b k d l', dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)

        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)

        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)

        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs,
            dts,
            As,
            Bs,
            Cs,
            Ds,
            z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L) if not test_mode else torch.randn(
            B, K, self.d_model, L, device=x.device)  # B, K, channelsize, L
        assert out_y.dtype == torch.float

        # hwd b, 1, c, l >
        # whd b, 1, c, l >
        # dwh b, 1, c, l >
        # hdw b, 1, c, l >
        # hwd reversed l
        # whd reversed l
        # dwh reversed l
        # hdw reversed l

        # revert back to all hwd forward l

        # out1 = out_y[:,0,:,:]
        out2 = torch.transpose(out_y[:, 1].view(B, -1, W, H, D), dim0=2, dim1=3).contiguous().view(B, -1, L)
        out3 = torch.transpose(out_y[:, 2].view(B, -1, W, H, D), dim0=2, dim1=4).contiguous().view(B, -1, L)
        out4 = torch.transpose(out_y[:, 3].view(B, -1, W, H, D), dim0=3, dim1=4).contiguous().view(B, -1, L)

        out5 = torch.flip(out_y[:, 0], dims=[-1]).view(B, -1, L)
        out6 = torch.flip(out2, dims=[-1]).view(B, -1, L)
        out7 = torch.flip(out3, dims=[-1]).view(B, -1, L)
        out8 = torch.flip(out4, dims=[-1]).view(B, -1, L)

        return out_y[:, 0], out2, out3, out4, out5, out6, out7, out8

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, D, C = x.shape  # !!!
        # d_model = C

        xz = self.in_proj(x)  # (b, h, w, d, d_model) -> (b, h, w, d, d_inner * 2)
        x, z = xz.chunk(2, dim=-1)  # (b, h, w, d, d_inner), z for the multiplicative path

        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = self.act(self.conv3d(x))  # (b, d, h, w)

        y1, y2, y3, y4, y5, y6, y7, y8 = self.forward_core(x)  # 1 1024 1728

        assert y1.dtype == torch.float32

        y = y1 + y2 + y3 + y4 + y5 + y6 + y7 + y8

        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, D, -1)  # bcl > blc > bhwdc
        y = self.out_norm(y)
        y = y * nn.functional.silu(z)  # multiplicative path, ignored in v2 because ssm is inherently selective, described in VMamba

        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)

        return out


class VSSBlock3D(nn.Module):

    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            expansion_factor=1,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS3D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, expand=expansion_factor, **kwargs)
        self.drop_path = DropPath(drop_path)

        self.skip_scale1 = nn.Parameter(torch.ones(hidden_dim))  # added
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))
        self.conv_blk = CAB(hidden_dim)
        self.ln_2 = nn.LayerNorm(hidden_dim)

    def forward(self, input: torch.Tensor, x_size):
        # x = self.ln_1(input)
        # x = input*self.skip_scale1+ self.drop_path(self.self_attention(x))

        # x_ln_conv = self.conv_blk(self.ln_2(x)) #check shape
        # x = x*self.skip_scale2 + x_ln_conv

        # x [B,HWD,C]
        B, L, C = input.shape
        input = input.view(B, *x_size, C).contiguous()  # [B,H,W,D,C]
        x = self.ln_1(input)
        x = input * self.skip_scale1 + self.drop_path(self.self_attention(x))
        x = x * self.skip_scale2 + self.conv_blk(self.ln_2(x).permute(0, 4, 1, 2, 3).contiguous()).permute(0, 2, 3, 4, 1).contiguous()
        x = x.view(B, -1, C).contiguous()
        return x


class BasicLayer(nn.Module):
    """ The Basic MambaIR Layer in one Residual State Space Group
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, depth, drop_path=0., d_state=16, mlp_ratio=2., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False, is_light_sr=False):

        super().__init__()
        self.dim = dim
        # self.input_resolution = input_resolution
        self.depth = depth
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(VSSBlock3D(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                d_state=d_state,
                expansion_factor=self.mlp_ratio,
                is_light_sr=is_light_sr,
            ))  # input_resolution=input_resolution, ))

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(  # input_resolution,
                dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size):
        for i, blk in enumerate(self.blocks):
            if self.use_checkpoint:
                x = torch.utils.checkpoint.checkpoint(blk, x, x_size, use_reentrant=False)
            else:
                x = blk(x, x_size)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class ResidualGroup(nn.Module):
    """Residual State Space Group (RSSG).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
    """

    def __init__(
            self,
            dim,  # input_resolution,
            depth,
            d_state=16,
            mlp_ratio=4.,
            drop_path=0.,
            norm_layer=nn.LayerNorm,
            downsample=None,  # img_size=None, patch_size=None,
            use_checkpoint=False,
            is_light_sr=False):
        super(ResidualGroup, self).__init__()

        self.dim = dim
        # self.input_resolution = input_resolution  # [64, 64]

        self.residual_group = BasicLayer(
            dim=dim,  # input_resolution=input_resolution,
            depth=depth,
            d_state=d_state,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint,
            is_light_sr=is_light_sr)

        # build the last conv layer in each residual state space group

        self.conv = nn.Conv3d(dim, dim, 3, 1, 1)

        self.patch_embed = PatchEmbed(  # img_size=img_size, patch_size=patch_size,
            in_chans=0, embed_dim=dim, norm_layer=None)

        self.patch_unembed = PatchUnEmbed(  # img_size=img_size, patch_size=patch_size,
            in_chans=0, embed_dim=dim, norm_layer=None)

    def forward(self, x, x_size):
        return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size), x_size))) + x


class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=2):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(nn.AdaptiveAvgPool3d(1), nn.Conv3d(num_feat, num_feat // squeeze_factor, 1, padding=0), nn.ReLU(inplace=True), nn.Conv3d(num_feat // squeeze_factor, num_feat, 1, padding=0), nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y


class CAB(nn.Module):

    def __init__(self, num_feat, compress_ratio=3, squeeze_factor=2):
        super(CAB, self).__init__()
        self.cab = nn.Sequential(nn.Conv3d(num_feat, num_feat // compress_ratio, 3, 1, 1), nn.GELU(), nn.Conv3d(num_feat // compress_ratio, num_feat, 3, 1, 1), ChannelAttention(num_feat, squeeze_factor))

    def forward(self, x):
        return self.cab(x)


class VSSBlock(nn.Module):

    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            expand: float = 1.,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS3D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, expand=expand, **kwargs)  # self.self_attention = SS2D(d_model=hidden_dim, d_state=d_state,expand=expand,dropout=attn_drop_rate, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.skip_scale = nn.Parameter(torch.ones(hidden_dim))
        self.conv_blk = CAB(hidden_dim)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, input, x_size):
        # x [B,HWD,C]
        B, L, C = input.shape
        input = input.view(B, *x_size, C).contiguous()  # [B,H,W,C]
        x = self.ln_1(input)
        x = input * self.skip_scale + self.drop_path(self.self_attention(x))
        x = x * self.skip_scale2 + self.conv_blk(self.ln_2(x).permute(0, 4, 1, 2, 3).contiguous()).permute(0, 2, 3, 4, 1).contiguous()
        x = x.view(B, -1, C).contiguous()
        return x


class PatchEmbed(nn.Module):
    r""" transfer 2D feature map into 1D token sequence

    Args:
        img_size (int): Image size.  Default: None.
        patch_size (int): Patch token size. Default: None.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(
            self,  # img_size=224, patch_size=4,
            in_chans=3,
            embed_dim=96,
            norm_layer=None):
        super().__init__()
        # img_size = to_3tuple(img_size)
        # patch_size = to_3tuple(patch_size)
        # patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2]]
        # self.img_size = img_size
        # self.patch_size = patch_size
        # self.patches_resolution = patches_resolution
        # self.num_patches = patches_resolution[0] * patches_resolution[1] * patches_resolution[2]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # b Ph*Pw c
        if self.norm is not None:
            x = self.norm(x)
        return x

    # def flops(self):
    #    flops = 0
    #    h, w = self.img_size
    #    if self.norm is not None:
    #        flops += h * w * self.embed_dim
    #    return flops


class PatchUnEmbed(nn.Module):
    r""" return 2D feature map from 1D token sequence

    Args:
        img_size (int): Image size.  Default: None.
        patch_size (int): Patch token size. Default: None.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(
            self,  # img_size=224, patch_size=4,
            in_chans=3,
            embed_dim=96,
            norm_layer=None):
        super().__init__()
        # img_size = to_3tuple(img_size)
        # patch_size = to_3tuple(patch_size)
        # patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2]]
        # self.img_size = img_size
        # self.patch_size = patch_size
        # self.patches_resolution = patches_resolution
        # self.num_patches = patches_resolution[0] * patches_resolution[1] * patches_resolution[2]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.embed_dim, x_size[0], x_size[1], x_size[2])  # b Ph*Pw*Pd c
        return x

    #  def flops(self):
    #    flops = 0
    #    return flops


# Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):

    def __init__(self, in_c=1, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv3d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        x = rearrange(x, 'b c h w d -> b (h w d) c').contiguous()
        return x


# Resizing modules
class PixelShuffle3d(nn.Module):

    def __init__(self, upscale_factor=None):
        super().__init__()

        if upscale_factor is None:
            raise TypeError('__init__() missing 1 required positional argument: \'upscale_factor\'')

        self.upscale_factor = upscale_factor

    def forward(self, x):
        if x.ndim < 3:
            raise RuntimeError(f'pixel_shuffle expects input to have at least 3 dimensions, but got input with {x.ndim} dimension(s)')
        elif x.shape[-4] % self.upscale_factor**3 != 0:
            raise RuntimeError(f'pixel_shuffle expects its input\'s \'channel\' dimension to be divisible by the cube of upscale_factor, but input.size(-4)={x.shape[-4]} is not divisible by {self.upscale_factor**3}')

        channels, in_depth, in_height, in_width = x.shape[-4:]
        nOut = channels // self.upscale_factor**3

        out_depth = in_depth * self.upscale_factor
        out_height = in_height * self.upscale_factor
        out_width = in_width * self.upscale_factor

        input_view = x.contiguous().view(*x.shape[:-4], nOut, self.upscale_factor, self.upscale_factor, self.upscale_factor, in_depth, in_height, in_width)

        axes = torch.arange(input_view.ndim)[:-6].tolist() + [-3, -6, -2, -5, -1, -4]
        output = input_view.permute(axes).contiguous()

        return output.view(*x.shape[:-4], nOut, out_depth, out_height, out_width)


class PixelUnshuffle3d(nn.Module):

    def __init__(self, upscale_factor=None):
        super().__init__()

        if upscale_factor is None:
            raise TypeError('__init__() missing 1 required positional argument: \'upscale_factor\'')

        self.upscale_factor = upscale_factor

    def forward(self, x):
        if x.ndim < 3:
            raise RuntimeError(f'pixel_unshuffle expects input to have at least 3 dimensions, but got input with {x.ndim} dimension(s)')
        elif x.shape[-3] % self.upscale_factor != 0:
            raise RuntimeError(f'pixel_unshuffle expects depth to be divisible by downscale_factor, but input.size(-3)={x.shape[-3]} is not divisible by {self.upscale_factor}')
        elif x.shape[-2] % self.upscale_factor != 0:
            raise RuntimeError(f'pixel_unshuffle expects height to be divisible by downscale_factor, but input.size(-2)={x.shape[-2]} is not divisible by {self.upscale_factor}')
        elif x.shape[-1] % self.upscale_factor != 0:
            raise RuntimeError(f'pixel_unshuffle expects width to be divisible by downscale_factor, but input.size(-1)={x.shape[-1]} is not divisible by {self.upscale_factor}')

        channels, in_depth, in_height, in_width = x.shape[-4:]

        out_depth = in_depth // self.upscale_factor
        out_height = in_height // self.upscale_factor
        out_width = in_width // self.upscale_factor
        nOut = channels * self.upscale_factor**3

        input_view = x.contiguous().view(*x.shape[:-4], channels, out_depth, self.upscale_factor, out_height, self.upscale_factor, out_width, self.upscale_factor)

        axes = torch.arange(input_view.ndim)[:-6].tolist() + [-5, -3, -1, -6, -4, -2]
        output = input_view.permute(axes).contiguous()

        return output.view(*x.shape[:-4], nOut, out_depth, out_height, out_width)


class Downsample(nn.Module):

    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv3d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False), PixelUnshuffle3d(2))

    def forward(self, x, H, W, D):
        x = rearrange(x, 'b (h w d) c -> b c h w d', h=H, w=W, d=D).contiguous()
        x = self.body(x)
        x = rearrange(x, 'b c h w d -> b (h w d) c').contiguous()
        return x


class Upsample(nn.Module):

    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv3d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False), PixelShuffle3d(2))

    def forward(self, x, H, W, D):
        x = rearrange(x, 'b (h w d) c -> b c h w d', h=H, w=W, d=D).contiguous()
        x = self.body(x)
        x = rearrange(x, 'b c h w d -> b (h w d) c').contiguous()
        return x


class MambaIRUNet(nn.Module):

    def __init__(
            self,
            inp_channels=3,
            out_channels=3,
            dim=48,
            num_blocks=[4, 6, 6, 8],
            mlp_ratio=2.,
            num_refinement_blocks=4,
            drop_path_rate=0.,
            bias=False,
            dual_pixel_task=False  # True for dual-pixel defocus deblurring only. Also set inp_channels=6
    ):

        super(MambaIRUNet, self).__init__()
        self.mlp_ratio = mlp_ratio
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        base_d_state = 4
        self.encoder_level1 = nn.ModuleList([VSSBlock(
            hidden_dim=dim,
            drop_path=drop_path_rate,
            norm_layer=nn.LayerNorm,
            attn_drop_rate=0,
            expand=self.mlp_ratio,
            d_state=base_d_state * 2**2,
        ) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim * 2**0)  # From Level 1 to Level 2
        self.encoder_level2 = nn.ModuleList([VSSBlock(
            hidden_dim=int(dim * 2**2),
            drop_path=drop_path_rate,
            norm_layer=nn.LayerNorm,
            attn_drop_rate=0,
            expand=self.mlp_ratio,
            d_state=int(base_d_state * 2**2),
        ) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2**2))  # From Level 2 to Level 3
        self.encoder_level3 = nn.ModuleList([VSSBlock(
            hidden_dim=int(dim * 2**4),
            drop_path=drop_path_rate,
            norm_layer=nn.LayerNorm,
            attn_drop_rate=0,
            expand=self.mlp_ratio,
            d_state=int(base_d_state * 2**2),
        ) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim * 2**4))  # From Level 3 to Level 4
        self.latent = nn.ModuleList([VSSBlock(
            hidden_dim=int(dim * 2**6),
            drop_path=drop_path_rate,
            norm_layer=nn.LayerNorm,
            attn_drop_rate=0,
            expand=self.mlp_ratio,
            d_state=int(base_d_state / 2 * 2**3),
        ) for i in range(num_blocks[3])])

        self.up4_3 = Upsample(int(dim * 2**6))  # From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv3d(int(dim * 2**5), int(dim * 2**4), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.ModuleList([VSSBlock(
            hidden_dim=int(dim * 2**4),
            drop_path=drop_path_rate,
            norm_layer=nn.LayerNorm,
            attn_drop_rate=0,
            expand=self.mlp_ratio,
            d_state=int(base_d_state * 2**2),
        ) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2**4))  # From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv3d(int(dim * 2**3), int(dim * 2**2), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.ModuleList([VSSBlock(
            hidden_dim=int(dim * 2**2),
            drop_path=drop_path_rate,
            norm_layer=nn.LayerNorm,
            attn_drop_rate=0,
            expand=self.mlp_ratio,
            d_state=int(base_d_state * 2**2),
        ) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2**2))  # From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.ModuleList([VSSBlock(
            hidden_dim=int(dim * 2**1),
            drop_path=drop_path_rate,
            norm_layer=nn.LayerNorm,
            attn_drop_rate=0,
            expand=self.mlp_ratio,
            d_state=int(base_d_state * 2**2),
        ) for i in range(num_blocks[0])])

        self.refinement = nn.ModuleList([VSSBlock(
            hidden_dim=int(dim * 2**1),
            drop_path=drop_path_rate,
            norm_layer=nn.LayerNorm,
            attn_drop_rate=0,
            expand=self.mlp_ratio,
            d_state=int(base_d_state * 2**2),
        ) for i in range(num_refinement_blocks)])

        # For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv3d(dim, int(dim * 2**1), kernel_size=1, bias=bias)
        # ###########################

        self.output = nn.Conv3d(int(dim * 2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):

        _, _, H, W, D = inp_img.shape
        inp_enc_level1 = self.patch_embed(inp_img)  # b,hw,c
        out_enc_level1 = inp_enc_level1
        for layer in self.encoder_level1:
            out_enc_level1 = layer(out_enc_level1, [H, W, D])

        inp_enc_level2 = self.down1_2(out_enc_level1, H, W, D)  # b, hwd//8, 2c
        out_enc_level2 = inp_enc_level2
        for layer in self.encoder_level2:
            out_enc_level2 = layer(out_enc_level2, [H // 2, W // 2, D // 2])

        inp_enc_level3 = self.down2_3(out_enc_level2, H // 2, W // 2, D // 2)  # b, hwd//16, 4c
        out_enc_level3 = inp_enc_level3
        for layer in self.encoder_level3:
            out_enc_level3 = layer(out_enc_level3, [H // 4, W // 4, D // 4])

        inp_enc_level4 = self.down3_4(out_enc_level3, H // 4, W // 4, D // 4)  # b, hw//64, 8c
        latent = inp_enc_level4
        for layer in self.latent:
            latent = layer(latent, [H // 8, W // 8, D // 8])

        inp_dec_level3 = self.up4_3(latent, H // 8, W // 8, D // 8)  # b, hwd//64, 4c
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 2)
        inp_dec_level3 = rearrange(inp_dec_level3, 'b (h w d) c -> b c h d w', h=H // 4, w=W // 4, d=D // 4).contiguous()
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        inp_dec_level3 = rearrange(inp_dec_level3, 'b c h w d -> b (h w d) c').contiguous()  # b, hw//16, 4c
        out_dec_level3 = inp_dec_level3
        for layer in self.decoder_level3:
            out_dec_level3 = layer(out_dec_level3, [H // 4, W // 4, D // 4])

        inp_dec_level2 = self.up3_2(out_dec_level3, H // 4, W // 4, D // 4)  # b, hw//4, 2c
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 2)
        inp_dec_level2 = rearrange(inp_dec_level2, 'b (h w d) c -> b c h d w', h=H // 2, w=W // 2, d=D // 2).contiguous()
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        inp_dec_level2 = rearrange(inp_dec_level2, 'b c h w d -> b (h w d) c').contiguous()  # b, hw//4, 2c
        out_dec_level2 = inp_dec_level2
        for layer in self.decoder_level2:
            out_dec_level2 = layer(out_dec_level2, [H // 2, W // 2, D // 2])

        inp_dec_level1 = self.up2_1(out_dec_level2, H // 2, W // 2, D // 2)  # b, hw, c
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 2)
        out_dec_level1 = inp_dec_level1
        for layer in self.decoder_level1:
            out_dec_level1 = layer(out_dec_level1, [H, W, D])

        for layer in self.refinement:
            out_dec_level1 = layer(out_dec_level1, [H, W, D])

        out_dec_level1 = rearrange(out_dec_level1, 'b (h w d) c -> b c h w d', h=H, w=W, d=D).contiguous()

        # For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        # ##########################
        else:
            out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1


class MambaIREncoder(nn.Module):

    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        dim=4,
        num_blocks=[4, 6, 6, 8],
        mlp_ratio=2.,
        num_refinement_blocks=4,
        drop_path_rate=0.,
        bias=False,
    ):

        super(MambaIREncoder, self).__init__()
        self.mlp_ratio = mlp_ratio
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        base_d_state = 4
        self.encoder_level1 = nn.ModuleList([VSSBlock(
            hidden_dim=dim,
            drop_path=drop_path_rate,
            norm_layer=nn.LayerNorm,
            attn_drop_rate=0,
            expand=self.mlp_ratio,
            d_state=base_d_state * 2**2,
        ) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim * 2**0)  # From Level 1 to Level 2
        self.encoder_level2 = nn.ModuleList([VSSBlock(
            hidden_dim=int(dim * 2**2),
            drop_path=drop_path_rate,
            norm_layer=nn.LayerNorm,
            attn_drop_rate=0,
            expand=self.mlp_ratio,
            d_state=int(base_d_state * 2**2),
        ) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2**2))  # From Level 2 to Level 3
        self.encoder_level3 = nn.ModuleList([VSSBlock(
            hidden_dim=int(dim * 2**4),
            drop_path=drop_path_rate,
            norm_layer=nn.LayerNorm,
            attn_drop_rate=0,
            expand=self.mlp_ratio,
            d_state=int(base_d_state * 2**2),
        ) for i in range(num_blocks[2])])

        self.latent = nn.ModuleList([VSSBlock(
            hidden_dim=int(dim * 2**4),
            drop_path=drop_path_rate,
            norm_layer=nn.LayerNorm,
            attn_drop_rate=0,
            expand=self.mlp_ratio,
            d_state=int(base_d_state / 2 * 2**3),
        ) for i in range(num_blocks[3])])

        self.refinement = nn.ModuleList([
            VSSBlock(
                hidden_dim=int(dim * 2**4),
                drop_path=drop_path_rate,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                expand=self.mlp_ratio,
                d_state=int(base_d_state / 2 * 2**3)  # base_d_state * 2 ** 2),
            ) for i in range(num_refinement_blocks)
        ])

        # self.output = nn.Conv3d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):

        _, _, H, W, D = inp_img.shape
        inp_enc_level1 = self.patch_embed(inp_img)  # b,hw,c
        out_enc_level1 = inp_enc_level1
        for layer in self.encoder_level1:
            out_enc_level1 = layer(out_enc_level1, [H, W, D])

        inp_enc_level2 = self.down1_2(out_enc_level1, H, W, D)  # b, hwd//8, 2c
        out_enc_level2 = inp_enc_level2
        for layer in self.encoder_level2:
            out_enc_level2 = layer(out_enc_level2, [H // 2, W // 2, D // 2])

        inp_enc_level3 = self.down2_3(out_enc_level2, H // 2, W // 2, D // 2)  # b, hwd//16, 4c
        latent = inp_enc_level3
        for layer in self.latent:
            latent = layer(latent, [H // 4, W // 4, D // 4])

        for layer in self.refinement:
            latent = layer(latent, [H // 4, W // 4, D // 4])

        latent = rearrange(latent, 'b (h w d) c -> b c h w d', h=H // 4, w=W // 4, d=D // 4).contiguous()

        return latent


class MambaIR(nn.Module):
    r""" MambaIR Model
           A PyTorch impl of : `A Simple Baseline for Image Restoration with State Space Model `.

       Args:
           img_size (int | tuple(int)): Input image size. Default 64
           patch_size (int | tuple(int)): Patch size. Default: 1
           in_chans (int): Number of input image channels. Default: 3
           embed_dim (int): Patch embedding dimension. Default: 96
           d_state (int): num of hidden state in the state space model. Default: 16
           depths (tuple(int)): Depth of each RSSG
           drop_rate (float): Dropout rate. Default: 0
           drop_path_rate (float): Stochastic depth rate. Default: 0.1
           norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
           patch_norm (bool): If True, add normalization after patch embedding. Default: True
           use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
           upscale: Upscale factor. 2/3/4 for image SR, 1 for denoising
           img_range: Image range. 1. or 255.
           upsampler: The reconstruction reconstruction module. 'pixelshuffle'/None
           resi_connection: The convolutional block before residual connection. '1conv'/'3conv'
       """

    def __init__(
            self,  # img_size=64, patch_size=1,
            in_chans=3,
            embed_dim=96,
            depths=(6, 6, 6, 6),
            drop_rate=0.,
            d_state=16,
            mlp_ratio=2.,
            drop_path_rate=0.1,
            norm_layer=nn.LayerNorm,
            patch_norm=True,
            use_checkpoint=False,
            upscale=2,  # img_range=1.,
            upsampler='',
            **kwargs):
        super(MambaIR, self).__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        # num_feat = 64
        # self.img_range = img_range
        # if in_chans == 3:
        #     rgb_mean = (0.4488, 0.4371, 0.4040)
        #     self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        # else:
        #     self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler
        self.mlp_ratio = mlp_ratio
        # ------------------------- 1, shallow feature extraction ------------------------- #
        self.conv_first = nn.Conv3d(num_in_ch, embed_dim, 3, 1, 1)

        # ------------------------- 2, deep feature extraction ------------------------- #
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = embed_dim

        # transfer 2D feature map into 1D token sequence, pay attention to whether using normalization
        self.patch_embed = PatchEmbed(  # img_size=img_size, patch_size=patch_size,
            in_chans=embed_dim, embed_dim=embed_dim, norm_layer=norm_layer if self.patch_norm else None)
        # num_patches = self.patch_embed.num_patches
        # patches_resolution = self.patch_embed.patches_resolution
        # self.patches_resolution = patches_resolution

        # return 2D feature map from 1D token sequence
        self.patch_unembed = PatchUnEmbed(  # img_size=img_size, patch_size=patch_size,
            in_chans=embed_dim, embed_dim=embed_dim, norm_layer=norm_layer if self.patch_norm else None)

        self.pos_drop = nn.Dropout(p=drop_rate)
        self.is_light_sr = False
        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build Residual State Space Group (RSSG)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):  # 6-layer
            layer = ResidualGroup(
                dim=embed_dim,
                # input_resolution=(patches_resolution[0], patches_resolution[1], patches_resolution[2]),
                depth=depths[i_layer],
                d_state=d_state,
                mlp_ratio=self.mlp_ratio,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # no impact on SR results
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                # img_size=img_size,
                # patch_size=patch_size,
                is_light_sr=self.is_light_sr)
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        # build the last conv layer in the end of all residual groups
        self.conv_after_body = nn.Conv3d(embed_dim, embed_dim, 3, 1, 1)
        # -------------------------3. high-quality image reconstruction ------------------------ #

        # for lightweight SR (to save parameters)
        if self.upsampler == 'pixelshuffledirect':
            self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch)
        else:
            self.conv_last = nn.Conv3d(embed_dim, embed_dim, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3], x.shape[4])
        x = self.patch_embed(x)  # N,L,C

        x = self.pos_drop(x)

        for i, layer in enumerate(self.layers):
            # print(f'before layer: x size: {x_size}, x shape: {x.shape}')
            x = layer(x, x_size)
            # print(f'after layer: x size: {x_size}, x shape: {x.shape}')

        x = self.norm(x)  # b seq_len c
        x = self.patch_unembed(x, x_size)
        # print(f'after unembed: x size: {x_size}, x shape: {x.shape}')

        return x

    def forward(self, x):
        # self.mean = self.mean.type_as(x)
        # x = (x - self.mean) * self.img_range
        # for lightweight SR
        if self.upsampler == 'pixelshuffledirect':
            # print(f'before conv_first: x size: {x.shape}')
            x = self.conv_first(x)
            # print(f'after conv_first: x size: {x.shape}')
            x_after_body = self.forward_features(x)
            x_after_body = self.conv_after_body(x_after_body)
            x = x_after_body + x
            # print(f'after conv_after_body: x size: {x.shape}')
        else:
            x_first = self.conv_first(x)
            res = self.conv_after_body(self.forward_features(x_first)) + x_first
            res = self.conv_last(res)
            x = x + res
        # remove for asr
        # x = self.upsample(x)
        # x = x / self.img_range + self.mean

        return x

    # def flops(self):
    #    flops = 0
    #    h, w = self.patches_resolution
    #    flops += h * w * 3 * self.embed_dim * 9
    #    flops += self.patch_embed.flops()
    #    for layer in self.layers:
    #        flops += layer.flops()
    #    flops += h * w * 3 * self.embed_dim * self.embed_dim
    #    flops += self.upsample.flops()
    #    return flops


class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """

    def __init__(self, scale, num_feat, num_out_ch):
        self.num_feat = num_feat
        m = []
        m.append(nn.Conv3d(num_feat, (scale**3) * num_out_ch, 3, 1, 1))
        m.append(PixelShuffle3d(scale))
        super(UpsampleOneStep, self).__init__(*m)


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution.

    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/weight_init.py

    The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/weight_init.py
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn('mean is more than 2 std from [a, b] in nn.init.trunc_normal_. '
                      'The distribution of values may be incorrect.', stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        low = norm_cdf((a - mean) / std)
        up = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [low, up], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * low - 1, 2 * up - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


# for MambaIR models (above) ###


class MLP_decoder(nn.Module):

    def __init__(self, latent_dim=128, out_dim=1, depth=4, width=256, activation='LeakyReLU', **kwargs):
        super(MLP_decoder, self).__init__()
        self.use_checkpoint = kwargs.get('use_checkpoint', False)
        self.chunk = kwargs.get('chunk', False)
        latent_dim = latent_dim + 3  # +3 for coordinates

        stage_one = []
        stage_two = []
        activation = getattr(nn, activation)
        for i in range(depth):
            if i == 0:
                stage_one.append(nn.Linear(latent_dim, width))
                stage_two.append(nn.Linear(latent_dim, width))
                stage_one.append(activation())
                stage_two.append(activation())
            elif i == depth - 1:
                stage_one.append(nn.Linear(width, latent_dim))
                stage_two.append(nn.Linear(width, out_dim))
            else:
                stage_one.append(nn.Linear(width, width))
                stage_two.append(nn.Linear(width, width))
                stage_one.append(activation())
                stage_two.append(activation())

        self.stage_one = nn.Sequential(*stage_one)
        self.stage_two = nn.Sequential(*stage_two)

    def forward(self, x):
        if self.chunk:
            x_chunks = x.chunk(8, dim=0)
            h_chunks = [torch.utils.checkpoint.checkpoint(self.stage_one, x_chunk, use_reentrant=False) if self.use_checkpoint else self.stage_one(x_chunk) for x_chunk in x_chunks]
            output_chunks = [torch.utils.checkpoint.checkpoint(self.stage_two, x_chunk + h_chunk, use_reentrant=False) if self.use_checkpoint else self.stage_two(x_chunk + h_chunk) for x_chunk, h_chunk in zip(x_chunks, h_chunks)]
            output = torch.cat(output_chunks, dim=0)
        else:
            h = torch.utils.checkpoint.checkpoint(self.stage_one, x, use_reentrant=False) if self.use_checkpoint else self.stage_one(x)
            output = torch.utils.checkpoint.checkpoint(self.stage_two, x + h, use_reentrant=False) if self.use_checkpoint else self.stage_two(x + h)
        return output


@torch.no_grad()
def make_coord(shape, ranges=None, flatten=True):
    """
    Make coordinates at grid centers.
    Shape: [D, H, W]
    ranges: [[z0, z1], [y0, y1], [x0, x1]]
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


# -------------------------------
# RDN encoder network from ArSSR repo
# <Zhang, Yulun, et al. "Residual dense network for image super-resolution.">
# Here code is modified from: https://github.com/yjn870/RDN-pytorch/blob/master/models.py
# -------------------------------
"""
class DenseLayer(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=3 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return torch.cat([x, self.relu(self.conv(x))], 1)


class RDB(nn.Module):

    def __init__(self, in_channels, growth_rate, num_layers):
        super(RDB, self).__init__()
        self.layers = nn.Sequential(*[DenseLayer(in_channels + growth_rate * i, growth_rate) for i in range(num_layers)])
        # local feature fusion
        self.lff = nn.Conv3d(in_channels + growth_rate * num_layers, in_channels, kernel_size=1)

    def forward(self, x):
        lff = self.lff(self.layers(x))
        x = x + lff
        return x


class RDN(nn.Module):

    def __init__(self, latent_dim=128, num_features=64, growth_rate=64, num_blocks=8, num_layers=3, **kwargs):
        super(RDN, self).__init__()
        self.use_checkpoint = kwargs.get('use_checkpoint', False)

        self.G0 = num_features
        self.G = growth_rate
        self.D = num_blocks
        self.C = num_layers
        # shallow feature extraction
        self.sfe1 = nn.Conv3d(1, num_features, kernel_size=3, padding=3 // 2)
        self.sfe2 = nn.Conv3d(num_features, num_features, kernel_size=3, padding=3 // 2)
        # residual dense blocks
        self.rdbs = nn.ModuleList([RDB(self.G0, self.G, self.C)])
        for _ in range(self.D - 1):
            self.rdbs.append(RDB(self.G0, self.G, self.C))
        # global feature fusion
        self.gff = nn.Sequential(nn.Conv3d(32, 32, kernel_size=1), nn.Conv3d(32, 8, kernel_size=3, padding=3 // 2))
        self.output = nn.Conv3d(8, latent_dim, kernel_size=3, padding=3 // 2)

    def forward(self, x):
        sfe1 = self.sfe1(x)
        sfe2 = self.sfe2(sfe1)
        x = sfe2
        local_features = []
        for i in range(self.D):
            if self.use_checkpoint:
                x = torch.utils.checkpoint.checkpoint(self.rdbs[i], x, use_reentrant=False)
            else:
                x = self.rdbs[i](x)
            local_features.append(x)
        gff = torch.cat(local_features, 1)
        gff = self.gff(gff)
        x = gff + sfe1  # global residual learning
        x = self.output(x)
        return x
"""


class DenseLayer(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=3 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_first = x
        # print(f"x_first shape in DenseLayer: {x_first.shape}")
        x = self.conv(x)
        # print(f"x shape after conv in DenseLayer: {x.shape}")
        x = self.relu(x)
        # print(f"x shape after relu in DenseLayer: {x.shape}")
        x = torch.cat([x_first, x], 1)
        # print(f"x shape after cat in DenseLayer: {x.shape}")
        return x


class RDB(nn.Module):

    def __init__(self, in_channels, growth_rate, num_layers):
        super(RDB, self).__init__()
        self.layers = nn.Sequential(*[DenseLayer(in_channels + growth_rate * i, growth_rate) for i in range(num_layers)])
        # local feature fusion
        self.lff = nn.Conv3d(in_channels + growth_rate * num_layers, growth_rate, kernel_size=1)

    def forward(self, x):
        # print(f"lff in channels : {self.lff.in_channels}, out channels : {self.lff.out_channels}")
        lff = self.layers(x)
        # print(f"lff shape in RDB after layers: {lff.shape}")
        lff = self.lff(lff)
        # print(f"lff shape in RDB: after lff: {lff.shape}")
        return x + lff  # local residual learning


class RDN(nn.Module):

    def __init__(self, latent_dim=16, num_features=16, growth_rate=16, num_blocks=8, num_layers=3, **kwargs):
        super(RDN, self).__init__()
        self.use_checkpoint = kwargs.get('use_checkpoint', False)

        self.G0 = num_features
        self.G = growth_rate
        self.D = num_blocks
        self.C = num_layers
        # shallow feature extraction
        self.sfe1 = nn.Conv3d(1, num_features, kernel_size=3, padding=3 // 2)
        self.sfe2 = nn.Conv3d(num_features, num_features, kernel_size=3, padding=3 // 2)
        # residual dense blocks
        self.rdbs = nn.ModuleList([RDB(self.G0, self.G, self.C)])
        for _ in range(self.D - 1):
            self.rdbs.append(RDB(self.G, self.G, self.C))
        # global feature fusion
        self.gff = nn.Sequential(nn.Conv3d(self.G * self.D, self.G0, kernel_size=1), nn.Conv3d(self.G0, self.G0, kernel_size=3, padding=3 // 2))
        self.output = nn.Conv3d(self.G0, latent_dim, kernel_size=3, padding=3 // 2)

    def forward(self, x):
        # print(f"1: shape: {x.shape}")
        sfe1 = self.sfe1(x)
        # print(f"2: shape: {sfe1.shape}")
        sfe2 = self.sfe2(sfe1)
        # print(f"3: shape: {sfe2.shape}")
        x = sfe2
        # print(f"4: shape: {x.shape}")
        local_features = []
        for i in range(self.D):
            x = self.rdbs[i](x)
            # print(f"5.{i}: shape: {x.shape}")
            local_features.append(x)
        gff = torch.cat(local_features, 1)
        # print(f"6: shape: {gff.shape}")
        gff = self.gff(gff)
        # print(f"7: shape: {gff.shape}")
        x = gff + sfe1  # global residual learning
        # print(f"8: shape: {x.shape}")
        x = self.output(x)
        # print(f"9: shape: {x.shape}")
        return x


if __name__ == '__main__':
    d = 64  # 96
    test_input = torch.randn(1, 1, d, d, d)  # .cuda()
    b, c, h, w, d = test_input.shape

    # encoder = modified_net2(downconv=True)#.cuda() #
    # out = encoder(test_input) #
    # print("encoder our shape test: ", out.shape) #

    encoder = MambaIR(  # img_size=32., patch_size=1,
        in_chans=1,
        embed_dim=16,
        depths=(6, 6, 6, 6),
        drop_rate=0.,
        d_state=16,
        mlp_ratio=1.,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        patch_norm=True,
        use_checkpoint=False,
        upscale=None,
    )  # img_range=1.,)
    # encoder = MambaIREncoder(
    #    inp_channels=1,
    #    out_channels=1,
    #    dim=4,
    #    num_blocks=[4, 4, 4, 4],
    #    num_refinement_blocks=4,
    #    mlp_ratio=1.,
    #    bias=False,
    # )
    test_input = torch.rand(2, 1, 64, 64, 64)
    out = encoder(test_input)
    # print('encoder out shape test: ', out.shape)
