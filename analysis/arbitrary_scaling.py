import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import os
import SR21cm.utils as utils
import torch
import yaml
from scipy.io import loadmat
from SR21cm.utils import calculate_power_spectrum
from tqdm import tqdm

from ASR21cm.archs import *
from ASR21cm.archs.arch_utils import make_coord
from basicsr.archs import build_network
from basicsr.utils.dist_util import get_dist_info
from basicsr.utils.options import ordered_yaml

if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    Npix = 512
    IC = 0

    T21_hr = loadmat(os.path.join(root_dir, f'datasets/varying_IC/T21_cubes/T21_cube_z18__Npix{Npix}_IC{IC}.mat'))['Tlin']
    T21_hr = torch.from_numpy(T21_hr).unsqueeze(0).unsqueeze(0).to(torch.float32).to(device)  # Convert to BCHWD format

    delta = loadmat(os.path.join(root_dir, f'datasets/varying_IC/IC_cubes/delta_Npix{Npix}_IC{IC}.mat'))['delta']
    delta = torch.from_numpy(delta).unsqueeze(0).unsqueeze(0).to(torch.float32).to(device)  # Convert to BCHWD format
    delta, _, _ = utils.normalize(delta, mode='standard')
    vbv = loadmat(os.path.join(root_dir, f'datasets/varying_IC/IC_cubes/vbv_Npix{Npix}_IC{IC}.mat'))['vbv']
    vbv = torch.from_numpy(vbv).unsqueeze(0).unsqueeze(0).to(torch.float32).to(device)  # Convert to BCHWD format
    vbv, _, _ = utils.normalize(vbv, mode='standard')
    z = torch.tensor([
        18.,
    ]).to(device)  # Example z vector

    print(f'delta shape: {delta.shape}', flush=True)
    print(f'vbv shape: {vbv.shape}', flush=True)
    print(f'z shape: {z.shape}', flush=True)

    with open(os.path.join(root_dir, 'good_experiments/ASR21cm_GAN_z_conditional_3/ASR21cm_GAN_options.yml'), mode='r') as f:
        opt = yaml.load(f, Loader=ordered_yaml()[0])
    opt['rank'], opt['world_size'] = get_dist_info()
    opt['is_train'] = False
    results_root = os.path.join(root_dir, 'results', opt['name'])
    opt['path']['results_root'] = results_root
    opt['path']['log'] = results_root
    opt['path']['visualization'] = os.path.join(results_root, 'visualization')
    opt['num_gpu'] = torch.cuda.device_count()

    net_g = build_network(opt['network_g'])
    load_path = os.path.join(root_dir, 'good_experiments/ASR21cm_GAN_z_conditional_3/models/net_g_2000.pth')
    load_net = torch.load(load_path, map_location=device)
    net_g.load_state_dict(load_net['params_ema'], strict=True)
    net_g = net_g.to(device)

    b, c, h, w, d = T21_hr.shape
    xyz_hr = make_coord([h, h, h], ranges=None, flatten=False)
    xyz_hr = xyz_hr.view(1, -1, 3)
    xyz_hr = xyz_hr.repeat(b, 1, 1).to(device)

    # sizes = [64, 77, 96, 128] # for 256
    sizes = (Npix * 3 / np.array([12, 10, 8, 6])).round().astype(int)  # for 512

    T21_sr = []
    T21_lr = []
    net_g.eval()
    for i, size in enumerate(tqdm(sizes, desc='Scaling...')):
        input = torch.nn.functional.interpolate(T21_hr, size=size, mode='trilinear')
        input_mean = torch.mean(input, dim=(1, 2, 3, 4), keepdim=True)
        input_std = torch.std(input, dim=(1, 2, 3, 4), keepdim=True)
        input, _, _ = utils.normalize(input, mode='standard')

        with torch.no_grad():
            # output, _ =  net_g(img_lr=input, xyz_hr=xyz_hr, delta=delta, vbv=vbv, z=z)
            output = torch.rand_like(delta)
            T21_sr.append(output * input_std + input_mean)
            input = input * input_std + input_mean  # denormalize input
            input = torch.nn.functional.interpolate(input, size=Npix, mode='trilinear')  # upsample input to match output size
            T21_lr.append(input)
    torch.cuda.empty_cache()

    T21_lr = torch.concat(T21_lr, dim=0)
    T21_sr = torch.concat(T21_sr, dim=0)
    # torch.save(T21_sr, os.path.join(current_dir, 'files', 'T21_sr.pt'))

    # T21_lr = torch.load(os.path.join(current_dir, 'files', 'T21_lr.pt'))  # Load the saved low-res data
    T21_sr = torch.load(os.path.join(current_dir, 'files', 'T21_sr.pt'))  # Load the saved super-resolved data

    print(f'T21_sr shape: {T21_sr.shape}', flush=True)
    print(f'T21_lr shape: {T21_lr.shape}', flush=True)

    # matplotlib settings
    rasterized = False
    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'serif',
        'font.serif': 'cm',
        'font.size': 22,
    })

    nrows = 5
    fig = plt.figure(figsize=(len(sizes) * 5.5, nrows * 6))
    gs = gridspec.GridSpec(2, len(sizes), figure=fig, wspace=0, hspace=0.05, height_ratios=[3, 2])
    sgs_im = gridspec.GridSpecFromSubplotSpec(3, len(sizes), subplot_spec=gs[0, :], wspace=0.05, hspace=0.15)
    sgs_stats = gridspec.GridSpecFromSubplotSpec(2, len(sizes), subplot_spec=gs[1, :], wspace=0.05, hspace=0.2)
    axes = np.empty((nrows, len(sizes)), dtype=object)
    axes[0, 0] = fig.add_subplot(sgs_im[0, 0])
    axes[1, 0] = fig.add_subplot(sgs_im[1, 0])
    axes[2, 0] = fig.add_subplot(sgs_im[2, 0])
    axes[3, 0] = fig.add_subplot(sgs_stats[0, 0])
    axes[4, 0] = fig.add_subplot(sgs_stats[1, 0])
    for i in range(1, len(sizes)):
        axes[0, i] = fig.add_subplot(sgs_im[0, i], sharey=axes[0, 0])
        axes[1, i] = fig.add_subplot(sgs_im[1, i], sharey=axes[1, 0])
        axes[2, i] = fig.add_subplot(sgs_im[2, i], sharey=axes[2, 0])
        axes[3, i] = fig.add_subplot(sgs_stats[0, i], sharey=axes[3, 0])
        axes[4, i] = fig.add_subplot(sgs_stats[1, i], sharey=axes[4, 0])
    for row in range(5):
        for col in range(len(sizes)):
            if col > 0:
                axes[row, col].tick_params(labelleft=False)  # Hides y tick labels only for this axis
            if row < 3:
                axes[row, col].tick_params(labelbottom=False)  # Hides x tick labels only for this axis
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    slice_idx = Npix // 2  # Middle slice for visualization
    vmin = T21_sr.min()  # torch.quantile(T21_sr, 0.01).item()
    vmax = T21_sr.max()  # torch.quantile(T21_sr, 0.99).item()
    k_hr, dsq_hr = calculate_power_spectrum(data_x=T21_hr, Lpix=3, kbins=100, dsq=True, method='torch', device='cpu')
    for i, size in enumerate(tqdm(sizes, desc='Plotting...')):
        k_sr, dsq_sr = calculate_power_spectrum(data_x=T21_sr[i:i + 1], Lpix=3, kbins=100, dsq=True, method='torch', device='cpu')
        k_lr, dsq_lr = calculate_power_spectrum(data_x=T21_lr[i:i + 1], Lpix=3, kbins=100, dsq=True, method='torch', device='cpu')
        scale = T21_sr[i].shape[2] / size

        axes[0, i].imshow(T21_hr[0, 0, :, :, slice_idx].cpu().numpy(), vmin=vmin, vmax=vmax, rasterized=rasterized)
        axes[0, i].set_title('HR')
        axes[1, i].imshow(T21_sr[i, 0, :, :, slice_idx].cpu().numpy(), vmin=vmin, vmax=vmax, rasterized=rasterized)
        axes[1, i].set_title('SR')
        axes[2, i].imshow(T21_lr[i, 0, :, :, slice_idx].cpu().numpy(), vmin=vmin, vmax=vmax, rasterized=rasterized)
        axes[2, i].set_title('LR Interpolated (LRI)')

        xmin = min(T21_sr[i].min(), T21_hr.min())
        xmax = max(T21_sr[i].max(), T21_hr.max())
        bins = np.linspace(xmin, xmax, 100)
        axes[3, i].hist(T21_hr.cpu().numpy().flatten(), bins=bins, alpha=0.5, label='HR', histtype='step', linewidth=4, density=True, zorder=2, rasterized=rasterized, color=colors[0])
        axes[3, i].hist(T21_sr[i].cpu().numpy().flatten(), bins=bins, alpha=0.5, label='SR', histtype='step', linewidth=4, density=True, zorder=3, rasterized=rasterized, color=colors[1])
        axes[3, i].hist(T21_lr[i].cpu().numpy().flatten(), bins=bins, alpha=0.5, label='LR', histtype='step', linewidth=4, density=True, zorder=1, rasterized=rasterized, color=colors[2])
        axes[3, i].set_xlabel('$T_{21}$ [$\\mathrm{mK}$]')
        rmse_sr = torch.sqrt(torch.mean((T21_sr[i:i + 1] - T21_hr)**2))
        rmse_lr = torch.sqrt(torch.mean((T21_lr[i:i + 1] - T21_hr)**2))
        axes[3, i].text(0.05, 0.95, rf's={Npix / size:.2f}, z={z[0].item():.0f}' + '\n' + rf'$\mathrm{{RMSE}}^{{T_{{21}}}}_{{SR}} = {rmse_sr.item():.2f}\ \mathrm{{mK}}$' + '\n' + rf'$\mathrm{{RMSE}}^{{T_{{21}}}}_{{LRI}} = {rmse_lr.item():.2f}\ \mathrm{{mK}}$', transform=axes[3, i].transAxes, fontsize=plt.rcParams['font.size'] - 2, ha='left', va='top')

        axes[4, i].loglog(k_hr, dsq_hr[0, 0], label='$T_{{21}}$ HR', ls='solid', lw=2, rasterized=rasterized, zorder=2, color=colors[0])
        axes[4, i].loglog(k_sr, dsq_sr[0, 0], label='$T_{{21}}$ SR', ls='solid', lw=2, rasterized=rasterized, zorder=3, color=colors[1])
        axes[4, i].loglog(k_lr, dsq_lr[0, 0], label='$T_{{21}}$ LRI', ls='solid', lw=2, rasterized=rasterized, zorder=1, color=colors[2])
        RMSE_dsq_sr = torch.sqrt(torch.nanmean((dsq_sr - dsq_hr)**2))
        RMSE_dsq_lr = torch.sqrt(torch.nanmean((dsq_lr - dsq_hr)**2))
        axes[4, i].text(0.05, 0.95, rf's={Npix / size:.2f}, z={z[0].item():.0f}' + '\n' + rf'$\mathrm{{RMSE}}^{{\Delta^2}}_{{SR}} = {RMSE_dsq_sr.item():.2f}\ \mathrm{{mK}}^2$' + '\n' + rf'$\mathrm{{RMSE}}^{{\Delta^2}}_{{LRI}} = {RMSE_dsq_lr.item():.2f}\ \mathrm{{mK}}^2$', transform=axes[4, i].transAxes, fontsize=plt.rcParams['font.size'] - 2, ha='left', va='top')

        axes[4, i].set_xlabel('$k\\ [\\mathrm{{cMpc^{-1}}}]$')
        # axes[4, i].legend(fontsize=plt.rcParams['font.size'] - 2, frameon=False, loc='lower left')

    # After plotting your histograms, before calling legend:
    line_hr = mlines.Line2D([], [], color=colors[0], linewidth=4, label='HR')
    line_sr = mlines.Line2D([], [], color=colors[1], linewidth=4, label='SR')
    line_lr = mlines.Line2D([], [], color=colors[2], linewidth=4, label='LRI')
    axes[3, 0].legend(handles=[line_hr, line_sr, line_lr], fontsize=plt.rcParams['font.size'] - 2, frameon=False, loc='center left')

    axes[3, 0].set_title(r'12 $\rightarrow$ 3 [cMpc/voxel]')
    axes[3, 1].set_title(r'10 $\rightarrow$ 3 [cMpc/voxel]')
    axes[3, 2].set_title(r'8 $\rightarrow$ 3 [cMpc/voxel]')
    axes[3, 3].set_title(r'6 $\rightarrow$ 3 [cMpc/voxel]')
    axes[3, 0].set_ylabel('PDF')
    axes[3, 0].set_xlim(-250, -50)
    axes[3, 0].set_ylim(0, 0.06)
    axes[4, 0].set_ylabel(r'$\Delta^2(k)$ [${\rm mK^2}$]')
    axes[4, 0].set_ylim(1e-1, 1e3)

    save_img_path = os.path.join(current_dir, 'plots', f'arbitrary_scaling_Npix{Npix}.pdf')
    plt.savefig(save_img_path, bbox_inches='tight')
    plt.close(fig)
