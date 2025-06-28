import matplotlib.gridspec as gridspec
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

    T21_hr = ['T21_cube_z14__Npix256_IC20.mat', 'T21_cube_z15__Npix256_IC20.mat', 'T21_cube_z16__Npix256_IC20.mat', 'T21_cube_z17__Npix256_IC20.mat', 'T21_cube_z18__Npix256_IC20.mat', 'T21_cube_z19__Npix256_IC20.mat', 'T21_cube_z20__Npix256_IC20.mat', 'T21_cube_z21__Npix256_IC20.mat']
    T21_hr = [loadmat(os.path.join(root_dir, 'datasets/varying_IC/T21_cubes', file))['Tlin'] for file in T21_hr]
    T21_hr = [torch.from_numpy(t).unsqueeze(0).unsqueeze(0).to(torch.float32) for t in T21_hr]
    T21_hr = torch.concat(T21_hr, dim=0)

    delta = loadmat(os.path.join(root_dir, 'datasets/varying_IC/IC_cubes/delta_Npix256_IC20.mat'))['delta']
    delta = torch.from_numpy(delta).unsqueeze(0).unsqueeze(0).to(torch.float32)  # Convert to BCHWD format
    delta, _, _ = utils.normalize(delta, mode='standard')
    vbv = loadmat(os.path.join(root_dir, 'datasets/varying_IC/IC_cubes/vbv_Npix256_IC20.mat'))['vbv']
    vbv = torch.from_numpy(vbv).unsqueeze(0).unsqueeze(0).to(torch.float32)  # Convert to BCHWD format
    vbv, _, _ = utils.normalize(vbv, mode='standard')
    z = torch.tensor([14., 15., 16., 17., 18., 19., 20., 21.])  # Example z vector

    print(f'T21_hr shape: {T21_hr.shape}', flush=True)
    print(f'delta shape: {delta.shape}', flush=True)
    print(f'vbv shape: {vbv.shape}', flush=True)
    print(f'z shape: {z.shape}', flush=True)

    with open(os.path.join(root_dir, 'good_experiments/ASR21cm_GAN_z_conditional_3/ASR21cm_GAN_options.yml'), mode='r') as f:
        opt = yaml.load(f, Loader=ordered_yaml()[0])
    opt['rank'], opt['world_size'] = get_dist_info()
    opt['is_train'] = False
    opt['num_gpu'] = 0
    results_root = os.path.join(root_dir, 'results', opt['name'])
    opt['path']['results_root'] = results_root
    opt['path']['log'] = results_root
    opt['path']['visualization'] = os.path.join(results_root, 'visualization')

    net_g = build_network(opt['network_g'])
    load_path = os.path.join(root_dir, 'good_experiments/ASR21cm_GAN_z_conditional_3/models/net_g_2000.pth')
    load_net = torch.load(load_path, map_location=lambda storage, loc: storage)
    net_g.load_state_dict(load_net['params_ema'], strict=True)

    b, c, h, w, d = T21_hr.shape
    xyz_hr = make_coord([h, h, h], ranges=None, flatten=False)
    xyz_hr = xyz_hr.view(1, -1, 3)
    xyz_hr = xyz_hr.repeat(1, 1, 1)

    T21_sr = []
    T21_lr = []
    net_g.eval()
    for i in tqdm(range(len(T21_hr)), desc='Scaling...', total=len(T21_hr)):
        input = torch.nn.functional.interpolate(T21_hr[i:i + 1], size=64, mode='trilinear')
        input_mean = torch.mean(input, dim=(1, 2, 3, 4), keepdim=True)
        input_std = torch.std(input, dim=(1, 2, 3, 4), keepdim=True)
        input, _, _ = utils.normalize(input, mode='standard')

        with torch.no_grad():
            # output, _ = net_g(img_lr=input, xyz_hr=xyz_hr, delta=delta, vbv=vbv, z=z[i:i+1])
            # T21_sr.append(output * input_std + input_mean)
            input = input * input_std + input_mean  # denormalize input
            input = torch.nn.functional.interpolate(input, size=256, mode='trilinear')  # upsample input to match output size
            T21_lr.append(input)
    torch.cuda.empty_cache()

    T21_lr = torch.concat(T21_lr, dim=0)
    # T21_sr = torch.concat(T21_sr, dim=0)
    # torch.save(T21_sr, os.path.join(current_dir, 'T21_sr_z.pt'))
    T21_sr = torch.load(os.path.join(current_dir, 'T21_sr_z.pt'))

    print(f'T21_sr shape: {T21_sr.shape}', flush=True)
    print(f'T21_lr shape: {T21_lr.shape}', flush=True)
    print(f'T21_hr shape: {T21_hr.shape}', flush=True)

    # matplotlib settings
    rasterized = False
    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'serif',
        'font.serif': 'cm',
        'font.size': 22,
    })

    nrows = 4
    ncols = 4
    fig = plt.figure(figsize=(ncols * 6, nrows * 6))
    gs = gridspec.GridSpec(2, ncols, figure=fig, hspace=0.1)
    sgs_hist = gridspec.GridSpecFromSubplotSpec(2, ncols, subplot_spec=gs[0, :], wspace=0, hspace=0)
    sgs_dsq = gridspec.GridSpecFromSubplotSpec(2, ncols, subplot_spec=gs[1, :], wspace=0, hspace=0)
    axes = np.empty((nrows, ncols), dtype=object)

    axes[1, 0] = fig.add_subplot(sgs_hist[1, 0])
    axes[0, 0] = fig.add_subplot(sgs_hist[0, 0], sharey=axes[1, 0], sharex=axes[1, 0])
    axes[3, 0] = fig.add_subplot(sgs_dsq[1, 0])
    axes[2, 0] = fig.add_subplot(sgs_dsq[0, 0], sharey=axes[3, 0], sharex=axes[3, 0])
    for i in range(1, ncols):
        axes[0, i] = fig.add_subplot(sgs_hist[0, i], sharey=axes[1, 0], sharex=axes[1, 0])
        axes[1, i] = fig.add_subplot(sgs_hist[1, i], sharey=axes[1, 0], sharex=axes[1, 0])
        axes[2, i] = fig.add_subplot(sgs_dsq[0, i], sharey=axes[3, 0], sharex=axes[3, 0])
        axes[3, i] = fig.add_subplot(sgs_dsq[1, i], sharey=axes[3, 0], sharex=axes[3, 0])
    for row in range(nrows):
        for col in range(ncols):
            if col > 0:
                axes[row, col].tick_params(labelleft=False)  # Hide y-ticks for all but the first column
            if row == 0 or row == 2:
                axes[row, col].tick_params(labelbottom=False)  # Hide x-ticks for the first dsq row

    slice_idx = 128

    k_hr, dsq_hr = calculate_power_spectrum(data_x=T21_hr, Lpix=3, kbins=100, dsq=True, method='torch', device='cpu')

    progress_bar = tqdm(total=8, desc='Plotting...', position=0, leave=True)
    for row in range(2):
        for col in range(4):
            idx = np.ravel_multi_index((row, col), (2, 4))
            k_sr, dsq_sr = calculate_power_spectrum(data_x=T21_sr[idx:idx + 1], Lpix=3, kbins=100, dsq=True, method='torch', device='cpu')
            k_lr, dsq_lr = calculate_power_spectrum(data_x=T21_lr[idx:idx + 1], Lpix=3, kbins=100, dsq=True, method='torch', device='cpu')

            xmin = min(T21_sr[idx].min(), T21_hr[idx].min())
            xmax = max(T21_sr[idx].max(), T21_hr[idx].max())
            bins = np.linspace(xmin, xmax, 100)
            axes[row, col].hist(T21_hr[idx].cpu().numpy().flatten(), bins=bins, alpha=0.5, label='HR', histtype='step', linewidth=4, density=True, zorder=2, rasterized=rasterized)
            axes[row, col].hist(T21_sr[idx].cpu().numpy().flatten(), bins=bins, alpha=0.5, label='SR', histtype='step', linewidth=4, density=True, zorder=3, rasterized=rasterized)
            axes[row, col].hist(T21_lr[idx].cpu().numpy().flatten(), bins=bins, alpha=0.5, label='LR', histtype='step', linewidth=4, density=True, zorder=1, rasterized=rasterized)
            rmse_sr = torch.sqrt(torch.mean((T21_sr[idx] - T21_hr[idx])**2))
            rmse_lr = torch.sqrt(torch.mean((T21_lr[idx] - T21_hr[idx])**2))
            axes[row, col].text(0.05, 0.95, rf'$z={z[idx]:.0f}$' + '\n' + rf'$\mathrm{{RMSE}}_{{SR}} = {rmse_sr.item():.2f}\ \mathrm{{mK}}$' + '\n' + rf'$\mathrm{{RMSE}}_{{LR}} = {rmse_lr.item():.2f}\ \mathrm{{mK}}$', transform=axes[row, col].transAxes, fontsize=plt.rcParams['font.size'] - 2, ha='left', va='top')

            axes[row + 2, col].loglog(k_hr, dsq_hr[idx, 0], label='$T_{{21}}$ HR', ls='solid', lw=2, rasterized=rasterized, zorder=2)
            axes[row + 2, col].loglog(k_sr, dsq_sr[0, 0], label='$T_{{21}}$ SR', ls='solid', lw=2, rasterized=rasterized, zorder=3)
            axes[row + 2, col].loglog(k_lr, dsq_lr[0, 0], label='$T_{{21}}$ LR (Interpolated)', ls='solid', lw=2, rasterized=rasterized, zorder=1)
            rmse_dsq_sr = torch.sqrt(torch.nanmean((dsq_sr - dsq_hr)**2))
            rmse_dsq_lr = torch.sqrt(torch.nanmean((dsq_lr - dsq_hr)**2))
            axes[row + 2, col].text(0.05, 0.95, rf's={256 / 64:.2f}, z={z[0].item():.0f}' + '\n' + rf'$\mathrm{{RMSE}}_{{SR}}^{{\Delta^2}} = {rmse_dsq_sr.item():.2f}\ \mathrm{{mK}}^2$' + '\n' + rf'$\mathrm{{RMSE}}_{{LR}}^{{\Delta^2}} = {rmse_dsq_lr.item():.2f}\ \mathrm{{mK}}^2$', transform=axes[row + 2, col].transAxes, fontsize=plt.rcParams['font.size'] - 2, ha='left', va='top')

            if row == 0:
                axes[row + 1, col].set_xlabel('$T_{21}$ [$\\mathrm{mK}$]')
                if col == 0:
                    axes[row, col].set_ylabel('PDF')
                    axes[row + 1, col].set_ylabel('PDF')
            if row == 1:
                axes[row + 2, col].set_xlabel('$k\\ [\\mathrm{{cMpc^{-1}}}]$')
                if col == 0:
                    axes[row + 1, col].set_ylabel(r'$P(k)$ [${\rm mK^2}$]')
                    axes[row + 2, col].set_ylabel(r'$P(k)$ [${\rm mK^2}$]')

            progress_bar.update(1)

    # Get the default color cycle as a list of hex colors
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    import matplotlib.lines as mlines

    # After plotting your histograms, before calling legend:
    line_hr = mlines.Line2D([], [], color=colors[0], linewidth=4, label='HR')
    line_sr = mlines.Line2D([], [], color=colors[1], linewidth=4, label='SR')
    line_lr = mlines.Line2D([], [], color=colors[2], linewidth=4, label='LRI')

    axes[0, 0].legend(handles=[line_hr, line_sr, line_lr], fontsize=plt.rcParams['font.size'] - 2, frameon=False, loc='center left')
    # axes[0, 0].legend(fontsize=plt.rcParams['font.size'] - 2, frameon=False, loc='center left')

    axes[1, 0].set_xlim(-220, 10)
    axes[3, 0].set_ylim(1e0, 1e3)

    save_img_path = os.path.join(current_dir, 'redshift_scaling.pdf')
    plt.savefig(save_img_path, bbox_inches='tight')
    plt.close(fig)
