import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from scipy.integrate import simpson as simps
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

def calculate_KL_divergence(vol1, vol2, num_bins=100, epsilon=1e-10):
    """
    vol1: Target volume (ground truth)
    vol2: Predicted volume
    num_bins: Number of bins for histogram
    epsilon: Small value to avoid division by zero
    Returns: KL divergence D_KL(P || Q) where P is the histogram of vol1 and Q is the histogram of vol2
    """
    # Flatten volumes
    vol1 = vol1.flatten()
    vol2 = vol2.flatten()

    # Get min/max range across both volumes
    vmin = torch.min(torch.cat([vol1, vol2]))
    vmax = torch.max(torch.cat([vol1, vol2]))

    # Histogram as PDF (density=True)
    hist1 = torch.histc(vol1, bins=num_bins, min=vmin.item(), max=vmax.item())
    hist2 = torch.histc(vol2, bins=num_bins, min=vmin.item(), max=vmax.item())

    # Add epsilon and normalize to get proper PDFs
    P = hist1 + epsilon
    Q = hist2 + epsilon

    P /= P.sum()
    Q /= Q.sum()

    # KL divergence: D_KL(P || Q)
    kl = torch.nn.functional.kl_div(Q.log(), P, reduction='sum')  # input=log(Q), target=P

    return kl



def plot_rmse_heatmap(data, sizes, redshifts, log=False, title='RMSE Heatmap', label='RMSE', cmap='viridis', hatch=None, legend=None, savefig=None, fig=None, ax=None, vmin=None, vmax=None):
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        own_fig = True
    else:
        own_fig = False
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()

    if log:
        data = np.log10(data)
        vmin = np.log10(vmin)
        vmax = np.log10(vmax)
        label = rf'$\log_{{10}}$ ({label})'

    N = 8
    vmin = np.floor(vmin * 10) / 10
    vmax = np.ceil(vmax * 10) / 10

    bounds = np.linspace(vmin, vmax, num=N).round(1)

    cmap = plt.get_cmap(cmap, lut=N)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    im = ax.imshow(
        data,
        aspect='auto',
        origin='lower',
        cmap=cmap,
        norm=norm,
    )
    ax.set_xlabel('Scale factor (s)')
    ax.set_ylabel('Redshift (z)')
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, boundaries=bounds, ticks=bounds)
    cbar.set_label(label)
    # Set ticks at the center of each pixel
    ax.set_xticks(np.arange(len(sizes)))
    ax.set_yticks(np.arange(len(redshifts)))
    # Set tick labels to the actual values
    ax.set_xticklabels(sizes)
    ax.set_yticklabels(list(redshifts))
    # ax.plot([-0.5, 3.5], [7.5, 7.5], linestyle='--', color='black', linewidth=1)
    # ax.plot([3.5, 3.5], [-0.5, 7.5], linestyle='--', color='black', linewidth=1)

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    x1, y1 = -0.5, ymax
    x2, y2 = -0.5, 7.5
    x3, y3 = 3.5, 7.5
    x4, y4 = 3.5, ymin
    x5, y5 = xmax, ymin
    x6, y6 = xmax, ymax

    if hatch:
        polygon = patches.Polygon(
            [[x1, y1], [x2, y2], [x3, y3], [x4, y4], [x5, y5], [x6, y6]],
            closed=True,
            facecolor='none',
            edgecolor='k',
            hatch='//',
            alpha=0.8,
            label='Out Of Distribution (OOD)',
        )
        ax.add_patch(polygon)
    if legend is not None:
        ax.legend()  # handles=[polygon], loc='upper right', fontsize='small')
    plt.tight_layout()
    if savefig is not None:
        fig.savefig(savefig, bbox_inches='tight')
        # plt.close(fig)
    elif own_fig:
        plt.show()


if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if False:
        with open(os.path.join(root_dir, 'good_experiments/ASR21cm_GAN_z_conditional_3/ASR21cm_GAN_options.yml'), mode='r') as f:
            opt = yaml.load(f, Loader=ordered_yaml()[0])
        opt['rank'], opt['world_size'] = get_dist_info()
        opt['is_train'] = False
        opt['num_gpu'] = 4
        results_root = os.path.join(root_dir, 'results', opt['name'])
        opt['path']['results_root'] = results_root
        opt['path']['log'] = results_root
        opt['path']['visualization'] = os.path.join(results_root, 'visualization')

    device = 'cuda'
    net_g = build_network(opt['network_g']).to(device=device)
    load_path = os.path.join(root_dir, 'good_experiments/ASR21cm_GAN_z_conditional_3/models/net_g_2000.pth')
    load_net = torch.load(load_path, map_location=device)
    net_g.load_state_dict(load_net['params_ema'], strict=True)
    net_g = net_g.to(device=device)
    net_g.eval()


        sizes = [32, 38, 48, 64, 77, 96, 128]
        redshifts = range(14, 25)
        ICs = range(80)

    xyz_hr = make_coord([256, 256, 256], ranges=None, flatten=False).to(device=device)
    xyz_hr = xyz_hr.view(1, -1, 3)
    xyz_hr = xyz_hr.repeat(1, 1, 1)

    global_signal_hr = torch.zeros((len(ICs), len(sizes), len(redshifts))).to(device=device)
    std_hr = torch.zeros((len(ICs), len(sizes), len(redshifts))).to(device=device)

    RMSE_sr = torch.empty((len(ICs), len(sizes), len(redshifts))).to(device=device)
    RMSE_sr_dsq = torch.empty((len(ICs), len(sizes), len(redshifts))).to(device=device)
    DKL_sr = torch.empty((len(ICs), len(sizes), len(redshifts))).to(device=device)
    global_signal_sr = torch.zeros((len(ICs), len(sizes), len(redshifts))).to(device=device)
    std_sr = torch.zeros((len(ICs), len(sizes), len(redshifts))).to(device=device)

    RMSE_lr = torch.empty((len(ICs), len(sizes), len(redshifts))).to(device=device)
    RMSE_lr_dsq = torch.empty((len(ICs), len(sizes), len(redshifts))).to(device=device)
    DKL_lr = torch.empty((len(ICs), len(sizes), len(redshifts))).to(device=device)
    global_signal_lr = torch.zeros((len(ICs), len(sizes), len(redshifts))).to(device=device)
    std_lr = torch.zeros((len(ICs), len(sizes), len(redshifts))).to(device=device)

        progress_bar = tqdm(total=len(ICs) * len(sizes) * len(redshifts), desc='Processing', unit='cube')

        for i, IC in enumerate(ICs):
            file_delta_IC = os.path.join(root_dir, 'datasets/varying_IC/IC_cubes', f'delta_Npix256_IC{IC}.mat')
            delta = loadmat(file_delta_IC)['delta']
            delta = torch.from_numpy(delta).unsqueeze(0).unsqueeze(0).to(torch.float32).to(device=device)  # Convert to BCHWD format
            delta, _, _ = utils.normalize(delta, mode='standard')

            file_vbv_IC = os.path.join(root_dir, 'datasets/varying_IC/IC_cubes', f'vbv_Npix256_IC{IC}.mat')
            vbv = loadmat(file_vbv_IC)['vbv']
            vbv = torch.from_numpy(vbv).unsqueeze(0).unsqueeze(0).to(torch.float32).to(device=device)  # Convert to BCHWD format
            vbv, _, _ = utils.normalize(vbv, mode='standard')

            for j, size in enumerate(sizes):
                for k, z in enumerate(redshifts):
                    file_T21 = os.path.join(root_dir, 'datasets/varying_IC/T21_cubes', f'T21_cube_z{z}__Npix256_IC{IC}.mat')
                    T21_hr = loadmat(file_T21)['Tlin']
                    T21_hr = torch.from_numpy(T21_hr).unsqueeze(0).unsqueeze(0).to(torch.float32).to(device=device)
                    T21_lr = torch.nn.functional.interpolate(T21_hr, size=size, mode='trilinear')
                    T21_lr, T21_lr_mean, T21_lr_std = utils.normalize(T21_lr, mode='standard')
                    z = torch.tensor([z]).to(device=device)  # Example z vector

                    with torch.no_grad():
                        # print(f'Devices: {T21_lr.device}, {xyz_hr.device}, {delta.device}, {vbv.device}, z: {z.device}, net_g: {net_g.device}', flush=True)
                        # assert False
                        T21_sr, _ = net_g(img_lr=T21_lr, xyz_hr=xyz_hr, delta=delta, vbv=vbv, z=z)
                        # T21_sr = torch.randn_like(T21_hr)
                        T21_sr = T21_sr * T21_lr_std + T21_lr_mean

                    T21_lr = T21_lr * T21_lr_std + T21_lr_mean  # denormalize input

                    T21_lr = torch.nn.functional.interpolate(T21_lr, size=256, mode='trilinear')  # upsample input to match output size
                    DKL_sr[i, j, k] = calculate_KL_divergence(T21_hr, T21_sr).item()
                    DKL_lr[i, j, k] = calculate_KL_divergence(T21_hr, T21_lr).item()

                    global_signal_hr[i, j, k] = torch.mean(T21_hr).item()
                    global_signal_sr[i, j, k] = torch.mean(T21_sr).item()
                    global_signal_lr[i, j, k] = torch.mean(T21_lr).item()

                    std_hr[i, j, k] = torch.std(T21_hr).item()
                    std_sr[i, j, k] = torch.std(T21_sr).item()
                    std_lr[i, j, k] = torch.std(T21_lr).item()

                        rmse_sr = torch.sqrt(torch.mean((T21_sr - T21_hr)**2))
                        rmse_lr = torch.sqrt(torch.mean((T21_lr - T21_hr)**2))
                        RMSE_sr[i, j, k] = rmse_sr.item()
                        RMSE_lr[i, j, k] = rmse_lr.item()

                    k_hr, dsq_hr = calculate_power_spectrum(data_x=T21_hr, Lpix=3, kbins=100, dsq=True, method='torch', device=device)
                    k_sr, dsq_sr = calculate_power_spectrum(data_x=T21_sr, Lpix=3, kbins=100, dsq=True, method='torch', device=device)
                    k_lr, dsq_lr = calculate_power_spectrum(data_x=T21_lr, Lpix=3, kbins=100, dsq=True, method='torch', device=device)
                    rmse_sr_dsq = torch.sqrt(torch.nanmean((dsq_sr - dsq_hr) ** 2))
                    rmse_lr_dsq = torch.sqrt(torch.nanmean((dsq_lr - dsq_hr) ** 2))
                    RMSE_sr_dsq[i, j, k] = rmse_sr_dsq.item()
                    RMSE_lr_dsq[i, j, k] = rmse_lr_dsq.item()

                    # Compute DKL for super-resolved and low-resolved outputs
                    T21_sr = T21_sr.squeeze(dim=(0,1)).cpu().numpy()
                    T21_lr = T21_lr.squeeze(dim=(0,1)).cpu().numpy()
                    T21_hr = T21_hr.squeeze(dim=(0,1)).cpu().numpy()








                    torch.cuda.empty_cache()
                    progress_bar.update(1)

    idx = 4
    torch.save(RMSE_sr, os.path.join(current_dir, 'files', f'RMSE_sr_{idx}.pt'))
    torch.save(RMSE_sr_dsq, os.path.join(current_dir, 'files', f'RMSE_sr_dsq_{idx}.pt'))
    torch.save(RMSE_lr, os.path.join(current_dir, 'files', f'RMSE_lr_{idx}.pt'))
    torch.save(RMSE_lr_dsq, os.path.join(current_dir, 'files', f'RMSE_lr_dsq_{idx}.pt'))
    torch.save(DKL_sr, os.path.join(current_dir, 'files', f'DKL_sr_{idx}.pt'))
    torch.save(DKL_lr, os.path.join(current_dir, 'files', f'DKL_lr_{idx}.pt'))
    torch.save(global_signal_hr, os.path.join(current_dir, 'files', f'global_signal_hr_{idx}.pt'))
    torch.save(global_signal_sr, os.path.join(current_dir, 'files', f'global_signal_sr_{idx}.pt'))
    torch.save(global_signal_lr, os.path.join(current_dir, 'files', f'global_signal_lr_{idx}.pt'))
    torch.save(std_hr, os.path.join(current_dir, 'files', f'std_hr_{idx}.pt'))
    torch.save(std_sr, os.path.join(current_dir, 'files', f'std_sr_{idx}.pt'))
    torch.save(std_lr, os.path.join(current_dir, 'files', f'std_lr_{idx}.pt'))

    else:
        metric1_sr = torch.load(os.path.join(current_dir, 'files', 'DKL_sr_4.pt'), map_location='cpu')
        metric1_lr = torch.load(os.path.join(current_dir, 'files', 'DKL_lr_4.pt'), map_location='cpu')

        metric2_sr = torch.load(os.path.join(current_dir, 'files', 'RMSE_sr_dsq_4.pt'), map_location='cpu')
        metric2_lr = torch.load(os.path.join(current_dir, 'files', 'RMSE_lr_dsq_4.pt'), map_location='cpu')

        metric3_sr = torch.load(os.path.join(current_dir, 'files', 'global_signal_sr_4.pt'), map_location='cpu')
        metric3_lr = torch.load(os.path.join(current_dir, 'files', 'global_signal_lr_4.pt'), map_location='cpu')
        metric3_hr = torch.load(os.path.join(current_dir, 'files', 'global_signal_hr_4.pt'), map_location='cpu')

        print(f'DKL_sr shape: {metric1_sr.shape}', flush=True)
        print(f'DKL_lr shape: {metric1_lr.shape}', flush=True)
        print(f'RMSE_lr_dsq shape: {metric2_lr.shape}', flush=True)
        print(f'RMSE_sr_dsq shape: {metric2_sr.shape}', flush=True)
        print(f'global_signal_sr shape: {metric3_sr.shape}', flush=True)
        print(f'global_signal_lr shape: {metric3_lr.shape}', flush=True)

        sizes = 256 / np.array([32, 38, 48, 64, 77, 96, 128])
        sizes = [f'{size:.2f}' for size in sizes]
        redshifts = range(14, 25)
        # reverse sizes and RMSE tensors for correct orientation
        sizes = sizes[::-1]
        metric1_sr = metric1_sr.numpy()[8:, ::-1, :]
        metric1_lr = metric1_lr.numpy()[8:, ::-1, :]
        metric2_sr = metric2_sr.numpy()[8:, ::-1, :]
        metric2_lr = metric2_lr.numpy()[8:, ::-1, :]
        metric3_sr = metric3_sr.numpy()[8:, ::-1, :]
        metric3_lr = metric3_lr.numpy()[8:, ::-1, :]
        metric3_hr = metric3_hr.numpy()[8:, ::-1, :]

        print(f'Shapes after slicing: metric_sr: {metric1_sr.shape}, RMSE_sr_dsq: {metric2_sr.shape}, metric_lr: {metric1_lr.shape}, RMSE_lr_dsq: {metric2_lr.shape}', flush=True)

        plt.rcParams.update({
            'text.usetex': True,
            'font.family': 'serif',
            'font.serif': 'cm',
            'font.size': 22,
        })
        fig, axes = plt.subplots(3, 3, figsize=(24, 24), dpi=300)

        metric1_ratio = (metric1_sr / metric1_lr).mean(axis=0).T
        metric1_sr = metric1_sr.mean(axis=0).T
        metric1_lr = metric1_lr.mean(axis=0).T
        vmin = min(metric1_sr.min(), metric1_lr.min())
        vmax = max(metric1_sr.max(), metric1_lr.max())
        vmin_ratio = metric1_ratio.min()
        vmax_ratio = metric1_ratio.max()
        plot_rmse_heatmap(
            metric1_sr,
            sizes=sizes,
            redshifts=redshifts,
            log=True,
            title=None,
            # KL divergence
            label=r'$\mathcal{{D}}_\mathrm{{KL}}(P_\mathrm{{HR}} \| P_\mathrm{{SR}})$',  # rf'$\mathrm{{RMSE}}_{{\mathrm{{SR}}}}$ [mK]',
            cmap='viridis',
            hatch=True,
            legend=True,
            fig=fig,
            ax=axes[0, 0],
            vmin=vmin,
            vmax=vmax)
        plot_rmse_heatmap(
            metric1_lr,
            sizes=sizes,
            redshifts=redshifts,
            log=True,
            title=None,
            label=r'$\mathcal{{D}}_\mathrm{{KL}}(P_\mathrm{{HR}} \| P_\mathrm{{LRI}})$',  # rf'$\mathrm{{RMSE}}_{{\mathrm{{LRI}}}}$ [mK]',
            cmap='viridis',
            hatch=False,
            fig=fig,
            ax=axes[0, 1],
            vmin=vmin,
            vmax=vmax)
        plot_rmse_heatmap(
            metric1_ratio,
            sizes=sizes,
            redshifts=redshifts,
            log=True,
            title=None,
            label=r'$\mathcal{{D}}_\mathrm{{KL}}(P_\mathrm{{HR}} \| P_\mathrm{{SR}})\ /\ \mathcal{{D}}_\mathrm{{KL}}(P_\mathrm{{HR}} \| P_\mathrm{{LRI}})$',  # rf'$\mathrm{{RMSE}}_{{\mathrm{{SR}}}}\ /\ \mathrm{{RMSE}}_{{\mathrm{{LRI}}}}$',
            cmap='coolwarm',
            hatch=True,
            fig=fig,
            ax=axes[0, 2],
            vmin=vmin_ratio,
            vmax=vmax_ratio)

        metric2_ratio = (metric2_sr / metric2_lr).mean(axis=0).T
        metric2_sr = metric2_sr.mean(axis=0).T
        metric2_lr = metric2_lr.mean(axis=0).T
        vmin_dsq = min(metric2_sr.min(), metric2_lr.min())
        vmax_dsq = max(metric2_sr.max(), metric2_lr.max())
        vmin_dsq_ratio = metric2_ratio.min()
        vmax_dsq_ratio = metric2_ratio.max()
        plot_rmse_heatmap(metric2_sr, sizes=sizes, redshifts=redshifts, title=None, label=r'$\mathrm{{RMSE}}_\mathrm{{SR}}^{{\Delta^2_{{21}}}}$ [mK$^2$]', cmap='viridis', hatch=True, fig=fig, ax=axes[1, 0], vmin=vmin_dsq, vmax=vmax_dsq)
        plot_rmse_heatmap(metric2_lr, sizes=sizes, redshifts=redshifts, title=None, label=r'$\mathrm{{RMSE}}_{{\mathrm{{LRI}}}}^{{\Delta^2_{{21}}}}$ [mK$^2$]', cmap='viridis', hatch=False, fig=fig, ax=axes[1, 1], vmin=vmin_dsq, vmax=vmax_dsq)
        plot_rmse_heatmap(metric2_ratio, sizes=sizes, redshifts=redshifts, title=None, label=r'$\mathrm{{RMSE}}_{{\mathrm{{SR}}}}^{{\Delta^2_{{21}}}}\ /\ \mathrm{{RMSE}}_{{\mathrm{{LRI}}}}^{{\Delta^2_{{21}}}}$', cmap='coolwarm', hatch=True, fig=fig, ax=axes[1, 2], vmin=vmin_dsq_ratio, vmax=vmax_dsq_ratio)

        metric3_sr = np.sqrt(np.mean(np.square(metric3_sr - metric3_hr), axis=0)).T
        metric3_lr = np.sqrt(np.mean(np.square(metric3_lr - metric3_hr), axis=0)).T
        metric3_ratio = (metric3_sr / metric3_lr)
        vmin = min(metric3_sr.min(), metric3_lr.min())
        vmax = max(metric3_sr.max(), metric3_lr.max())
        vmin_ratio = metric3_ratio.min()
        vmax_ratio = metric3_ratio.max()
        plot_rmse_heatmap(
            metric3_sr,
            sizes=sizes,
            redshifts=redshifts,
            log=True,
            title=None,
            label=r'$\mathrm{{RMSE}}_\mathrm{{SR}}^{{\bar{{T}}_{{21}}}}$ [mK]',
            cmap='viridis',
            hatch=True,
            fig=fig,
            ax=axes[2, 0],
            # vmin=vmin,
            # vmax=vmax
        )
        plot_rmse_heatmap(
            metric3_lr,
            sizes=sizes,
            redshifts=redshifts,
            log=True,
            title=None,
            label=r'$\mathrm{{RMSE}}_{{\mathrm{{LRI}}}}^{{\bar{{T}}_{{21}}}}$ [mK]',
            cmap='viridis',
            hatch=False,
            fig=fig,
            ax=axes[2, 1],
            # vmin=vmin,
            # vmax=vmax
        )
        plot_rmse_heatmap(metric3_ratio, sizes=sizes, redshifts=redshifts, log=True, title=None, label=r'$\mathrm{{RMSE}}_{{\mathrm{{SR}}}}^{{\bar{{T}}_{{21}}}}\ /\ \mathrm{{RMSE}}_{{\mathrm{{LRI}}}}^{{\bar{{T}}_{{21}}}}$', cmap='coolwarm', hatch=True, savefig=os.path.join(current_dir, 'RMSE_heatmaps.pdf'), fig=fig, ax=axes[2, 2], vmin=vmin_ratio, vmax=vmax_ratio)
        # plt.tight_layout()
        # plt.show()

    print(f'DKL_sr: {DKL_sr.mean(dim=(0))}')
    print(f'DKL_lr: {DKL_lr.mean(dim=(0))}')
    print(f'global_signal_hr: {global_signal_hr.mean(dim=(0))}')
    print(f'global_signal_sr: {global_signal_sr.mean(dim=(0))}')
    print(f'global_signal_lr: {global_signal_lr.mean(dim=(0))}')
    print(f'std_hr: {std_hr.mean(dim=(0))}')
    print(f'std_sr: {std_sr.mean(dim=(0))}')
    print(f'std_lr: {std_lr.mean(dim=(0))}')