import os
from scipy.io import loadmat
import torch
import numpy as np
import matplotlib.pyplot as plt

import SR21cm.utils as utils
from SR21cm.utils import calculate_power_spectrum
import yaml
from ASR21cm.archs import *
from ASR21cm.archs.arch_utils import *
from basicsr.utils.options import ordered_yaml
from basicsr.utils.dist_util import get_dist_info
from basicsr.archs import build_network


if __name__ == '__main__':
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    T21_hr = loadmat(os.path.join(root_dir, 'datasets/varying_IC/T21_cubes/T21_cube_z18__Npix256_IC20.mat'))['Tlin']
    T21_hr = torch.from_numpy(T21_hr).unsqueeze(0).unsqueeze(0).to(torch.float32)  # Convert to BCHWD

    delta = loadmat(os.path.join(root_dir, 'datasets/varying_IC/IC_cubes/delta_Npix256_IC20.mat'))['delta']
    delta = torch.from_numpy(delta).unsqueeze(0).unsqueeze(0).to(torch.float32)  # Convert to BCHWD format
    delta, _, _ = utils.normalize(delta, mode='standard')
    vbv = loadmat(os.path.join(root_dir, 'datasets/varying_IC/IC_cubes/vbv_Npix256_IC20.mat'))['vbv']
    vbv = torch.from_numpy(vbv).unsqueeze(0).unsqueeze(0).to(torch.float32)  # Convert to BCHWD format
    vbv, _, _ = utils.normalize(vbv, mode='standard')
    z = torch.tensor([18.,])  # Example z vector

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
    xyz_hr = xyz_hr.repeat(b, 1, 1)



    sizes = [64, 77, 96, 128]
    T21_sr = []
    T21_lr = []
    net_g.eval()
    for i, size in enumerate(sizes):
        input = torch.nn.functional.interpolate(T21_hr, size=size, mode='trilinear')
        input_mean = torch.mean(input, dim=(1, 2, 3, 4), keepdim=True)
        input_std = torch.std(input, dim=(1, 2, 3, 4), keepdim=True)
        input, _, _ = utils.normalize(input, mode='standard')
        T21_lr.append(input)

        with torch.no_grad():
            output = net_g(img_lr=input, xyz_hr=xyz_hr, delta=delta, vbv=vbv, z=z)
            input = input * input_std + input_mean
            input = torch.nn.functional.interpolate(input, size=256, mode='trilinear')

            T21_sr.append(output * input_std + input_mean)
            T21_lr.append(input)

    T21_sr = torch.stack(T21_sr, dim=0)
    T21_lr = torch.stack(T21_lr, dim=0)

    nrows = 5
    fig, axes = plt.subplots(nrows, len(sizes),) #figsize=(len(sizes) * 5, nrows * 5))
    slice_idx = 128

    k_hr, dsq_hr = calculate_power_spectrum(data_x=T21_hr, Lpix=3, kbins=100, dsq=True, method='torch', device='cpu')
    for i, size in enumerate(sizes):
        k_sr, dsq_sr = calculate_power_spectrum(data_x=T21_sr[i], Lpix=3, kbins=100, dsq=True, method='torch', device='cpu')
        k_lr, dsq_lr = calculate_power_spectrum(data_x=T21_lr[i], Lpix=3, kbins=100, dsq=True, method='torch', device='cpu')
        RMSE = torch.sqrt(torch.mean((T21_sr[i] - T21_hr) ** 2))

        axes[0, i].imshow(T21_hr[0,0,:,:, slice_idx].cpu().numpy())
        axes[0, i].set_title('HR')
        axes[1, i].imshow(T21_sr[i][0,0,:,:, slice_idx].cpu().numpy())
        axes[1, i].set_title(f'SR {size}')
        axes[2, i].imshow(T21_lr[i][0,0,:,:, slice_idx].cpu().numpy())
        axes[2, i].set_title(f'LR {size}')

        xmin = min(T21_sr[i].min(), T21_hr.min())
        xmax = max(T21_sr[i].max(), T21_hr.max())
        bins = np.linspace(xmin, xmax, 100)
        axes[3, i].hist(T21_hr.cpu().numpy().flatten(), bins=bins, alpha=0.5, label='HR', density=True)
        axes[3, i].hist(T21_sr[i].cpu().numpy().flatten(), bins=bins, alpha=0.5, label='SR', density=True)
        axes[3, i].hist(T21_lr[i].cpu().numpy().flatten(), bins=bins, alpha=0.5, label='LR', density=True)
        axes[3, i].set_xlabel(r'$T_{{21}}$ [${\rm mK}$]')

        axes[4, i].loglog(k_hr, dsq_hr[i, 0], label='$T_{{21}}$ HR', ls='solid', lw=2)
        axes[4, i].loglog(k_sr, dsq_sr[i, 0], label='$T_{{21}}$ SR', ls='solid', lw=2)
        axes[4, i].loglog(k_lr, dsq_lr[i, 0], label='$T_{{21}}$ LR', ls='dashed', lw=2)
        axes[4, i].set_xlabel('$k\\ [\\mathrm{{cMpc^{-1}}}]$')
        axes[4, i].set_ylabel(r'$\Delta^2(k)$ [${\rm mK^2}$]')
        axes[4, i].legend()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_img_path = os.path.join(current_dir, 'progressive_scaling_results.png')
    plt.savefig(save_img_path, bbox_inches='tight')
    plt.close(fig)










