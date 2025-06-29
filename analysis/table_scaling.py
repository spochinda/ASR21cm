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
    load_net = torch.load(load_path, map_location='cuda')
    net_g.load_state_dict(load_net['params_ema'], strict=True)
    net_g = net_g.to(device=device)
    net_g.eval()


    sizes = [32, 38, 48, 64, 77, 96, 128]
    redshifts = range(14, 25)
    ICs = range(80)

    xyz_hr = make_coord([256, 256, 256], ranges=None, flatten=False).to(device=device)
    xyz_hr = xyz_hr.view(1, -1, 3)
    xyz_hr = xyz_hr.repeat(1, 1, 1)

    RMSE_sr = torch.empty((len(ICs), len(sizes), len(redshifts))).to(device=device)
    RMSE_sr_dsq = torch.empty((len(ICs), len(sizes), len(redshifts))).to(device=device)
    RMSE_lr = torch.empty((len(ICs), len(sizes), len(redshifts))).to(device=device)
    RMSE_lr_dsq = torch.empty((len(ICs), len(sizes), len(redshifts))).to(device=device)

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

                    k_hr, dsq_hr = calculate_power_spectrum(data_x=T21_hr, Lpix=3, kbins=100, dsq=True, method='torch', device='cuda')
                    k_sr, dsq_sr = calculate_power_spectrum(data_x=T21_sr, Lpix=3, kbins=100, dsq=True, method='torch', device='cuda')
                    k_lr, dsq_lr = calculate_power_spectrum(data_x=T21_lr, Lpix=3, kbins=100, dsq=True, method='torch', device='cuda')

                    rmse_sr = torch.sqrt(torch.mean((T21_sr - T21_hr) ** 2))
                    rmse_lr = torch.sqrt(torch.mean((T21_lr - T21_hr) ** 2))
                    RMSE_sr[i, j, k] = rmse_sr.item()
                    RMSE_lr[i, j, k] = rmse_lr.item()

                    rmse_sr_dsq = torch.sqrt(torch.nanmean((dsq_sr - dsq_hr) ** 2))
                    rmse_lr_dsq = torch.sqrt(torch.nanmean((dsq_lr - dsq_hr) ** 2))
                    RMSE_sr_dsq[i, j, k] = rmse_sr_dsq.item()
                    RMSE_lr_dsq[i, j, k] = rmse_lr_dsq.item()

                torch.cuda.empty_cache()
                progress_bar.update(1)

    idx = 3
    torch.save(RMSE_sr, os.path.join(current_dir, f'RMSE_sr_{idx}.pt'))
    torch.save(RMSE_sr_dsq, os.path.join(current_dir, f'RMSE_sr_dsq_{idx}.pt'))
    torch.save(RMSE_lr, os.path.join(current_dir, f'RMSE_lr_{idx}.pt'))
    torch.save(RMSE_lr_dsq, os.path.join(current_dir, f'RMSE_lr_dsq_{idx}.pt'))

    print(f'RMSE_sr: {RMSE_sr.mean(dim=(0))}')
    print(f'RMSE_sr_dsq: {RMSE_sr_dsq.mean(dim=(0))}')
    print(f'RMSE_lr: {RMSE_lr.mean(dim=(0))}')
    print(f'RMSE_lr_dsq: {RMSE_lr_dsq.mean(dim=(0))}')