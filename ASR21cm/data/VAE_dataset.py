import glob
import os
import pandas as pd
import torch
from scipy.io import loadmat
from torch.utils import data as data

from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class VAEDataset(data.Dataset):
    """Dataset for VAE training: loads T21 brightness temperature cubes.

    Returns batches of {'gt': cube, 'T21_lr_mean': mean, 'T21_lr_std': std}
    where gt is normalised by the cube's own mean and std.

    Supports two dataset layouts (select via config keys):
      varying_IC   — provide IC_seeds and Npix
      varying_astro — provide run_ids

    Args:
        opt (dict): Config keys:
            dataroot_gt (str): Root directory containing T21 .mat files.
            redshifts (list): Redshift values to include.
            IC_seeds (list): IC seed values (varying_IC layout).
            Npix (int): Cube side length (varying_IC layout only).
            run_ids (list): Run IDs (varying_astro layout).
            gt_size (int): Random crop size. If 0 or absent, no crop.
            load_full_dataset (bool): Pre-load all cubes into CPU RAM.
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        self.path_T21 = opt['dataroot_gt']
        self.redshifts = opt['redshifts']
        self.gt_size = opt.get('gt_size', 0)

        self.df = self._build_dataframe()

        if opt.get('load_full_dataset', False):
            print('Loading full dataset into memory...')
            self._preload()
        else:
            print('Reading data from disk...')

    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if hasattr(self, '_T21_cache'):
            T21 = self._T21_cache[idx]  # (1, H, W, D)
        else:
            T21 = self._load_cube(self.df.iloc[idx]['T21'])

        if self.gt_size:
            T21 = T21.unsqueeze(0)  # (1, 1, H, W, D) for crop helper
            T21 = random_spatial_crop(T21, self.gt_size)
            T21 = T21.squeeze(0)  # (1, gt_size, gt_size, gt_size)

        mean = T21.mean(dim=(1, 2, 3), keepdim=True)
        std = T21.std(dim=(1, 2, 3), keepdim=True).clamp(min=1e-8)
        T21 = (T21 - mean) / std

        return {'gt': T21, 'T21_lr_mean': mean, 'T21_lr_std': std}

    # ------------------------------------------------------------------

    def _load_cube(self, path):
        cube = loadmat(path)['Tlin']
        return torch.from_numpy(cube).to(torch.float32).unsqueeze(0)  # (1, H, W, D)

    def _build_dataframe(self):
        rows = []
        if 'IC_seeds' in self.opt:
            # varying_IC layout: T21_cube_z{z}__Npix{Npix}_IC{seed}.mat
            Npix = self.opt['Npix']
            for seed in self.opt['IC_seeds']:
                for z in self.redshifts:
                    matches = glob.glob(os.path.join(self.path_T21, f'T21_*z{z}_*Npix{Npix}_*IC{seed}.mat'))
                    if not matches:
                        raise FileNotFoundError(f'No T21 file for z={z}, IC={seed} in {self.path_T21}')
                    rows.append({'z': z, 'run_id': seed, 'T21': matches[0]})
        elif 'run_ids' in self.opt or 'run_id_range' in self.opt:
            # varying_astro layout: T21_cube_z{z}__diffusion_{run_id}.mat
            if 'run_id_range' in self.opt:
                start, end = self.opt['run_id_range']
                run_ids = range(start, end + 1)
            else:
                run_ids = self.opt['run_ids']
            missing = []
            for run_id in run_ids:
                for z in self.redshifts:
                    path = os.path.join(self.path_T21, f'T21_cube_z{z}__diffusion_{run_id}.mat')
                    if not os.path.exists(path):
                        missing.append(run_id)
                        continue
                    rows.append({'z': z, 'run_id': run_id, 'T21': path})
            if missing:
                print(f'Warning: skipped {len(missing)} missing run_ids: {missing}')
        else:
            raise ValueError("Dataset opt must contain either 'IC_seeds' or 'run_ids'")
        return pd.DataFrame(rows)

    @torch.no_grad()
    def _preload(self):
        cubes = []
        for _, row in self.df.iterrows():
            cubes.append(self._load_cube(row['T21']))
        self._T21_cache = cubes


def random_spatial_crop(tensor, crop_size):
    """Random cubic crop of a 5D tensor (B, C, H, W, D)."""
    assert tensor.ndim == 5, 'Input tensor must be 5D (B, C, H, W, D)'
    _, _, H, W, D = tensor.shape
    assert crop_size <= H and crop_size <= W and crop_size <= D

    h0 = torch.randint(0, H - crop_size + 1, (1, )).item()
    w0 = torch.randint(0, W - crop_size + 1, (1, )).item()
    d0 = torch.randint(0, D - crop_size + 1, (1, )).item()

    return tensor[:, :, h0:h0 + crop_size, w0:w0 + crop_size, d0:d0 + crop_size]
