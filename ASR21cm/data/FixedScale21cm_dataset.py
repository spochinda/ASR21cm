import glob
import os
import pandas as pd
import torch
from scipy.io import loadmat
from torch.utils import data as data

from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class FixedScale21cmDataset(data.Dataset):
    """Example dataset.

    1. Read GT image
    2. Generate LQ (Low Quality) image with cv2 bicubic downsampling and JPEG compression

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            io_backend (dict): IO backend type and other kwarg.
            gt_size (int): Cropped patched size for gt patches.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    Note that this dataset class only works with a batch size of 1 because of the randomly drawn scale factor.
    """

    def __init__(self, opt):
        self.opt = opt
        self.path_T21 = opt['dataroot_gt']
        self.path_IC = opt['dataroot_IC']
        self.redshifts = opt['redshifts']
        self.IC_seeds = opt['IC_seeds']
        self.Npix = opt['Npix']
        self.scale = opt.get('scale', 1)
        self.df = self.getDataFrame()
        if opt.get('load_full_dataset', False):
            print('Loading full dataset into memory...')
            self.dataset = self.getFullDataset()
        else:
            print('Reading data from disk...')
        self.gt_size = opt.get('gt_size', 128)

    def __getitem__(self, idx):
        if hasattr(self, 'dataset'):
            T21 = self.dataset.tensors[0][idx]
            delta = self.dataset.tensors[1][idx]
            vbv = self.dataset.tensors[2][idx]
            labels = self.dataset.tensors[3][idx]
        else:
            row = self.df.iloc[idx]
            T21_file = row['T21']
            delta_file = row['delta']
            vbv_file = row['vbv']
            labels = torch.tensor(row['labels (z)'], dtype=torch.float32)
            T21 = loadmat(T21_file)['Tlin']
            T21 = torch.from_numpy(T21).to(torch.float32).unsqueeze(0)
            delta = loadmat(delta_file)['delta']
            delta = torch.from_numpy(delta).to(torch.float32).unsqueeze(0)
            vbv = loadmat(vbv_file)['vbv']
            vbv = torch.from_numpy(vbv).to(torch.float32).unsqueeze(0)

        # random crop
        T21 = T21.unsqueeze(0)
        delta = delta.unsqueeze(0)
        vbv = vbv.unsqueeze(0)
        cubes = torch.cat([T21, delta, vbv], dim=0)
        cubes = random_spatial_crop(cubes, crop_size=self.gt_size)
        T21, delta, vbv = cubes.chunk(3, dim=0)

        # generate LR data
        T21_lr = torch.nn.functional.interpolate(T21, size=self.gt_size // self.scale, mode='trilinear')

        # normalize data
        T21_lr_mean = torch.mean(T21_lr, dim=(1, 2, 3, 4), keepdim=True)
        T21_lr_std = torch.std(T21_lr, dim=(1, 2, 3, 4), keepdim=True)
        T21_lr = (T21_lr - T21_lr_mean) / T21_lr_std
        T21 = (T21 - T21_lr_mean) / T21_lr_std
        delta_mean = torch.mean(delta, dim=(1, 2, 3, 4), keepdim=True)
        delta_std = torch.std(delta, dim=(1, 2, 3, 4), keepdim=True)
        delta = (delta - delta_mean) / delta_std
        vbv_mean = torch.mean(vbv, dim=(1, 2, 3, 4), keepdim=True)
        vbv_std = torch.std(vbv, dim=(1, 2, 3, 4), keepdim=True)
        vbv = (vbv - vbv_mean) / vbv_std
        labels = (labels - min(self.redshifts)) / (max(self.redshifts) - min(self.redshifts)) if len(self.redshifts) > 1 else torch.tensor(0.0)

        # interpolate LR data back to GT size
        T21_lr = torch.nn.functional.interpolate(T21_lr, size=self.gt_size, mode='trilinear')

        # remove redundant dimension
        T21 = T21.squeeze(0)
        T21_lr = T21_lr.squeeze(0)
        delta = delta.squeeze(0)
        vbv = vbv.squeeze(0)
        T21_lr_mean = T21_lr_mean.squeeze(0)
        T21_lr_std = T21_lr_std.squeeze(0)

        data = dict(lq=T21_lr, gt=T21, labels=labels, T21_lr_mean=T21_lr_mean, T21_lr_std=T21_lr_std, delta=delta, vbv=vbv)

        return data

    def __len__(self):
        return len(self.df)

    def getDataFrame(self):
        rows = []
        for IC_seed in self.IC_seeds:
            for redshift in self.redshifts:
                row = [
                    IC_seed,
                    [redshift],
                ]
                T21_files = glob.glob(os.path.join(self.path_T21, f'T21_*z{redshift}_*Npix{self.Npix}_*IC{IC_seed}.mat'))
                row.append(T21_files[0])

                delta_files = glob.glob(os.path.join(self.path_IC, f'delta_Npix{self.Npix}_*IC{IC_seed}.mat'))
                row.append(delta_files[0])

                vbv_files = glob.glob(os.path.join(self.path_IC, f'vbv_Npix{self.Npix}_*IC{IC_seed}.mat'))
                row.append(vbv_files[0])

                rows.append(row)

        df = pd.DataFrame(rows, columns=['IC', 'labels (z)', 'T21', 'delta', 'vbv'])

        return df

    @torch.no_grad()
    def getFullDataset(self):
        # load full dataset into CPU memory

        T21 = torch.empty(0, 1, self.Npix, self.Npix, self.Npix, device='cpu')
        delta = torch.empty(0, 1, self.Npix, self.Npix, self.Npix, device='cpu')
        vbv = torch.empty(0, 1, self.Npix, self.Npix, self.Npix, device='cpu')
        labels = torch.empty(0, device='cpu')

        for index, row in self.df.iterrows():

            T21_file = os.path.join(self.path_T21, row['T21'])
            delta_file = os.path.join(self.path_IC, row['delta'])
            vbv_file = os.path.join(self.path_IC, row['vbv'])
            label = row['labels (z)']

            T21_cube = loadmat(T21_file)['Tlin']
            T21_cube = torch.from_numpy(T21_cube).to(torch.float32).unsqueeze(0).unsqueeze(0)
            T21 = torch.cat([T21, T21_cube], dim=0)

            delta_cube = loadmat(delta_file)['delta']
            delta_cube = torch.from_numpy(delta_cube).to(torch.float32).unsqueeze(0).unsqueeze(0)
            delta = torch.cat([delta, delta_cube], dim=0)

            vbv_cube = loadmat(vbv_file)['vbv']
            vbv_cube = torch.from_numpy(vbv_cube).to(torch.float32).unsqueeze(0).unsqueeze(0)
            vbv = torch.cat([vbv, vbv_cube], dim=0)

            label = torch.as_tensor(label, dtype=torch.float32, device='cpu').unsqueeze(0)
            labels = torch.cat([labels, label], dim=0)

        self.dataset = torch.utils.data.TensorDataset(T21, delta, vbv, labels)

        return self.dataset


def random_spatial_crop(tensor, crop_size):
    """
    Perform a random crop on a 5D tensor (B, C, H, W, D).

    Args:
        tensor (torch.Tensor): Input tensor of shape (B, C, H, W, D).
        crop_size (int): Size of the cube to crop along spatial dimensions.

    Returns:
        torch.Tensor: Cropped tensor of shape (B, C, crop_size, crop_size, crop_size).
    """
    assert tensor.ndim == 5, 'Input tensor must be 5D (B, C, H, W, D)'
    B, C, H, W, D = tensor.shape
    assert crop_size <= H and crop_size <= W and crop_size <= D, 'Crop size must be <= each spatial dimension'

    # Random starting indices for each spatial dimension
    h_start = torch.randint(0, H - crop_size + 1, (1, )).item()
    w_start = torch.randint(0, W - crop_size + 1, (1, )).item()
    d_start = torch.randint(0, D - crop_size + 1, (1, )).item()

    return tensor[:, :, h_start:h_start + crop_size, w_start:w_start + crop_size, d_start:d_start + crop_size]
