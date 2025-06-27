import numpy as np
import os
import pandas as pd
import SR21cm.utils as utils
import torch
from functools import partial
from scipy.io import loadmat
import time

from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class Custom21cmDataset(torch.utils.data.Dataset):
    """Custom21cmDataset dataset.

    1. Read GT image
    2. Generate LQ (Low Quality) image with downsampling

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
    """

    def __init__(self, opt, device='cpu', phase='train'):

        self.phase = phase
        self.device = device
        self.opt = opt
        self.path_T21 = opt['dataroot_gt']
        self.path_IC = opt['dataroot_IC']

        self.redshifts = opt['redshifts']
        self.IC_seeds = opt['IC_seeds']
        self.Npix = opt['Npix']
        self.df = self.getDataFrame()
        self.dataset = self.getFullDataset()

        self.cut_factor = opt['cut_factor']
        self.scale_max = opt['scale_max']
        self.scale_min = opt['scale_min']
        self.n_augment = opt['n_augment']
        self.gt_size = opt['gt_size']
        self.conditional_cubes = opt.get('conditional_cubes', False)

    def __len__(self):
        return len(self.df)

    @torch.no_grad()
    def __getitem__(self, idx):
        assert hasattr(self, 'df'), 'DataFrame not found. Please run getDataFrame() first.'
        T21 = self.dataset.tensors[0][idx]
        delta = self.dataset.tensors[1][idx]
        vbv = self.dataset.tensors[2][idx]
        labels = self.dataset.tensors[3][idx]
        return [T21, delta, vbv, labels]

    def getDataFrame(self):
        rows = []
        for IC_seed in self.IC_seeds:
            for redshift in self.redshifts:
                row = [[
                    IC_seed,
                ], [
                    redshift,
                ]]
                for file in os.listdir(self.path_T21):
                    if ('T21_cube' in file) and (f'Npix{self.Npix}' in file):
                        z = int(file.split('z')[1].split('_')[0])
                        try:
                            IC = int(file.split('_')[7])
                        except Exception:
                            IC = int(file.split('IC')[-1].split('.mat')[0])
                        if (z == redshift) and (IC == IC_seed):
                            row.append(file)
                for file in os.listdir(self.path_IC):
                    if ('delta' in file) and (f'Npix{self.Npix}' in file):
                        IC = int(file.split('IC')[1].split('.')[0])
                        if IC == IC_seed:
                            row.append(file)
                for file in os.listdir(self.path_IC):
                    if ('vbv' in file) and (f'Npix{self.Npix}' in file):
                        IC = int(file.split('IC')[1].split('.')[0])
                        if IC == IC_seed:
                            row.append(file)
                if len(row) == 5:  # match number of columns
                    rows.append(row)
        df = pd.DataFrame(rows, columns=['IC,subcube', 'labels (z)', 'T21', 'delta', 'vbv'])

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

            label = torch.as_tensor(label, dtype=torch.float32, device='cpu').unsqueeze(0).unsqueeze(0)
            labels = torch.cat([labels, label], dim=0)

        self.dataset = torch.utils.data.TensorDataset(T21, delta, vbv, labels)

        return self.dataset


def collate_fn(batch, cut_factor, scale_min, scale_max, n_augment, gt_size, h_lr=None, conditional_cubes=False, phase='train', **kwargs):
    T21, delta, vbv, labels = zip(*batch)
    T21 = torch.concatenate(T21, dim=0).unsqueeze(1)
    # T21 = utils.get_subcubes(cubes=T21, cut_factor=cut_factor)
    if conditional_cubes:
        delta = torch.concatenate(delta, dim=0).unsqueeze(1)
        # delta = utils.get_subcubes(cubes=delta, cut_factor=cut_factor)
        vbv = torch.concatenate(vbv, dim=0).unsqueeze(1)
        # vbv = utils.get_subcubes(cubes=vbv, cut_factor=cut_factor)
        cubes = torch.cat([T21, delta, vbv], dim=1)
        cubes = random_spatial_crop(cubes, crop_size=gt_size)
        T21, delta, vbv = cubes.chunk(3, dim=1)
    else:
        T21 = random_spatial_crop(T21, crop_size=gt_size)
        # cubes = random_spatial_crop(cubes, crop_size=64)

    labels = torch.concatenate(labels, dim=0) # labels is tuple of tensors because each sample can have different astro parameters
    #b, label_dim = labels.shape
    #expansion_factor = 2**(3 * cut_factor)
    #labels = labels.repeat(1, expansion_factor)
    #labels = labels.view(b * expansion_factor, label_dim)

    b, c, h, w, d = T21.shape
    size_min = h // scale_max
    size_max = h // scale_min + 1

    if phase == 'train':
        available_sizes = torch.arange(start=size_min, end=size_max, step=1, dtype=torch.int32)

        random_idx = torch.randint(low=0, high=available_sizes.shape[0], size=(1, ))
        h_lr = available_sizes[random_idx][0].item()

        # h_lr = np.random.randint(size_min, size_max, size=1)[0]
        scale_factor = torch.full((b, 1, 1, 1, 1), h / h_lr)
        T21_lr = torch.nn.functional.interpolate(T21, size=h_lr, mode='trilinear')
        # T21, delta, vbv, T21_lr = utils.augment_dataset(T21, delta, vbv, T21_lr, n=n_augment)
        T21_lr_mean = torch.mean(T21_lr, dim=(1, 2, 3, 4), keepdim=True)
        T21_lr_std = torch.std(T21_lr, dim=(1, 2, 3, 4), keepdim=True)
        T21_lr, _, _ = utils.normalize(T21_lr, mode='standard')
        T21, _, _ = utils.normalize(T21, mode='standard', x_mean=T21_lr_mean, x_std=T21_lr_std)
        if conditional_cubes:
            delta, _, _ = utils.normalize(delta, mode='standard')
            vbv, _, _ = utils.normalize(vbv, mode='standard')

        conditional_cubes = {'delta': delta, 'vbv': vbv} if conditional_cubes else {}
    elif phase == 'val':
        if h_lr is None:
            h_lr = [h // i for i in range(2, 5)]#5)]
        elif isinstance(h_lr, int):
            h_lr = [h // h_lr]
        elif isinstance(h_lr, list):
            h_lr = h_lr
        elif h_lr == 'random':
            h_lr = np.random.randint(size_min, size_max, size=1)

        scale_factor = [torch.full((b, 1, 1, 1, 1), h / size) for size in h_lr]

        T21_lr = [torch.nn.functional.interpolate(T21, size=size, mode='trilinear') for size in h_lr]
        T21_lr_mean = [torch.mean(T21_lr_i, dim=(1, 2, 3, 4), keepdim=True) for T21_lr_i in T21_lr]
        T21_lr_std = [torch.std(T21_lr_i, dim=(1, 2, 3, 4), keepdim=True) for T21_lr_i in T21_lr]
        T21_lr = [utils.normalize(T21_lr_i, mode='standard')[0] for T21_lr_i in T21_lr]
        T21 = [utils.normalize(T21, mode='standard', x_mean=T21_lr_mean[i], x_std=T21_lr_std[i])[0] for i in range(len(h_lr))]
        labels = [labels for i in range(len(h_lr))]

        if conditional_cubes:
            delta = [utils.normalize(delta, mode='standard')[0] for i in range(len(h_lr))]
            vbv = [utils.normalize(vbv, mode='standard')[0] for i in range(len(h_lr))]

        conditional_cubes = {'delta': delta, 'vbv': vbv} if conditional_cubes else {}
    else:
        raise ValueError(f'Unknown phase: {phase}')
    # return {'lq': T21_lr, 'gt': T21, 'delta': delta, 'vbv': vbv, 'labels': labels, 'T21_lr_mean': T21_lr_mean, 'T21_lr_std': T21_lr_std, 'scale_factor': scale_factor}
    data = dict(lq=T21_lr, gt=T21, labels=labels, T21_lr_mean=T21_lr_mean, T21_lr_std=T21_lr_std, scale_factor=scale_factor, **conditional_cubes)
    return data


def create_collate_fn(opt, phase='train'):
    cut_factor = opt.get('cut_factor', 1)
    scale_min = opt.get('scale_min', 1.1)
    scale_max = opt.get('scale_max', 4.5)
    n_augment = opt.get('n_augment', 1)
    h_lr = opt.get('val_size', None)
    gt_size = opt.get('gt_size', 64)
    conditional_cubes = opt.get('conditional_cubes', False)
    return partial(collate_fn, cut_factor=cut_factor, scale_min=scale_min, scale_max=scale_max, n_augment=n_augment, h_lr=h_lr, gt_size=gt_size, conditional_cubes=conditional_cubes, phase=phase)


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


if __name__ == '__main__':

    # Example usage
    opt = {
        'dataroot_gt': '/Users/simonpochinda/Documents/PhD/dataset/varying_IC/T21_cubes/',
        'dataroot_IC': '/Users/simonpochinda/Documents/PhD/dataset/varying_IC/IC_cubes/',
        'redshifts': [
            10,
        ],
        'IC_seeds': [0, 1, 2, 3],
        'Npix': 256,
        'cut_factor': 1,
        'scale_max': 4,
        'scale_min': 1,
        'n_augment': 1,
        'one_box': False,
    }
    dataset = Custom21cmDataset(opt)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=True, collate_fn=create_collate_fn(opt=opt))
    iterator = iter(dataloader)
    batch = next(iterator)
    print(batch['lq'].shape)
    print(batch['gt'].shape)
