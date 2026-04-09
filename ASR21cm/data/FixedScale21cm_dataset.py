import glob
import os
import pandas as pd
import psutil
import torch
from scipy.io import loadmat
from torch.utils import data as data
from tqdm import tqdm

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
        self.scale = opt['scale']
        self.preload_to_gpu = opt.get('preload_to_gpu', False)
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
        # generate LR data and save stats to normalize HR
        T21_lr = torch.nn.functional.interpolate(T21, size=self.gt_size // self.scale, mode='trilinear')
        T21_lr_mean = torch.mean(T21_lr, dim=(1, 2, 3, 4), keepdim=True)
        T21_lr_std = torch.std(T21_lr, dim=(1, 2, 3, 4), keepdim=True)
        # interpolate LR data back to GT size and normalize
        T21_lr = torch.nn.functional.interpolate(T21_lr, size=self.gt_size, mode='trilinear')
        T21_lr_upsampled_mean = torch.mean(T21_lr, dim=(1, 2, 3, 4), keepdim=True)
        T21_lr_upsampled_std = torch.std(T21_lr, dim=(1, 2, 3, 4), keepdim=True)
        T21_lr = (T21_lr - T21_lr_upsampled_mean) / T21_lr_upsampled_std

        # normalize data HR to LR stats and normalize delta and vbv
        T21 = (T21 - T21_lr_mean) / T21_lr_std
        delta_mean = torch.mean(delta, dim=(1, 2, 3, 4), keepdim=True)
        delta_std = torch.std(delta, dim=(1, 2, 3, 4), keepdim=True)
        delta = (delta - delta_mean) / delta_std
        vbv_mean = torch.mean(vbv, dim=(1, 2, 3, 4), keepdim=True)
        vbv_std = torch.std(vbv, dim=(1, 2, 3, 4), keepdim=True)
        vbv = (vbv - vbv_mean) / vbv_std
        # Normalize labels to [0, 1] and ensure correct shape (scalar or 1D)
        if len(self.redshifts) > 1:
            labels = (labels - min(self.redshifts)) / (max(self.redshifts) - min(self.redshifts))
        else:
            labels = torch.tensor(0.0, dtype=torch.float32)

        # Ensure labels is 1D with shape matching expectations
        if labels.dim() == 0:  # scalar
            labels = labels.unsqueeze(0)  # make it (1,)
        elif labels.dim() > 1:
            labels = labels.squeeze()  # remove extra dimensions

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

    def estimate_memory_usage(self, check_gpu=False):
        """
        Calculate and print memory usage estimates for loading the full dataset.

        Args:
            check_gpu (bool): If True, check GPU memory instead of CPU RAM

        Returns:
            dict: Dictionary containing memory estimates in bytes and human-readable format
        """
        # Calculate memory per cube (3 fields: T21, delta, vbv)
        memory_per_cube = 3 * (self.Npix**3) * 4  # float32 = 4 bytes
        total_cubes = len(self.df)
        total_memory_needed = memory_per_cube * total_cubes

        if check_gpu:
            # Get GPU memory info
            if torch.cuda.is_available():
                gpu_free, gpu_total = torch.cuda.mem_get_info()
                available_memory = gpu_free
                total_memory = gpu_total
                memory_type = 'GPU'
                device_name = torch.cuda.get_device_name(0)
            else:
                raise RuntimeError('GPU preloading requested but CUDA is not available!')
        else:
            # Get available system memory
            available_memory = psutil.virtual_memory().available
            total_memory = psutil.virtual_memory().total
            memory_type = 'CPU RAM'
            device_name = 'System RAM'

        # Calculate safe maximum with 75% usage factor for CPU, 50% for GPU
        # (GPU needs more headroom for model, optimizer, activations)
        safety_factor = 0.5 if check_gpu else 0.75
        safe_max_memory = available_memory * safety_factor
        safe_max_cubes = int(safe_max_memory / memory_per_cube)

        # Helper function to convert bytes to human-readable format
        def bytes_to_human(bytes_val):
            for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
                if bytes_val < 1024.0:
                    return f'{bytes_val:.2f} {unit}'
                bytes_val /= 1024.0
            return f'{bytes_val:.2f} PB'

        # Print memory information
        print('\n' + '=' * 70)
        print(f'MEMORY USAGE ESTIMATE ({memory_type})')
        print('=' * 70)
        if check_gpu:
            print(f'Target device:                 {device_name}')
        print(f'Cube dimensions (Npix):        {self.Npix}³')
        print(f'Memory per cube:               {bytes_to_human(memory_per_cube)}')
        print(f'Number of cubes to load:       {total_cubes}')
        print(f'Total memory required:         {bytes_to_human(total_memory_needed)}')
        print(f'\n{memory_type} (total):            {bytes_to_human(total_memory)}')
        print(f'{memory_type} (available):        {bytes_to_human(available_memory)}')
        safety_pct = int(safety_factor * 100)
        print(f'Safe maximum ({safety_pct}% of avail):   {bytes_to_human(safe_max_memory)}')
        print(f'Estimated max loadable cubes:  {safe_max_cubes}')

        # Warning if likely to run out of memory
        if total_memory_needed > safe_max_memory:
            print(f"\n{'⚠ WARNING':^70}")
            print(f"{'':-^70}")
            print(f'Dataset size ({bytes_to_human(total_memory_needed)}) exceeds safe memory limit!')
            print('You may experience OOM errors. Consider:')
            print(f'  - Loading fewer cubes ({safe_max_cubes} max recommended)')
            if check_gpu:
                print('  - Using preload_to_gpu=False to load to CPU instead')
            else:
                print('  - Using load_full_dataset=False to read from disk')
            print('  - Reducing Npix if possible')
        else:
            memory_percentage = (total_memory_needed / available_memory) * 100
            print(f'\n✓ Dataset should fit in memory ({memory_percentage:.1f}% of available {memory_type})')

        print('=' * 70 + '\n')

        return {'memory_per_cube_bytes': memory_per_cube, 'total_cubes': total_cubes, 'total_memory_bytes': total_memory_needed, 'available_memory_bytes': available_memory, 'safe_max_cubes': safe_max_cubes, 'will_fit': total_memory_needed <= safe_max_memory, 'device': 'cuda' if check_gpu else 'cpu'}

    @torch.no_grad()
    def getFullDataset(self):
        # load full dataset into memory (CPU or GPU)

        # Determine target device
        if self.preload_to_gpu:
            if not torch.cuda.is_available():
                print('⚠ WARNING: GPU preloading requested but CUDA not available!')
                print('   Falling back to CPU loading...\n')
                device = 'cpu'
            else:
                device = 'cuda'
                print(f'Preloading dataset to GPU: {torch.cuda.get_device_name(0)}')
        else:
            device = 'cpu'

        # Estimate memory usage before loading
        mem_info = self.estimate_memory_usage(check_gpu=(device == 'cuda'))

        # If dataset won't fit, warn but proceed (or could raise error)
        if not mem_info['will_fit']:
            print('⚠ Proceeding with loading despite memory warning...')
            if device == 'cuda':
                print('   (You can set preload_to_gpu=False to load to CPU instead)')
            else:
                print('   (You can set load_full_dataset=False in config to avoid this)')
            print()

        T21 = torch.empty(0, 1, self.Npix, self.Npix, self.Npix, device=device)
        delta = torch.empty(0, 1, self.Npix, self.Npix, self.Npix, device=device)
        vbv = torch.empty(0, 1, self.Npix, self.Npix, self.Npix, device=device)
        labels = torch.empty(0, device=device)

        for index, row in tqdm(self.df.iterrows(), total=len(self.df), desc='Loading dataset'):

            T21_file = row['T21']
            delta_file = row['delta']
            vbv_file = row['vbv']
            label = row['labels (z)']

            T21_cube = loadmat(T21_file)['Tlin']
            T21_cube = torch.from_numpy(T21_cube).to(torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            T21 = torch.cat([T21, T21_cube], dim=0)

            delta_cube = loadmat(delta_file)['delta']
            delta_cube = torch.from_numpy(delta_cube).to(torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            delta = torch.cat([delta, delta_cube], dim=0)

            vbv_cube = loadmat(vbv_file)['vbv']
            vbv_cube = torch.from_numpy(vbv_cube).to(torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            vbv = torch.cat([vbv, vbv_cube], dim=0)

            # Process label: normalize and ensure correct shape
            label_tensor = torch.as_tensor(label, dtype=torch.float32, device=device)
            # Normalize to [0, 1]
            if len(self.redshifts) > 1:
                label_tensor = (label_tensor - min(self.redshifts)) / (max(self.redshifts) - min(self.redshifts))
            else:
                label_tensor = torch.tensor(0.0, dtype=torch.float32, device=device)
            # Ensure correct shape: should be (1,) for each sample
            if label_tensor.dim() == 0:
                label_tensor = label_tensor.unsqueeze(0)
            elif label_tensor.dim() > 1:
                label_tensor = label_tensor.squeeze()
            labels = torch.cat([labels, label_tensor.unsqueeze(0)], dim=0)

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
