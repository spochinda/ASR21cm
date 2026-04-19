import matplotlib.pyplot as plt
import numpy as np
import torch
from collections import OrderedDict
from os import path as osp
from SR21cm.utils import calculate_power_spectrum
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.models.sr_model import SRModel
from basicsr.utils import get_root_logger
from basicsr.utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class VAEModel(SRModel):

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        if train_opt.get('dsq_opt'):
            dsq_opt = dict(train_opt['dsq_opt'])
            dsq_opt['device'] = str(self.device)
            self.cri_dsq = build_loss(dsq_opt).to(self.device)
        else:
            self.cri_dsq = None
        self.kl_weight = train_opt.get('kl_weight', 1e-6)
        self.grad_clip = train_opt.get('grad_clip', None)
        self.accum_iter = train_opt.get('accum_iter', 1)

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use EMA with decay: {self.ema_decay}')
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weights to net_g_ema
            self.net_g_ema.eval()

        self.setup_optimizers()
        self.setup_schedulers()

    def feed_data(self, data):
        self.gt = data['gt'].to(self.device)
        if 'T21_lr_mean' in data:
            self.T21_lr_mean = data['T21_lr_mean'].to(self.device)
            self.T21_lr_std = data['T21_lr_std'].to(self.device)

    def optimize_parameters(self, current_iter):
        reconstruction, posterior = self.net_g(self.gt)
        self.output = reconstruction

        l_rec = self.cri_pix(reconstruction, self.gt)
        l_kl = posterior.kl().mean()
        l_total = l_rec + self.kl_weight * l_kl

        loss_dict = OrderedDict(l_rec=l_rec, l_kl=l_kl)
        if self.cri_dsq is not None:
            l_dsq = self.cri_dsq(reconstruction, self.gt)
            l_total = l_total + l_dsq
            loss_dict['l_dsq'] = l_dsq
        loss_dict['l_total'] = l_total

        (l_total / self.accum_iter).backward()

        if self.grad_clip is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), self.grad_clip)
            loss_dict['grad_norm'] = grad_norm

        if current_iter % self.accum_iter == 0:
            self.optimizer_g.step()
            self.optimizer_g.zero_grad()
            if self.ema_decay > 0:
                self.model_ema(decay=self.ema_decay)

        self.log_dict = self.reduce_loss_dict(loss_dict)

    def update_learning_rate(self, current_iter, warmup_iter=-1):
        # Only step the scheduler when the optimizer actually steps, so the
        # schedule advances at the same rate regardless of accum_iter.
        if current_iter % self.accum_iter == 0:
            super().update_learning_rate(current_iter // self.accum_iter, warmup_iter)

    def test(self):
        net = getattr(self, 'net_g_ema', self.net_g)
        net.eval()
        with torch.no_grad():
            self.output, self.posterior = net(self.gt)
        if net is self.net_g:
            self.net_g.train()

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['gt'] = self.gt.detach().cpu()
        out_dict['reconstruction'] = self.output.detach().cpu()
        if hasattr(self, 'T21_lr_mean'):
            out_dict['mean'] = self.T21_lr_mean.detach().cpu()
            out_dict['std'] = self.T21_lr_std.detach().cpu()
        return out_dict

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            self._initialize_best_metric_results(dataset_name)
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='cube')

        for idx, val_data in enumerate(dataloader):
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            metric_data['reconstruction'] = visuals['reconstruction']
            metric_data['gt'] = visuals['gt']

            del self.gt
            del self.output
            del self.posterior
            if hasattr(self, 'T21_lr_mean'):
                del self.T21_lr_mean
                del self.T21_lr_std
            torch.cuda.empty_cache()

            gt_cube = visuals['gt']
            rec_cube = visuals['reconstruction']
            if 'mean' in visuals:
                gt_cube = gt_cube * visuals['std'] + visuals['mean']
                rec_cube = rec_cube * visuals['std'] + visuals['mean']

            if save_img:
                with torch.no_grad():
                    rmse = ((rec_cube - gt_cube)**2).mean(dim=[1, 2, 3, 4]).sqrt()
                    k_gt, dsq_gt = calculate_power_spectrum(data_x=gt_cube, Lpix=3, kbins=100, dsq=True, method='torch', device='cpu')
                    k_rec, dsq_rec = calculate_power_spectrum(data_x=rec_cube, Lpix=3, kbins=100, dsq=True, method='torch', device='cpu')

                    b = gt_cube.shape[0]
                    nrows = 4
                    fig, axes = plt.subplots(nrows, b, figsize=(b * 5, nrows * 5))
                    if b == 1:
                        axes = np.expand_dims(axes, axis=1)

                    slice_idx = gt_cube.shape[-1] // 2
                    for i in range(b):
                        vmin = min(rec_cube[i, 0, :, :, slice_idx].min(), gt_cube[i, 0, :, :, slice_idx].min())
                        vmax = max(rec_cube[i, 0, :, :, slice_idx].max(), gt_cube[i, 0, :, :, slice_idx].max())

                        axes[0, i].imshow(gt_cube[i, 0, :, :, slice_idx].cpu().numpy(), vmin=vmin, vmax=vmax)
                        axes[0, i].set_title('GT')

                        axes[1, i].imshow(rec_cube[i, 0, :, :, slice_idx].cpu().numpy(), vmin=vmin, vmax=vmax)
                        axes[1, i].set_title(f'Rec (RMSE={rmse[i].item():.2f})')

                        xmin = min(rec_cube[i].min(), gt_cube[i].min())
                        xmax = max(rec_cube[i].max(), gt_cube[i].max())
                        bins = np.linspace(xmin, xmax, 100)
                        axes[2, i].hist(gt_cube[i].cpu().numpy().flatten(), bins=bins, alpha=0.5, label='GT', density=True)
                        axes[2, i].hist(rec_cube[i].cpu().numpy().flatten(), bins=bins, alpha=0.5, label='Rec', density=True)
                        axes[2, i].set_xlabel(r'$T_{21}$ [mK]')
                        axes[2, i].legend()

                        axes[3, i].loglog(k_gt, dsq_gt[i, 0], label='GT', ls='solid', lw=2)
                        axes[3, i].loglog(k_rec, dsq_rec[i, 0], label='Rec', ls='dashed', lw=2)
                        axes[3, i].set_xlabel(r'$k\ [\mathrm{cMpc^{-1}}]$')
                        axes[3, i].legend()

                    axes[0, 0].set_ylabel('GT')
                    axes[1, 0].set_ylabel('Reconstruction')
                    axes[2, 0].set_ylabel('PDF')
                    axes[3, 0].set_ylabel(r'$\Delta^2_{21}\ \mathrm{[mK^2]}$')

                    save_img_path = osp.join(self.opt['path']['visualization'], f'{current_iter}_{idx}.png')
                    plt.savefig(save_img_path, bbox_inches='tight')
                    plt.close(fig)

            if with_metrics:
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {idx}')

        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)
            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

        logger = get_root_logger()
        logger.info(f'Finished validation on {dataset_name}')


if __name__ == '__main__':
    import importlib.util
    import pathlib

    # Load ldm_arch directly to avoid triggering ASR21cm/__init__.py,
    # which would re-import vae_model.py and cause a double registration error.
    _arch_path = pathlib.Path(__file__).resolve().parent.parent / 'archs' / 'ldm_arch.py'
    _spec = importlib.util.spec_from_file_location('ldm_arch', _arch_path)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)

    autoencoder_opt = dict(
        double_z=True,
        z_channels=4,
        embed_dim=4,
        resolution=32,
        in_channels=1,
        out_ch=1,
        ch=8,
        ch_mult=(1, 2, 4),
        num_res_blocks=1,
        attn_resolutions=[],
        dropout=0.0,
    )

    # Minimal opt dict that mirrors what BasicSR passes to the model
    opt = {
        'is_train': True,
        'dist': False,
        'num_gpu': 0,
        'network_g': {
            'type': 'AutoencoderKL',
            'autoencoder_opt': autoencoder_opt
        },
        'path': {
            'pretrain_network_g': None,
            'strict_load_g': True,
            'resume_state': None
        },
        'train': {
            'optim_g': {
                'type': 'Adam',
                'lr': 4.5e-6
            },
            'scheduler': {
                'type': 'MultiStepLR',
                'milestones': [50000],
                'gamma': 1.0
            },
            'pixel_opt': {
                'type': 'L1Loss',
                'loss_weight': 1.0,
                'reduction': 'mean'
            },
            'kl_weight': 1e-6,
        },
        'logger': {
            'print_freq': 1
        },
    }

    device = torch.device('cpu')
    model = VAEModel(opt)

    # --- feed_data ---
    batch = {'gt': torch.randn(2, 1, 32, 32, 32)}
    model.feed_data(batch)
    assert model.gt.shape == (2, 1, 32, 32, 32), 'feed_data failed'
    print(f'feed_data OK — gt shape: {tuple(model.gt.shape)}')

    # --- optimize_parameters ---
    model.optimize_parameters(current_iter=1)
    assert 'l_rec' in model.log_dict
    assert 'l_kl' in model.log_dict
    assert 'l_total' in model.log_dict
    print('optimize_parameters OK — losses: ' + ', '.join(f'{k}={v:.6f}' for k, v in model.log_dict.items()))

    # --- test ---
    model.feed_data(batch)
    model.test()
    assert model.output.shape == (2, 1, 32, 32, 32), 'test output shape mismatch'
    print(f'test OK — output shape: {tuple(model.output.shape)}')

    # --- get_current_visuals ---
    visuals = model.get_current_visuals()
    assert 'gt' in visuals and 'reconstruction' in visuals
    assert visuals['gt'].shape == visuals['reconstruction'].shape
    print(f'get_current_visuals OK — keys: {list(visuals.keys())}')

    # --- feed_data with mean/std ---
    batch_normed = {
        'gt': torch.randn(2, 1, 32, 32, 32),
        'T21_lr_mean': torch.zeros(2, 1, 1, 1, 1),
        'T21_lr_std': torch.ones(2, 1, 1, 1, 1),
    }
    model.feed_data(batch_normed)
    assert hasattr(model, 'T21_lr_mean'), 'T21_lr_mean not set'
    visuals2 = model.get_current_visuals()
    assert 'mean' in visuals2 and 'std' in visuals2
    print(f'feed_data with mean/std OK — visuals keys: {list(visuals2.keys())}')

    print('\nAll VAEModel checks passed.')
