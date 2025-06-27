import matplotlib.pyplot as plt
import numpy as np
import os.path as osp
import torch
from collections import OrderedDict
from SR21cm.utils import calculate_power_spectrum
from tqdm import tqdm

from ASR21cm.archs.arch_utils import make_coord
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.models.srgan_model import SRGANModel
from basicsr.utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class ESRASRGANModel(SRGANModel):
    """ESRGAN model for single image super-resolution."""

    def init_training_settings(self):
        super(ESRASRGANModel, self).init_training_settings()
        train_opt = self.opt['train']

        self.net_g_iters = train_opt.get('net_g_iters', 1)
        self.net_g_init_iters = train_opt.get('net_g_init_iters', 0)

        # dsq loss
        if train_opt.get('dsq_opt'):
            self.cri_dsq = build_loss(train_opt['dsq_opt']).to(self.device)
        else:
            self.cri_dsq = None

        assert self.opt['network_g']['encoder_opt'].get('redshift_embedding', False) == self.opt['network_d'].get('redshift_embedding', False), \
            'Redshift embedding should be the same for both generator and discriminator.'

    def optimize_parameters(self, current_iter):
        # optimize net_g
        b, c, h, w, d = self.gt.shape
        xyz_hr = make_coord([h, h, h], ranges=None, flatten=True)
        xyz_hr = xyz_hr.view(1, -1, 3).float()
        xyz_hr = xyz_hr.repeat(b, 1, 1)

        for p in self.net_d.parameters():
            p.requires_grad = False

        self.optimizer_g.zero_grad()

        kwargs = {'z': self.z} if self.opt['network_g']['encoder_opt'].get('redshift_embedding', False) else {}
        if hasattr(self, 'delta'):
            kwargs['delta'] = self.delta
        if hasattr(self, 'vbv'):
            kwargs['vbv'] = self.vbv

        output = self.net_g(self.lq, xyz_hr, **kwargs)
        self.output = output[0]
        feature_map = output[1]
        if False:  # current_iter == 100:

            fig, axes = plt.subplots(2, 2, figsize=(10, 5))
            sr_clone = self.output[0, 0, :, :, d // 2].clone().detach().cpu().numpy()
            hr_clone = self.gt[0, 0, :, :, d // 2].clone().detach().cpu().numpy()
            d_lr = self.lq.shape[-1] // 2
            lr_clone = self.lq[0, 0, :, :, d_lr].clone().detach().cpu().numpy()
            minimum = min(sr_clone.min(), hr_clone.min(), lr_clone.min())
            maximum = max(sr_clone.max(), hr_clone.max(), lr_clone.max())
            axes[0, 0].imshow(hr_clone, vmin=minimum, vmax=maximum)
            axes[0, 0].set_title('HR')
            axes[0, 1].imshow(lr_clone, vmin=minimum, vmax=maximum)
            axes[0, 1].set_title('LR')
            axes[1, 0].imshow(sr_clone, vmin=minimum, vmax=maximum)
            axes[1, 0].set_title('SR')
            bins = np.linspace(minimum, maximum, 100)
            n, bins, edges = axes[1, 1].hist(hr_clone.flatten(), bins=bins, alpha=0.5, label='HR', density=True)
            axes[1, 1].hist(sr_clone.flatten(), bins=bins, alpha=0.5, label='SR', density=True)
            axes[1, 1].hist(lr_clone.flatten(), bins=bins, alpha=0.5, label='LR', density=True)
            axes[1, 1].set_xlabel(r'$T_{{21}}$')
            axes[1, 1].legend()
            axes[1, 1].set_ylim(0, max(n) * 1.1)
            save_img_path = osp.join(self.opt['path']['visualization'], f'input_output_{current_iter}.png')
            plt.savefig(save_img_path, bbox_inches='tight')
            plt.close(fig)

            fig, axes = plt.subplots(
                feature_map.shape[1],
                3,
                figsize=(5 * 2, feature_map.shape[1] * 5),
                width_ratios=[1, 1, 0.1],
                sharex='col',
            )
            for i in range(feature_map.shape[1]):
                fb, fc, fh, fw, fd = feature_map.shape
                feature_map_clone = feature_map[0, i, :, :, fd // 2].clone().detach().cpu().numpy()
                im = axes[i, 1].imshow(feature_map_clone)
                axes[i, 1].set_title(f'Feature map {i}')
                fig.colorbar(im, cax=axes[i, 2], orientation='vertical')
                axes[i, 0].hist(feature_map_clone.flatten(), bins=100, density=True)
                axes[i, 0].set_xlabel('Feature map value')
                axes[i, 0].set_ylabel('Density')

            save_img_path = osp.join(self.opt['path']['visualization'], f'feature_map_{current_iter}.png')
            plt.savefig(save_img_path, bbox_inches='tight')
            plt.close(fig)

        l_g_total = 0
        loss_dict = OrderedDict()
        if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):
            # pixel loss
            if self.cri_pix:
                l_g_pix = self.cri_pix(self.output, self.gt)
                l_g_total += l_g_pix
                loss_dict['l_g_pix'] = l_g_pix
            # perceptual loss
            if self.cri_perceptual:
                l_g_percep, l_g_style = self.cri_perceptual(self.output, self.gt)
                if l_g_percep is not None:
                    l_g_total += l_g_percep
                    loss_dict['l_g_percep'] = l_g_percep
                if l_g_style is not None:
                    l_g_total += l_g_style
                    loss_dict['l_g_style'] = l_g_style
            # dsq loss
            if self.cri_dsq:
                l_g_dsq = self.cri_dsq(self.output, self.gt)
                l_g_total += l_g_dsq
                loss_dict['l_g_dsq'] = l_g_dsq

            # gan loss (relativistic gan)
            real_d_pred = self.net_d(self.gt, **kwargs).detach()
            fake_g_pred = self.net_d(self.output, **kwargs)
            l_g_real = self.cri_gan(real_d_pred - torch.mean(fake_g_pred), False, is_disc=False)
            l_g_fake = self.cri_gan(fake_g_pred - torch.mean(real_d_pred), True, is_disc=False)
            l_g_gan = (l_g_real + l_g_fake) / 2

            l_g_total += l_g_gan
            loss_dict['l_g_gan'] = l_g_gan

            l_g_total.backward()
            self.optimizer_g.step()

        # optimize net_d
        if (current_iter % self.net_g_iters == 0 and current_iter > self.net_g_init_iters):
            for p in self.net_d.parameters():
                p.requires_grad = True

            self.optimizer_d.zero_grad()
            # gan loss (relativistic gan)

            # In order to avoid the error in distributed training:
            # "Error detected in CudnnBatchNormBackward: RuntimeError: one of
            # the variables needed for gradient computation has been modified by
            # an inplace operation",
            # we separate the backwards for real and fake, and also detach the
            # tensor for calculating mean.

            # real
            fake_d_pred = self.net_d(self.output, **kwargs).detach()
            real_d_pred = self.net_d(self.gt, **kwargs)
            l_d_real = self.cri_gan(real_d_pred - torch.mean(fake_d_pred), True, is_disc=True) * 0.5
            l_d_real.backward()
            # fake
            fake_d_pred = self.net_d(self.output.detach(), **kwargs)
            l_d_fake = self.cri_gan(fake_d_pred - torch.mean(real_d_pred.detach()), False, is_disc=True) * 0.5
            l_d_fake.backward()
            self.optimizer_d.step()

            loss_dict['l_d_real'] = l_d_real
            loss_dict['l_d_fake'] = l_d_fake
            loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
            loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
        # else:
        #    loss_dict['l_d_real'] = torch.tensor(0.0, device=self.device)
        #    loss_dict['l_d_fake'] = torch.tensor(0.0, device=self.device)
        #    loss_dict['out_d_real'] = torch.tensor(0.0, device=self.device)
        #    loss_dict['out_d_fake'] = torch.tensor(0.0, device=self.device)

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='cube')

        for idx, val_data in enumerate(dataloader):
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()

            metric_data['sr'] = visuals['sr']
            metric_data['hr'] = visuals['hr']
            metric_data['mean'] = visuals['mean']
            metric_data['std'] = visuals['std']
            # metric_data['scale_factor'] = visuals['scale_factor']
            # metric_data['labels'] = visuals['labels']
            # metric_data['z'] = visuals['z']
            # metric_data['astro_params'] = visuals['astro_params']

            # tentative for out of GPU memory
            del self.gt
            del self.lq
            del self.output
            del self.T21_lr_mean
            del self.T21_lr_std
            del self.scale_factor
            del self.labels
            del self.z
            del self.astro_params
            del self.delta
            del self.vbv

            torch.cuda.empty_cache()

            # scale back to mK
            visuals['sr'] = visuals['sr'] * visuals['std'] + visuals['mean']
            visuals['hr'] = visuals['hr'] * visuals['std'] + visuals['mean']

            if save_img:
                with torch.no_grad():
                    sizes = torch.unique(visuals['scale_factor'])
                    skip = len(visuals['scale_factor']) // len(sizes)

                    hr_cube = visuals['hr'][::skip]
                    sr_cube = visuals['sr'][::skip]
                    scale_factor = visuals['scale_factor'][::skip]

                    rmse = ((sr_cube - hr_cube)**2).mean(dim=[1, 2, 3, 4]).sqrt()
                    k_hr, dsq_hr = calculate_power_spectrum(data_x=hr_cube, Lpix=3, kbins=100, dsq=True, method='torch', device='cpu')
                    k_sr, dsq_sr = calculate_power_spectrum(data_x=sr_cube, Lpix=3, kbins=100, dsq=True, method='torch', device='cpu')

                    nrows = 4
                    b, c, h, w, d = sr_cube.shape
                    fig, axes = plt.subplots(nrows, b, figsize=(b * 5, nrows * 5))
                    if b == 1:
                        axes = np.expand_dims(axes, axis=0)
                        axes = axes.reshape(nrows, b)

                    slice_idx = d // 2
                    for i in range(b):
                        z = visuals['z'][i].item() if isinstance(visuals['z'], torch.Tensor) else visuals['z'][i][0].item()
                        vmin = min(sr_cube[i, :, :, slice_idx].min(), hr_cube[i, :, :, slice_idx].min())
                        vmax = max(sr_cube[i, :, :, slice_idx].max(), hr_cube[i, :, :, slice_idx].max())
                        rmse_i = rmse[i].item()
                        scale_i = scale_factor[i].item()

                        axes[0, i].imshow(hr_cube[i, 0, :, :, slice_idx].cpu().numpy(), vmin=vmin, vmax=vmax)
                        axes[0, i].set_title('HR')

                        axes[1, i].imshow(sr_cube[i, 0, :, :, slice_idx].cpu().numpy(), vmin=vmin, vmax=vmax)
                        axes[1, i].set_title('SR')

                        xmin = min(sr_cube[i].min(), hr_cube[i].min())
                        xmax = max(sr_cube[i].max(), hr_cube[i].max())
                        bins = np.linspace(xmin, xmax, 100)
                        axes[2, i].hist(hr_cube[i].cpu().numpy().flatten(), bins=bins, alpha=0.5, label='HR', density=True)
                        axes[2, i].hist(sr_cube[i].cpu().numpy().flatten(), bins=bins, alpha=0.5, label='SR', density=True)
                        # axes[2, i].legend(title=f'RMSE: {rmse[i]:.2f}, scale: {self.scale_factor:.2f}')
                        axes[2, i].set_xlabel(r'$T_{{21}}$ [${\rm mK}$]')

                        axes[3, i].loglog(k_hr, dsq_hr[i, 0], label='$T_{{21}}$ HR', ls='solid', lw=2)
                        axes[3, i].loglog(k_sr, dsq_sr[i, 0], label='$T_{{21}}$ SR', ls='solid', lw=2)
                        axes[3, i].set_xlabel('$k\\ [\\mathrm{{cMpc^{-1}}}]$')
                        axes[3, i].legend(title=f'RMSE: {rmse_i:.2f}, \nscale: {scale_i:.2f}, \nz: {z:.1f}', loc='lower right')

                    axes[0, 0].set_ylabel('HR')
                    axes[1, 0].set_ylabel('SR')
                    axes[2, 0].set_ylabel('PDF')
                    axes[3, 0].set_ylabel('$\\Delta^2_{{21}}\\ \\mathrm{{[mK^2]}}$ ')

                    save_img_path = osp.join(self.opt['path']['visualization'], f'{current_iter}_z_{z:.0f}.png')
                    plt.savefig(save_img_path, bbox_inches='tight')
                    plt.close(fig)

                    # if self.opt['is_train']:
                    #    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                    #                             f'{img_name}_{current_iter}.png')
                    # else:
                    #    if self.opt['val']['suffix']:
                    #        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                    #                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    #    else:
                    #        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                    #                                 f'{img_name}_{self.opt["name"]}.png')
                    # imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
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
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def test(self):
        b, c, h, w, d = self.gt.shape if isinstance(self.gt, torch.Tensor) else self.gt[0].shape
        xyz_hr = make_coord([h, h, h], ranges=None, flatten=False)
        xyz_hr = xyz_hr.view(1, -1, 3)
        xyz_hr = xyz_hr.repeat(b, 1, 1).to(self.device)
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                if isinstance(self.lq, torch.Tensor):
                    with torch.amp.autocast(device_type=self.device.type, enabled=False):
                        kwargs = {'z': self.z} if self.opt['network_g']['encoder_opt'].get('redshift_embedding', False) else {}
                        if hasattr(self, 'delta'):
                            kwargs['delta'] = self.delta
                        if hasattr(self, 'vbv'):
                            kwargs['vbv'] = self.vbv
                        output = self.net_g_ema(self.lq, xyz_hr, **kwargs)
                        self.output = output[0]
                        # feature_map = output[1]
                elif isinstance(self.lq, list):
                    # tqdm loop verbose argument
                    with torch.amp.autocast(device_type=self.device.type, enabled=False):
                        self.output = []
                        for i,lq in tqdm(enumerate(self.lq), total=len(self.lq), disable=True): # not self.opt['val'].get('pbar', False), desc='testing scales'):
                            kwargs = {'z': self.z[i]} if self.opt['network_g']['encoder_opt'].get('redshift_embedding', False) else {}
                            if hasattr(self, 'delta'):
                                kwargs['delta'] = self.delta[i]
                            if hasattr(self, 'vbv'):
                                kwargs['vbv'] = self.vbv[i]
                            output = self.net_g_ema(lq, xyz_hr, **kwargs)
                            self.output.append(output[0])
                            # feature_map = output[1]
                            # self.output.append(self.net_g_ema(lq, xyz_hr))
        else:
            self.net_g.eval()
            with torch.no_grad():
                if isinstance(self.lq, torch.Tensor):
                    with torch.amp.autocast(device_type=self.device.type, enabled=False):
                        kwargs = {'z': self.z} if self.opt['network_g']['encoder_opt'].get('redshift_embedding', False) else {}
                        if hasattr(self, 'delta'):
                            kwargs['delta'] = self.delta
                        if hasattr(self, 'vbv'):
                            kwargs['vbv'] = self.vbv
                        output = self.net_g(self.lq, xyz_hr, **kwargs)
                        self.output = output[0]
                        # feature_map = output[1]
                elif isinstance(self.lq, list):
                    with torch.amp.autocast(device_type=self.device.type, enabled=False):
                        self.output = []
                        for i,lq in tqdm(enumerate(self.lq), total=len(self.lq), disable=True): # not self.opt['val'].get('pbar', False), desc='testing scales'):
                            kwargs = {'z': self.z[i]} if self.opt['network_g']['encoder_opt'].get('redshift_embedding', False) else {}
                            if hasattr(self, 'delta'):
                                kwargs['delta'] = self.delta[i]
                            if hasattr(self, 'vbv'):
                                kwargs['vbv'] = self.vbv[i]
                            output = self.net_g(lq, xyz_hr, **kwargs)
                            self.output.append(output[0])
                            # feature_map = output[1]
            self.net_g.train()

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device) if isinstance(data['lq'], torch.Tensor) else [lq.to(self.device) for lq in data['lq']]
        self.gt = data['gt'].to(self.device) if isinstance(data['gt'], torch.Tensor) else [gt.to(self.device) for gt in data['gt']]
        self.T21_lr_mean = data['T21_lr_mean'].to(self.device) if isinstance(data['T21_lr_mean'], torch.Tensor) else [mean.to(self.device) for mean in data['T21_lr_mean']]
        self.T21_lr_std = data['T21_lr_std'].to(self.device) if isinstance(data['T21_lr_std'], torch.Tensor) else [std.to(self.device) for std in data['T21_lr_std']]
        self.scale_factor = data['scale_factor'].to(self.device) if isinstance(data['scale_factor'], torch.Tensor) else [scale.to(self.device) for scale in data['scale_factor']]
        self.labels = data['labels'].to(self.device) if isinstance(data['labels'], torch.Tensor) else [label.to(self.device) for label in data['labels']]
        self.z = self.labels[:,0].to(self.device) if isinstance(self.labels, torch.Tensor) else [label[:, 0].to(self.device) for label in self.labels]
        self.astro_params = self.labels[:,1:].to(self.device) if isinstance(self.labels, torch.Tensor) else [label[:, 1:].to(self.device) for label in self.labels]
        if 'delta' in data.keys(): # if not data.get('delta', None) is None:
            self.delta = data['delta'].to(self.device) if isinstance(data['delta'], torch.Tensor) else [delta.to(self.device) for delta in data['delta']]
        else:
            self.delta = None
        if 'vbv' in data.keys():
            self.vbv = data['vbv'].to(self.device) if isinstance(data['vbv'], torch.Tensor) else [vbv.to(self.device) for vbv in data['vbv']]
        else:
            self.vbv = None

    def get_current_visuals(self):
        out_dict = OrderedDict()
        # out_dict['lq'] = self.lq.detach().cpu() if isinstance(self.lq, torch.Tensor) else [lq.detach().cpu() for lq in self.lq]
        out_dict['sr'] = self.output.detach().cpu() if isinstance(self.output, torch.Tensor) else torch.cat(self.output, dim=0).detach().cpu()
        out_dict['hr'] = self.gt.detach().cpu() if isinstance(self.gt, torch.Tensor) else torch.cat(self.gt, dim=0).detach().cpu()
        out_dict['mean'] = self.T21_lr_mean.detach().cpu() if isinstance(self.T21_lr_mean, torch.Tensor) else torch.cat(self.T21_lr_mean, dim=0).detach().cpu()
        out_dict['std'] = self.T21_lr_std.detach().cpu() if isinstance(self.T21_lr_std, torch.Tensor) else torch.cat(self.T21_lr_std, dim=0).detach().cpu()
        out_dict['scale_factor'] = self.scale_factor if isinstance(self.scale_factor, torch.Tensor) else torch.cat(self.scale_factor, dim=0).detach().cpu()
        out_dict['labels'] = self.labels.detach().cpu() if isinstance(self.labels, torch.Tensor) else torch.cat(self.labels, dim=0).detach().cpu()
        out_dict['z'] = self.z.detach().cpu() if isinstance(self.z, torch.Tensor) else torch.cat(self.z, dim=0).detach().cpu()
        out_dict['astro_params'] = self.astro_params.detach().cpu() if isinstance(self.astro_params, torch.Tensor) else torch.cat(self.astro_params, dim=0).detach().cpu()
        return out_dict
