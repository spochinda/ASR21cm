import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
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


def _weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)


def _hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(torch.nn.functional.relu(1. - logits_real))
    loss_fake = torch.mean(torch.nn.functional.relu(1. + logits_fake))
    return 0.5 * (loss_real + loss_fake)


def _vanilla_d_loss(logits_real, logits_fake):
    return 0.5 * (torch.mean(torch.nn.functional.softplus(-logits_real)) + torch.mean(torch.nn.functional.softplus(logits_fake)))


def _adopt_weight(weight, global_step, threshold=0, value=0.0):
    if global_step < threshold:
        weight = value
    return weight


class _LogVarModule(nn.Module):
    """Tiny trainable module holding a single scalar log-variance parameter."""

    def __init__(self, init=0.0):
        super().__init__()
        self.logvar = nn.Parameter(torch.ones(size=()) * init)


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

        # --- optional learned log-variance ---
        self.use_logvar = train_opt.get('use_logvar', False)
        if self.use_logvar:
            logvar_init = train_opt.get('logvar_init', 0.0)
            self.net_logvar = _LogVarModule(init=logvar_init).to(self.device)
            self.net_logvar.train()

        # --- optional PatchGAN discriminator ---
        self.net_d = None
        if self.opt.get('network_d') is not None:
            self.net_d = build_network(self.opt['network_d']).to(self.device)
            self.net_d.apply(_weights_init)
            self.net_d.train()

            self.disc_start = train_opt.get('disc_start', 0)
            self.disc_factor = train_opt.get('disc_factor', 1.0)
            self.disc_weight = train_opt.get('disc_weight', 1.0)
            self.use_adaptive_weight = train_opt.get('use_adaptive_weight', True)

            disc_loss_type = train_opt.get('disc_loss', 'hinge')
            if disc_loss_type == 'hinge':
                self.disc_loss_fn = _hinge_d_loss
            elif disc_loss_type == 'vanilla':
                self.disc_loss_fn = _vanilla_d_loss
            else:
                raise ValueError(f'Unknown disc_loss type: {disc_loss_type}')

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use EMA with decay: {self.ema_decay}')
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)
            self.net_g_ema.eval()

        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']

        # generator optimizer — also includes logvar if enabled
        optim_params_g = list(self.net_g.parameters())
        if self.use_logvar:
            optim_params_g += list(self.net_logvar.parameters())
        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params_g, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

        if self.net_d is not None:
            optim_type_d = train_opt['optim_d'].pop('type')
            self.optimizer_d = self.get_optimizer(optim_type_d, self.net_d.parameters(), **train_opt['optim_d'])
            self.optimizers.append(self.optimizer_d)

    def calculate_adaptive_weight(self, rec_loss, g_loss, last_layer):
        rec_grads = torch.autograd.grad(rec_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        d_weight = torch.norm(rec_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        return d_weight * self.disc_weight

    def feed_data(self, data):
        self.gt = data['gt'].to(self.device)
        if 'T21_lr_mean' in data:
            self.T21_lr_mean = data['T21_lr_mean'].to(self.device)
            self.T21_lr_std = data['T21_lr_std'].to(self.device)

    def optimize_parameters(self, current_iter):
        reconstruction, posterior = self.net_g(self.gt)
        self.output = reconstruction

        loss_dict = OrderedDict()

        # ------------------------------------------------------------------ #
        # Generator / encoder update                                           #
        # ------------------------------------------------------------------ #
        if self.net_d is not None:
            for p in self.net_d.parameters():
                p.requires_grad = False

        l_rec = self.cri_pix(reconstruction, self.gt)

        if self.use_logvar:
            logvar = self.net_logvar.logvar
            rec_loss_for_weight = torch.mean(l_rec / torch.exp(logvar) + logvar)
        else:
            rec_loss_for_weight = l_rec

        l_kl = posterior.kl().mean()
        l_g_total = rec_loss_for_weight + self.kl_weight * l_kl
        loss_dict['l_rec'] = l_rec
        loss_dict['l_kl'] = l_kl

        if self.cri_dsq is not None:
            l_dsq = self.cri_dsq(reconstruction, self.gt)
            l_g_total = l_g_total + l_dsq
            loss_dict['l_dsq'] = l_dsq

        if self.net_d is not None:
            disc_factor = _adopt_weight(self.disc_factor, current_iter // self.accum_iter, threshold=self.disc_start)
            logits_fake = self.net_d(reconstruction.contiguous())
            g_loss = -torch.mean(logits_fake)

            if disc_factor > 0.0:
                if self.use_adaptive_weight:
                    try:
                        d_weight = self.calculate_adaptive_weight(rec_loss_for_weight, g_loss, last_layer=self.net_g.get_last_layer())
                    except RuntimeError:
                        assert not self.training
                        d_weight = torch.tensor(0.0)
                else:
                    d_weight = self.disc_weight
            else:
                d_weight = torch.tensor(0.0)

            l_g_gan = d_weight * disc_factor * g_loss
            l_g_total = l_g_total + l_g_gan
            loss_dict['l_g_gan'] = l_g_gan
            loss_dict['d_weight'] = d_weight if isinstance(d_weight, torch.Tensor) else torch.tensor(d_weight)

        loss_dict['l_total'] = l_g_total
        (l_g_total / self.accum_iter).backward()

        if self.grad_clip is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), self.grad_clip)
            loss_dict['grad_norm'] = grad_norm

        if current_iter % self.accum_iter == 0:
            self.optimizer_g.step()
            self.optimizer_g.zero_grad()
            if self.ema_decay > 0:
                self.model_ema(decay=self.ema_decay)

        # ------------------------------------------------------------------ #
        # Discriminator update                                                 #
        # ------------------------------------------------------------------ #
        if self.net_d is not None:
            for p in self.net_d.parameters():
                p.requires_grad = True

            logits_real = self.net_d(self.gt.contiguous().detach())
            logits_fake = self.net_d(reconstruction.contiguous().detach())
            d_loss = disc_factor * self.disc_loss_fn(logits_real, logits_fake)
            (d_loss / self.accum_iter).backward()

            if current_iter % self.accum_iter == 0:
                self.optimizer_d.step()
                self.optimizer_d.zero_grad()

            loss_dict['l_d'] = d_loss
            loss_dict['out_d_real'] = logits_real.detach().mean()
            loss_dict['out_d_fake'] = logits_fake.detach().mean()

        self.log_dict = self.reduce_loss_dict(loss_dict)

    def update_learning_rate(self, current_iter, warmup_iter=-1):
        if current_iter % self.accum_iter == 0:
            super().update_learning_rate(current_iter // self.accum_iter, warmup_iter)

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        if self.net_d is not None:
            self.save_network(self.net_d, 'net_d', current_iter)
        if self.use_logvar:
            self.save_network(self.net_logvar, 'net_logvar', current_iter)
        self.save_training_state(epoch, current_iter)

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
            metric_data['sr'] = visuals['reconstruction']
            metric_data['hr'] = visuals['gt']

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

    base_opt = {
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
        'logger': {
            'print_freq': 1
        },
    }

    base_train_opt = {
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
    }

    device = torch.device('cpu')
    batch = {'gt': torch.randn(2, 1, 32, 32, 32)}

    # ------------------------------------------------------------------ #
    # Test 1: plain VAE (no GAN, no logvar)                               #
    # ------------------------------------------------------------------ #
    import copy
    opt1 = copy.deepcopy(base_opt)
    opt1['train'] = copy.deepcopy(base_train_opt)
    model1 = VAEModel(opt1)
    model1.feed_data(batch)
    model1.optimize_parameters(current_iter=1)
    assert 'l_rec' in model1.log_dict and 'l_kl' in model1.log_dict
    print('Test 1 (plain VAE) OK — losses: ' + ', '.join(f'{k}={v:.4f}' for k, v in model1.log_dict.items()))

    # ------------------------------------------------------------------ #
    # Test 2: VAE + logvar                                                #
    # ------------------------------------------------------------------ #
    opt2 = copy.deepcopy(base_opt)
    opt2['train'] = copy.deepcopy(base_train_opt)
    opt2['train']['use_logvar'] = True
    model2 = VAEModel(opt2)
    model2.feed_data(batch)
    model2.optimize_parameters(current_iter=1)
    assert 'l_rec' in model2.log_dict
    print('Test 2 (VAE + logvar) OK — losses: ' + ', '.join(f'{k}={v:.4f}' for k, v in model2.log_dict.items()))

    # ------------------------------------------------------------------ #
    # Test 3: VAE + PatchGAN (fixed weight, disc not yet active)          #
    # ------------------------------------------------------------------ #
    opt3 = copy.deepcopy(base_opt)
    opt3['train'] = copy.deepcopy(base_train_opt)
    opt3['train']['optim_d'] = {'type': 'Adam', 'lr': 4.5e-6}
    opt3['train']['disc_start'] = 10
    opt3['train']['disc_weight'] = 1.0
    opt3['train']['use_adaptive_weight'] = False
    opt3['network_d'] = {'type': 'NLayerDiscriminator3D', 'input_nc': 1, 'ndf': 8, 'n_layers': 2}
    model3 = VAEModel(opt3)
    model3.feed_data(batch)
    model3.optimize_parameters(current_iter=1)
    assert 'l_g_gan' in model3.log_dict
    print('Test 3 (VAE + GAN, disc_start not reached) OK — losses: ' + ', '.join(f'{k}={v:.4f}' for k, v in model3.log_dict.items()))

    # ------------------------------------------------------------------ #
    # Test 4: VAE + PatchGAN (adaptive weight, disc active)               #
    # ------------------------------------------------------------------ #
    opt4 = copy.deepcopy(base_opt)
    opt4['train'] = copy.deepcopy(base_train_opt)
    opt4['train']['optim_d'] = {'type': 'Adam', 'lr': 4.5e-6}
    opt4['train']['disc_start'] = 0
    opt4['train']['disc_weight'] = 1.0
    opt4['train']['use_adaptive_weight'] = True
    opt4['network_d'] = {'type': 'NLayerDiscriminator3D', 'input_nc': 1, 'ndf': 8, 'n_layers': 2}
    model4 = VAEModel(opt4)
    model4.feed_data(batch)
    model4.optimize_parameters(current_iter=1)
    assert 'l_g_gan' in model4.log_dict and 'l_d' in model4.log_dict
    print('Test 4 (VAE + GAN adaptive weight) OK — losses: ' + ', '.join(f'{k}={v:.4f}' for k, v in model4.log_dict.items()))

    # ------------------------------------------------------------------ #
    # Test 5: VAE + PatchGAN + logvar                                     #
    # ------------------------------------------------------------------ #
    opt5 = copy.deepcopy(base_opt)
    opt5['train'] = copy.deepcopy(base_train_opt)
    opt5['train']['optim_d'] = {'type': 'Adam', 'lr': 4.5e-6}
    opt5['train']['disc_start'] = 0
    opt5['train']['disc_weight'] = 1.0
    opt5['train']['use_adaptive_weight'] = True
    opt5['train']['use_logvar'] = True
    opt5['network_d'] = {'type': 'NLayerDiscriminator3D', 'input_nc': 1, 'ndf': 8, 'n_layers': 2}
    model5 = VAEModel(opt5)
    model5.feed_data(batch)
    model5.optimize_parameters(current_iter=1)
    assert 'l_g_gan' in model5.log_dict and 'l_d' in model5.log_dict
    print('Test 5 (VAE + GAN + logvar) OK — losses: ' + ', '.join(f'{k}={v:.4f}' for k, v in model5.log_dict.items()))

    print('\nAll VAEModel checks passed.')
