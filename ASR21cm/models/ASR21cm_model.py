import matplotlib.pyplot as plt
import numpy as np
import os
import time
import torch
from collections import OrderedDict
from os import path as osp
from SR21cm.utils import calculate_power_spectrum
from tqdm import tqdm

from ASR21cm.archs.arch_utils import make_coord
from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.models.sr_model import SRModel
from basicsr.utils import get_root_logger
from basicsr.utils.dist_util import master_only
from basicsr.utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class ASR21cmModel(SRModel):
    """Example model based on the SRModel class.

    In this example model, we want to implement a new model that trains with both L1 and L2 loss.

    New defined functions:
        init_training_settings(self)
        feed_data(self, data)
        optimize_parameters(self, current_iter)

    Inherited functions:
        __init__(self, opt)
        setup_optimizers(self)
        test(self)
        dist_validation(self, dataloader, current_iter, tb_logger, save_img)
        nondist_validation(self, dataloader, current_iter, tb_logger, save_img)
        _log_validation_metric_values(self, current_iter, dataset_name, tb_logger)
        get_current_visuals(self)
        save(self, epoch, current_iter)
    """

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        logger = get_root_logger()
        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        self.l1_loss = build_loss(train_opt['l1_opt']).to(self.device)
        if train_opt.get('dsq_opt'):
            self.dsq_loss = build_loss(train_opt['dsq_opt']).to(self.device)
        else:
            self.dsq_loss = None

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

        # use automatic mixed precision (AMP) for training
        self.use_amp = self.opt.get('use_amp', False)
        if self.use_amp:
            self.scaler = torch.amp.GradScaler(enabled=True)
        else:
            self.scaler = None

        # test that model can run 512x512x512 data with scale_min
        if self.opt.get('test_forward_size', False):
            scale_min = self.opt['datasets']['train'].get('scale_min', 1.1)
            h_hr = self.opt.get('test_forward_size', False)
            h_lr = int(h_hr // scale_min)
            self.gt = torch.randn(1, 1, h_hr, h_hr, h_hr).to(self.device)
            self.lq = torch.randn(1, 1, h_lr, h_lr, h_lr).to(self.device)
            logger.info(f'Running {h_hr}x{h_hr}x{h_hr} test with scale_min: {scale_min}, h_lr: {h_lr}, h_hr: {h_hr}')
            self.test()
            del self.gt
            del self.lq
            del self.output
            torch.cuda.empty_cache()
            logger.info(f'Model passed {h_hr}x{h_hr}x{h_hr} test with scale_min: {scale_min}')

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device) if isinstance(data['lq'], torch.Tensor) else [lq.to(self.device) for lq in data['lq']]
        self.gt = data['gt'].to(self.device) if isinstance(data['gt'], torch.Tensor) else [gt.to(self.device) for gt in data['gt']]
        self.T21_lr_mean = data['T21_lr_mean'].to(self.device) if isinstance(data['T21_lr_mean'], torch.Tensor) else [mean.to(self.device) for mean in data['T21_lr_mean']]
        self.T21_lr_std = data['T21_lr_std'].to(self.device) if isinstance(data['T21_lr_std'], torch.Tensor) else [std.to(self.device) for std in data['T21_lr_std']]
        self.scale_factor = data['scale_factor'].to(self.device) if isinstance(data['scale_factor'], torch.Tensor) else [scale.to(self.device) for scale in data['scale_factor']]

    def optimize_parameters(self, current_iter):
        # with torch.autograd.profiler.emit_nvtx():
        b, c, h, w, d = self.gt.shape
        xyz_hr = make_coord([h, h, h], ranges=None, flatten=False)
        xyz_hr = xyz_hr.view(1, -1, 3)
        xyz_hr = xyz_hr.repeat(b, 1, 1)

        self.optimizer_g.zero_grad()
        with torch.amp.autocast(device_type=self.device.type, enabled=self.scaler is not None):
            self.output = self.net_g(self.lq, xyz_hr)

            l_total = 0
            loss_dict = OrderedDict()

            # l1 loss
            l_l1 = self.l1_loss(self.output, self.gt)
            l_total += l_l1
            loss_dict['l_l1'] = l_l1

            # dsq loss
            if self.dsq_loss:
                l_dsq = self.dsq_loss(self.output, self.gt)
                l_total += l_dsq
                loss_dict['l_dsq'] = l_dsq

        if self.scaler is not None:
            self.scaler.scale(l_total).backward()
            self.scaler.step(self.optimizer_g)
            self.scaler.update()
        else:
            l_total.backward()
            self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)
        self.log_dict.update({'l_total': l_total.item()})

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
        # torch.cuda.memory._dump_snapshot(filename=osp.join(self.opt['path']['experiments_root'], f'{current_iter}_memory_snapshot.pickle')) if torch.cuda.is_available() and self.opt.get('memory_profiling', False) else None

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
            metric_data['scale_factor'] = visuals['scale_factor']

            # tentative for out of GPU memory
            del self.gt
            del self.lq
            del self.output
            del self.T21_lr_mean
            del self.T21_lr_std
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
                        axes[2, i].set_xlabel(r'$T_{{21}}$ [${\rm mK}$]')

                        axes[3, i].loglog(k_hr, dsq_hr[i, 0], label='$T_{{21}}$ HR', ls='solid', lw=2)
                        axes[3, i].loglog(k_sr, dsq_sr[i, 0], label='$T_{{21}}$ SR', ls='solid', lw=2)
                        axes[3, i].set_xlabel('$k\\ [\\mathrm{{cMpc^{-1}}}]$')
                        axes[3, i].legend(title=f'RMSE: {rmse_i:.2f}, \nscale: {scale_i:.2f}', loc='upper left')

                    axes[0, 0].set_ylabel('HR')
                    axes[1, 0].set_ylabel('SR')
                    axes[2, 0].set_ylabel('PDF')
                    axes[3, 0].set_ylabel('$\\Delta^2_{{21}}\\ \\mathrm{{[mK^2]}}$ ')

                    save_img_path = osp.join(self.opt['path']['visualization'], f'{current_iter}.png')
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
        print(f'Finished testing {dataset_name} dataset')

    def test(self):
        b, c, h, w, d = self.gt.shape if isinstance(self.gt, torch.Tensor) else self.gt[0].shape
        xyz_hr = make_coord([h, h, h], ranges=None, flatten=False)
        xyz_hr = xyz_hr.view(1, -1, 3)
        xyz_hr = xyz_hr.repeat(b, 1, 1)
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                if isinstance(self.lq, torch.Tensor):
                    with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                        self.output = self.net_g_ema(self.lq, xyz_hr)
                elif isinstance(self.lq, list):
                    with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                        self.output = []
                        for lq in tqdm(self.lq, total=len(self.lq), disable=not self.opt['val'].get('pbar', False), desc='testing scales'):
                            self.output.append(self.net_g_ema(lq, xyz_hr))
        else:
            self.net_g.eval()
            with torch.no_grad():
                if isinstance(self.lq, torch.Tensor):
                    with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                        self.output = self.net_g(self.lq, xyz_hr)
                elif isinstance(self.lq, list):
                    with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):

                        self.output = []
                        for lq in tqdm(self.lq, total=len(self.lq), disable=not self.opt['val'].get('pbar', False), desc='testing scales'):
                            self.output.append(self.net_g(lq, xyz_hr))
            self.net_g.train()

    def get_current_visuals(self):
        out_dict = OrderedDict()
        # out_dict['lq'] = self.lq.detach().cpu() if isinstance(self.lq, torch.Tensor) else [lq.detach().cpu() for lq in self.lq]
        out_dict['sr'] = self.output.detach().cpu() if isinstance(self.output, torch.Tensor) else torch.cat(self.output, dim=0).detach().cpu()
        out_dict['hr'] = self.gt.detach().cpu() if isinstance(self.gt, torch.Tensor) else torch.cat(self.gt, dim=0).detach().cpu()
        out_dict['mean'] = self.T21_lr_mean.detach().cpu() if isinstance(self.T21_lr_mean, torch.Tensor) else torch.cat(self.T21_lr_mean, dim=0).detach().cpu()
        out_dict['std'] = self.T21_lr_std.detach().cpu() if isinstance(self.T21_lr_std, torch.Tensor) else torch.cat(self.T21_lr_std, dim=0).detach().cpu()
        out_dict['scale_factor'] = self.scale_factor if isinstance(self.scale_factor, torch.Tensor) else torch.cat(self.scale_factor, dim=0).detach().cpu()
        return out_dict

    @master_only
    def save_training_state(self, epoch, current_iter):
        """Save training states during training, which will be used for
        resuming.

        Args:
            epoch (int): Current epoch.
            current_iter (int): Current iteration.
        """
        if current_iter != -1:
            state = {'epoch': epoch, 'iter': current_iter, 'optimizers': [], 'schedulers': []}
            for o in self.optimizers:
                state['optimizers'].append(o.state_dict())
            for s in self.schedulers:
                state['schedulers'].append(s.state_dict())
            if self.scaler is not None:
                state['scaler'] = self.scaler.state_dict()
            save_filename = f'{current_iter}.state'
            save_path = os.path.join(self.opt['path']['training_states'], save_filename)

            # avoid occasional writing errors
            retry = 3
            while retry > 0:
                try:
                    torch.save(state, save_path)
                except Exception as e:
                    logger = get_root_logger()
                    logger.warning(f'Save training state error: {e}, remaining retry times: {retry - 1}')
                    time.sleep(1)
                else:
                    break
                finally:
                    retry -= 1
            if retry == 0:
                logger.warning(f'Still cannot save {save_path}. Just ignore it.')
                # raise IOError(f'Cannot save {save_path}.')

    def resume_training(self, resume_state):
        """Reload the optimizers and schedulers for resumed training.

        Args:
            resume_state (dict): Resume state.
        """
        resume_optimizers = resume_state['optimizers']
        resume_schedulers = resume_state['schedulers']
        assert len(resume_optimizers) == len(self.optimizers), 'Wrong lengths of optimizers'
        assert len(resume_schedulers) == len(self.schedulers), 'Wrong lengths of schedulers'
        for i, o in enumerate(resume_optimizers):
            self.optimizers[i].load_state_dict(o)
        for i, s in enumerate(resume_schedulers):
            self.schedulers[i].load_state_dict(s)
        if self.scaler is not None:
            self.scaler.load_state_dict(resume_state['scaler'])
