import matplotlib.pyplot as plt
import numpy as np
import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

from ASR21cm.archs.arch_utils import make_coord
from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.models.sr_model import SRModel
from basicsr.utils import get_root_logger
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

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
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
        self.l1_pix = build_loss(train_opt['l1_opt']).to(self.device)

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def feed_data(self, data):
        # {'lq': T21_lr, 'gt': T21, 'delta': delta, 'vbv': vbv, 'labels': labels, 'T21_lr_mean': T21_lr_mean, 'T21_lr_std': T21_lr_std, 'scale_factor': scale_factor}
        self.lq = data['lq'].to(self.device)
        self.gt = data['gt'].to(self.device)
        # self.delta = data['delta'].to(self.device)
        # self.vbv = data['vbv'].to(self.device)
        # self.labels = data['labels'].to(self.device)
        self.T21_lr_mean = data['T21_lr_mean'].to(self.device)
        self.T21_lr_std = data['T21_lr_std'].to(self.device)
        # self.scale_factor = data['scale_factor']

    def optimize_parameters(self, current_iter):
        b, c, h, w, d = self.gt.shape
        xyz_hr = make_coord([h, h, h], ranges=None, flatten=False)
        xyz_hr = xyz_hr.view(1, -1, 3)
        xyz_hr = xyz_hr.repeat(b, 1, 1)

        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq, xyz_hr)

        l_total = 0
        loss_dict = OrderedDict()
        # l1 loss
        l_l1 = self.l1_pix(self.output, self.gt)
        l_total += l_l1
        loss_dict['l_l1'] = l_l1

        l_total.backward()
        self.optimizer_g.step()

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
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()

            metric_data['sr'] = visuals['sr']
            metric_data['hr'] = visuals['hr']
            metric_data['mean'] = visuals['mean']
            metric_data['std'] = visuals['std']

            # tentative for out of GPU memory
            del self.gt
            del self.lq
            del self.output
            del self.T21_lr_mean
            del self.T21_lr_std
            torch.cuda.empty_cache()

            if save_img:
                b, c, h, w, d = visuals['sr'].shape
                visuals['sr'] = visuals['sr'] * visuals['std'] + visuals['mean']
                visuals['hr'] = visuals['hr'] * visuals['std'] + visuals['mean']
                rmse = ((visuals['sr'] - visuals['hr'])**2).mean(dim=[1, 2, 3, 4]).sqrt()
                rmse = rmse.cpu().numpy()
                slice_idx = d // 2
                fig, axes = plt.subplots(3, b, figsize=(b * 5, 3 * 5))
                if b == 1:
                    axes = np.expand_dims(axes, axis=0)
                    axes = axes.reshape(3, b)
                for i in range(b):
                    vmin = min(visuals['sr'][i, :, :, slice_idx].min(), visuals['hr'][i, :, :, slice_idx].min())
                    vmax = max(visuals['sr'][i, :, :, slice_idx].max(), visuals['hr'][i, :, :, slice_idx].max())
                    axes[0, i].imshow(visuals['hr'][i, 0, :, :, slice_idx].cpu().numpy(), vmin=vmin, vmax=vmax)
                    axes[0, i].set_title('HR')
                    axes[1, i].imshow(visuals['sr'][i, 0, :, :, slice_idx].cpu().numpy(), vmin=vmin, vmax=vmax)
                    axes[1, i].set_title('SR')

                    xmin = min(visuals['sr'][i].min(), visuals['hr'][i].min())
                    xmax = max(visuals['sr'][i].max(), visuals['hr'][i].max())
                    bins = np.linspace(xmin, xmax, 100)
                    axes[2, i].hist(visuals['hr'][i].cpu().numpy().flatten(), bins=bins, alpha=0.5, label='HR', density=True)
                    axes[2, i].hist(visuals['sr'][i].cpu().numpy().flatten(), bins=bins, alpha=0.5, label='SR', density=True)
                    axes[2, i].legend(title='RMSE: {:.2f}'.format(rmse[i]))
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
        b, c, h, w, d = self.gt.shape
        xyz_hr = make_coord([h, h, h], ranges=None, flatten=False)
        xyz_hr = xyz_hr.view(1, -1, 3)
        xyz_hr = xyz_hr.repeat(b, 1, 1)
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.lq, xyz_hr)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.lq, xyz_hr)
            self.net_g.train()

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['sr'] = self.output.detach().cpu()
        out_dict['hr'] = self.gt.detach().cpu()
        out_dict['mean'] = self.T21_lr_mean.detach().cpu()
        out_dict['std'] = self.T21_lr_std.detach().cpu()
        return out_dict
