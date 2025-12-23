import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as osp
import torch
from collections import OrderedDict
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.models.sr_model import SRModel
from basicsr.utils import get_root_logger
from basicsr.utils.registry import MODEL_REGISTRY
from ASR21cm.utils import calculate_power_spectrum


@MODEL_REGISTRY.register()
class ScoreDiffusionVPSDEvpredModel(SRModel):
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

    def __init__(self, opt):
        super(ScoreDiffusionVPSDEvpredModel, self).__init__(opt)
        self.beta_max = opt.get('beta_max', 20.)
        self.beta_min = opt.get('beta_min', 0.1)
        self.epsilon_t = opt.get('epsilon_t', 1e-5)
        # Gradient clipping
        self.grad_clip = opt['train'].get('grad_clip', None) if 'train' in opt else None

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        self.gt = data['gt'].to(self.device)
        self.labels = data['labels'].to(self.device)
        self.T21_lr_mean = data['T21_lr_mean'].to(self.device)
        self.T21_lr_std = data['T21_lr_std'].to(self.device)
        self.delta = data['delta'].to(self.device)
        self.vbv = data['vbv'].to(self.device)

    def optimize_parameters(self, current_iter):
        b, c, *d = self.gt.shape
        t = torch.rand((b // 2 + b % 2, ), device=self.gt.device) * (1. - self.epsilon_t) + self.epsilon_t
        t = torch.cat([t, 1 - t + self.epsilon_t], dim=0)[:b]
        eps = torch.randn_like(self.gt, device=self.gt.device)
        C_t = self.C_t(t)
        std = self.sigma_t(t)
        gt_noisy = self.gt * torch.exp(C_t) + std * eps
        v_target = torch.exp(C_t) * eps - std * self.gt

        input = torch.cat([gt_noisy, self.lq, self.delta, self.vbv], dim=1)

        self.optimizer_g.zero_grad()
        v_theta = self.net_g(x=input, noise_labels=t, class_labels=None, augment_labels=None)


        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(v_target, v_theta)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix

        l_total.backward()

        # Gradient clipping
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), self.grad_clip)

        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def _save_validation_plot(self, metric_data, dataset_name, current_iter, sample_idx):
        """Save visualization plot with SR/HR slices, histogram, and power spectrum comparison.

        Args:
            metric_data: Dictionary containing 'sr', 'hr', 'mean', 'std'
            dataset_name: Name of validation dataset
            current_iter: Current training iteration
            sample_idx: Sample index in validation set
        """

        sr = metric_data['sr'].squeeze().numpy()
        hr = metric_data['hr'].squeeze().numpy()
        mean = metric_data['mean'].squeeze().numpy()
        std = metric_data['std'].squeeze().numpy()

        # Denormalize
        sr_denorm = sr * std + mean
        hr_denorm = hr * std + mean

        # Take middle slice
        mid_slice = sr.shape[0] // 2
        sr_slice = sr_denorm[mid_slice, :, :]
        hr_slice = hr_denorm[mid_slice, :, :]

        # Get vmin/vmax from HR for consistent color scale
        vmin = hr_denorm.min()
        vmax = hr_denorm.max()

        # Create figure with 4 panels
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        # Plot SR slice
        im0 = axes[0].imshow(sr_slice, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[0].set_title(f'SR (Sample {sample_idx})')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')
        plt.colorbar(im0, ax=axes[0])

        # Plot HR slice
        im1 = axes[1].imshow(hr_slice, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[1].set_title(f'HR (Sample {sample_idx})')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('y')
        plt.colorbar(im1, ax=axes[1])

        # Plot histogram
        axes[2].hist(sr_denorm.flatten(), bins=50, alpha=0.5, label='SR', density=True)
        axes[2].hist(hr_denorm.flatten(), bins=50, alpha=0.5, label='HR', density=True)
        axes[2].set_title('Distribution Comparison')
        axes[2].set_xlabel('Value')
        axes[2].set_ylabel('Density')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        # Calculate and plot power spectra
        # Convert to torch tensors and add batch/channel dimensions
        sr_tensor = torch.from_numpy(sr_denorm).unsqueeze(0).unsqueeze(0).float()
        hr_tensor = torch.from_numpy(hr_denorm).unsqueeze(0).unsqueeze(0).float()

        # Calculate power spectra using torch method
        k_vals_sr, P_k_sr = calculate_power_spectrum(sr_tensor, Lpix=3, kbins=50, dsq=True, method="torch", device="cpu")
        k_vals_hr, P_k_hr = calculate_power_spectrum(hr_tensor, Lpix=3, kbins=50, dsq=True, method="torch", device="cpu")

        # Convert to numpy for plotting
        k_vals_sr = k_vals_sr.cpu().numpy()
        P_k_sr = P_k_sr.squeeze().cpu().numpy()
        k_vals_hr = k_vals_hr.cpu().numpy()
        P_k_hr = P_k_hr.squeeze().cpu().numpy()

        # Plot power spectra
        axes[3].loglog(k_vals_sr, P_k_sr, label='SR', alpha=0.7, linewidth=2)
        axes[3].loglog(k_vals_hr, P_k_hr, label='HR', alpha=0.7, linewidth=2)
        axes[3].set_title('Power Spectrum Comparison')
        axes[3].set_xlabel('k [Mpc$^{-1}$]')
        axes[3].set_ylabel('$\\Delta^2(k)$ [mK$^2$]')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3, which='both')

        plt.tight_layout()

        # Save figure
        save_dir = osp.join(self.opt['path']['visualization'], dataset_name)
        os.makedirs(save_dir, exist_ok=True)
        save_path = osp.join(save_dir, f'sample_{sample_idx}_iter_{current_iter}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def _save_forward_reverse_validation_plot(self, metric_data, dataset_name, current_iter, sample_idx):
        """Save forward and reverse diffusion process visualization.

        Creates a plot showing:
        - Row 1: Forward process (adding noise) at specified time steps
        - Row 2: Reverse process (denoising) at approximate same time steps (from x_sequence)
        - Row 3: Overlaid histograms (forward + reverse)
        - Row 4: Power spectrum comparison (forward vs reverse)

        Args:
            metric_data: Dictionary containing 'sr_sequence', 'hr'
            dataset_name: Name of validation dataset
            current_iter: Current training iteration
            sample_idx: Sample index in validation set
            t_values: List of time values to visualize
        """
        import os
        x_sequence = metric_data['sr_sequence']

        num_steps = x_sequence.shape[1]

        # Extract data
        T21_normalized = metric_data['hr'].squeeze(0).squeeze(0)  # Remove batch and channel dims (B, C, H, W, D) -> (H, W, D)

        time_steps = torch.linspace(1.-self.epsilon_t, self.epsilon_t, num_steps, device=T21_normalized.device)

        t_values=[self.epsilon_t, 0.1, 0.2, 0.6, 1.-self.epsilon_t]
        n_times = len(t_values)
        t_values = sorted(t_values)
        idx_list = []

        for i,target_t in enumerate(t_values):
            # Find the closest time step
            diffs = torch.abs(time_steps - target_t)
            closest_idx = torch.argmin(diffs).item()
            idx_list.append(closest_idx)

            # Extract the state at this index by slicing dim 1
            # x_sequence[:, idx:idx+1, :, :, :] gives us (B, 1, H, W, D)
            t_values[i] = target_t

        # Create figure with 4 rows
        fig, axes = plt.subplots(4, n_times, figsize=(4 * (n_times), 16))

        # Add title
        experiment_name = osp.basename(self.opt['path']['experiments_root'])
        fig.suptitle(f'Forward and Reverse Diffusion Process - {experiment_name} (Iter {current_iter})',
                     fontsize=16, fontweight='bold', y=0.995)

        # Get middle slice index
        mid_slice = T21_normalized.shape[0] // 2

        # ============ Forward process ============
        forward_states = []
        forward_slices = []
        vmin_vmax = []

        # Compute all forward states
        for i, t in enumerate(t_values):
            eps = torch.randn_like(T21_normalized)
            t_tensor = torch.tensor([t], dtype=torch.float32, device=T21_normalized.device)

            C = self.C_t(t_tensor)
            sigma = self.sigma_t(t_tensor)
            C = C.squeeze()
            sigma = sigma.squeeze()

            x_t = T21_normalized * torch.exp(C) + sigma * eps
            x_t_cpu = x_t.cpu()
            x_t_numpy = x_t_cpu.numpy()
            forward_states.append(x_t_cpu)
            forward_slices.append(x_t_numpy[mid_slice, :, :])
            vmin_vmax.append((x_t_numpy.min(), x_t_numpy.max()))

        # Plot forward slices
        for i, (t, forward_state, forward_slice) in enumerate(zip(t_values, forward_states, forward_slices)):
            vmin, vmax = vmin_vmax[i]
            im = axes[0, i].imshow(forward_slice, cmap='viridis', vmin=vmin, vmax=vmax)

            t_tensor = torch.tensor([t], dtype=torch.float32)
            sig_t = self.sigma_t(t_tensor).squeeze().item()
            exp_Ct = torch.exp(self.C_t(t_tensor)).squeeze().item()

            axes[0, i].set_title(f't = {t:.3f} (Forward)\nexp(C_t)={exp_Ct:.3f}, σ_t={sig_t:.3f}',
                                fontsize=11, fontweight='bold')
            axes[0, i].set_xlabel('x')
            axes[0, i].set_ylabel('y')
            plt.colorbar(im, ax=axes[0, i])

            axes[2, i].hist(forward_state.flatten(), bins=50, alpha=0.5, density=True, color='red', edgecolor='black', label='Forward')

        # ============ Reverse process (using pre-computed x_sequence) ============
        # Extract reverse states at target times
        # x_sequence has shape (B, num_steps, H, W, D) where states are concatenated along dim 1
        reverse_states = [x_sequence[:, closest_idx, :, :, :] for closest_idx in idx_list]
        reverse_slices = [x_seq.squeeze(0).cpu().numpy()[mid_slice, :, :] for x_seq in reverse_states]

        # Plot reverse slices and histograms
        for i, (t, forward_state, forward_slice, reverse_state, reverse_slice) in enumerate(zip(t_values, forward_states, forward_slices, reverse_states, reverse_slices)):
            vmin, vmax = vmin_vmax[i]
            im = axes[1, i].imshow(reverse_slice, cmap='viridis', vmin=vmin, vmax=vmax)
            axes[1, i].set_title(f't ≈ {t_values[i]:.3f} (Reverse)', fontsize=11, fontweight='bold')
            plt.colorbar(im, ax=axes[1, i])

            axes[2, i].hist(reverse_state.flatten(), bins=50, alpha=0.5, density=True, color='green', edgecolor='black', label='Reverse')

            fwd_mean = forward_state.mean().item()
            fwd_std = forward_state.std().item()
            rev_mean = reverse_state.mean().item()
            rev_std = reverse_state.std().item()
            axes[2, i].text(0.05, 0.95, f'Fwd: μ={fwd_mean:.3f}, σ={fwd_std:.3f}\nRev: μ={rev_mean:.3f}, σ={rev_std:.3f}', transform=axes[2, i].transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=8)
            axes[2, i].set_title('Distribution Overlay', fontsize=10)
            axes[2, i].set_xlabel('Value')
            axes[2, i].set_ylabel('Density')
            axes[2, i].grid(True, alpha=0.3)
            axes[2, i].set_xlim(-12, 5)
            axes[2, i].legend(loc='upper right', fontsize=8)

            # Calculate and plot power spectra
            # Add batch and channel dimensions for power spectrum calculation
            forward_tensor = forward_state.unsqueeze(0).unsqueeze(0).float()
            reverse_tensor = reverse_state.unsqueeze(0).float()

            k_vals_fwd, P_k_fwd = calculate_power_spectrum(forward_tensor, Lpix=3, kbins=50, dsq=True, method="torch", device="cpu")
            k_vals_rev, P_k_rev = calculate_power_spectrum(reverse_tensor, Lpix=3, kbins=50, dsq=True, method="torch", device="cpu")

            k_vals_fwd = k_vals_fwd.cpu().numpy()
            P_k_fwd = P_k_fwd.squeeze().cpu().numpy()
            k_vals_rev = k_vals_rev.cpu().numpy()
            P_k_rev = P_k_rev.squeeze().cpu().numpy()

            axes[3, i].loglog(k_vals_fwd, P_k_fwd, label='Forward', alpha=0.7, linewidth=2, color='red')
            axes[3, i].loglog(k_vals_rev, P_k_rev, label='Reverse', alpha=0.7, linewidth=2, color='green')
            axes[3, i].set_title('Power Spectrum', fontsize=10)
            axes[3, i].set_xlabel('k [Mpc$^{-1}$]')
            axes[3, i].set_ylabel('$\\Delta^2(k)$ [mK$^2$]')
            axes[3, i].legend(loc='upper right', fontsize=8)
            axes[3, i].grid(True, alpha=0.3, which='both')

        plt.tight_layout()

        # Save figure
        save_dir = osp.join(self.opt['path']['visualization'], dataset_name)
        os.makedirs(save_dir, exist_ok=True)
        save_path = osp.join(save_dir, f'forward_reverse_sample_{sample_idx}_iter_{current_iter}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

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
            x_sequence = self.test(x_lr=self.lq, delta=self.delta, vbv=self.vbv, class_labels=None, num_steps=50, verbose=True)

            metric_data['sr_sequence'] = x_sequence.detach().cpu()
            metric_data['sr'] = self.output.detach().cpu()
            metric_data['hr'] = self.gt.detach().cpu()
            metric_data['mean'] = self.T21_lr_mean.detach().cpu()
            metric_data['std'] = self.T21_lr_std.detach().cpu()

            # tentative for out of GPU memory
            del self.gt
            del self.output
            del self.lq
            del self.labels
            del self.T21_lr_mean
            del self.T21_lr_std
            del self.delta
            del self.vbv

            torch.cuda.empty_cache()

            if save_img:
                #self._save_validation_plot(metric_data, dataset_name, current_iter, idx + 1)
                self._save_forward_reverse_validation_plot(metric_data, dataset_name, current_iter, idx + 1)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {idx + 1}/{len(dataloader)}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def test(self, x_lr=None, delta=None, vbv=None, class_labels=None, num_steps=1000, verbose=False):
        """
        Euler Maruyama sampler

        Args:
            x_lr: low resolution input for conditional generation
            conditionals: list of conditional inputs (delta and vbc)
            class_labels: class labels for (could be for astrophysics parameters - not implemented yet)
            num_steps: number of sampling steps
            verbose: whether to show tqdm progress bar
        Returns:
            x_sequence: sampled high resolution outputs at each step
        """

        b, *d = delta.shape  # select the last conditional to get the shape (order is delta,vbv)

        time_steps = torch.linspace(1.-self.epsilon_t, self.epsilon_t, num_steps, device=x_lr.device)
        dt = torch.abs(time_steps[1] - time_steps[0])  # Positive step size

        x = torch.randn_like(delta, device=delta.device)

        x_sequence = torch.empty_like(x, device='cpu')

        for time_step in tqdm(time_steps, desc='sampling', disable=not verbose):
            batch_time_step = torch.tensor(b * [time_step], device=x_lr.device)

            input = torch.cat([x, x_lr, delta, vbv], dim=1)
            if hasattr(self, 'net_g_ema'):
                self.net_g_ema.eval()
                with torch.no_grad():
                    v_theta = self.net_g_ema(x=input, noise_labels=batch_time_step, class_labels=class_labels, augment_labels=None)
            else:
                self.net_g.eval()
                with torch.no_grad():
                    v_theta = self.net_g(x=input, noise_labels=batch_time_step, class_labels=class_labels, augment_labels=None)
                self.net_g.train()

            std = self.sigma_t(batch_time_step)
            alpha = torch.exp(self.C_t(batch_time_step))
            eps_theta = alpha * v_theta + std * x
            score = -eps_theta / std

            # Reverse-time SDE: dx = [f(x,t) - g(t)²*score(x,t)] * (-dt) + g(t) dw
            # With dt > 0, this becomes: x + [f - g²*score] * (-dt)
            f_drift = self.f_drift(x, batch_time_step)
            g_diffusion = self.g_diffusion(batch_time_step)

            eps = torch.randn_like(delta, device=delta.device)

            # Euler-Maruyama: going backwards means subtracting the forward drift
            x = x - f_drift * dt + g_diffusion**2 * score * dt + g_diffusion * torch.sqrt(dt) * eps

            x_sequence = torch.cat([x_sequence, x.detach().cpu()], dim=1)

        self.output = x

        return x_sequence

    def C_t(self, t):
        C_t = -0.25 * t**2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        C_t = C_t[:, None, None, None, None]
        return C_t

    def beta_t(self, t):
        beta_t = self.beta_min + t * (self.beta_max - self.beta_min)
        beta_t = beta_t[:, None, None, None, None]
        return beta_t

    def sigma_t(self, t):
        sigma_t = (1. - torch.exp(2. * self.C_t(t))).sqrt()
        sigma_t = sigma_t
        return sigma_t

    def f_drift(self, x, t):
        beta_t = self.beta_t(t)
        f_drift = -0.5 * beta_t * x
        return f_drift

    def g_diffusion(self, t):
        beta_t = self.beta_t(t)
        g_diffusion = torch.sqrt(beta_t)
        return g_diffusion

    def get_optimizer(self, optim_type, params, lr, **kwargs):
        if optim_type == 'Adam':
            optimizer = torch.optim.Adam(params, lr, **kwargs)
        elif optim_type == 'AdamW':
            optimizer = torch.optim.AdamW(params, lr, **kwargs)
        else:
            raise NotImplementedError(f'optimizer {optim_type} is not supperted yet.')
        return optimizer
