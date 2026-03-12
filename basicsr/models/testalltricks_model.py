import torch
import torch.nn.functional as F
from collections import OrderedDict, defaultdict
from os import path as osp
from tqdm import tqdm
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 避免在服务器上的显示问题
import matplotlib.pyplot as plt

import seaborn as sns

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel


def _make_2d_hann_window(h: int, w: int, device, dtype):
    """2D Hann window for smooth overlap blending."""
    wy = torch.hann_window(h, periodic=False, device=device, dtype=dtype)
    wx = torch.hann_window(w, periodic=False, device=device, dtype=dtype)
    win = wy[:, None] * wx[None, :]
    # avoid zeros at borders causing division issues in extreme cases
    return win.clamp_min(1e-6)

@torch.no_grad()
def sliding_window_infer(net, x, patch=128, overlap=32, pad_mode="reflect", use_hann=False):
    """
    Sliding window inference with overlap-add blending.

    Args:
        net: model, expects [B,C,H,W] -> [B,C,H,W]
        x: input tensor [B,C,H,W]
        patch: window size
        overlap: overlap size (stride = patch - overlap)
        pad_mode: reflect/replicate/constant
        use_hann: whether to use Hann weighting (recommended)

    Returns:
        y: output tensor [B,C,H,W] (same H,W as input)
    """
    assert x.dim() == 4, "x must be [B,C,H,W]"
    b, c, h, w = x.shape
    stride = patch - overlap
    assert stride > 0, "patch must be > overlap"

    # If image is small enough, run directly (still keep padding behavior consistent)
    if h <= patch and w <= patch:
        # pad to patch for consistent behavior, then crop back
        pad_h = patch - h
        pad_w = patch - w
        x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode=pad_mode)
        y_pad = net(x_pad)
        return y_pad[:, :, :h, :w]

    # Compute necessary padding so that we can cover the full image with stride steps
    # We want last patch to start at <= H-patch, similarly for W.
    n_h = (h - patch + stride - 1) // stride + 1
    n_w = (w - patch + stride - 1) // stride + 1
    out_h = (n_h - 1) * stride + patch
    out_w = (n_w - 1) * stride + patch
    pad_bottom = max(0, out_h - h)
    pad_right = max(0, out_w - w)

    x_pad = F.pad(x, (0, pad_right, 0, pad_bottom), mode=pad_mode)
    _, _, hp, wp = x_pad.shape

    device = x.device
    dtype = x.dtype

    if use_hann:
        weight_2d = _make_2d_hann_window(patch, patch, device=device, dtype=dtype)
    else:
        weight_2d = torch.ones((patch, patch), device=device, dtype=dtype)

    weight_2d = weight_2d[None, None, :, :]  # [1,1,patch,patch]

    # Accumulators
    y_acc = torch.zeros((b, c, hp, wp), device=device, dtype=dtype)
    w_acc = torch.zeros((b, 1, hp, wp), device=device, dtype=dtype)

    for iy in range(n_h):
        y0 = iy * stride
        y1 = y0 + patch
        for ix in range(n_w):
            x0 = ix * stride
            x1 = x0 + patch

            patch_in = x_pad[:, :, y0:y1, x0:x1]
            patch_out = net(patch_in)

            # Blend
            y_acc[:, :, y0:y1, x0:x1] += patch_out * weight_2d
            w_acc[:, :, y0:y1, x0:x1] += weight_2d

    y_pad = y_acc / w_acc.clamp_min(1e-6)
    return y_pad[:, :, :h, :w]


@MODEL_REGISTRY.register()
class TestAllTricksModel(BaseModel):

    def __init__(self, opt):
        super(TestAllTricksModel, self).__init__(opt)

        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)

        if not hasattr(self.net_g, 'use_prompt') or not self.net_g.use_prompt:
            raise ValueError("Network must support prompt pools (use_prompt=True)")

        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        if self.is_train:
            self.init_training_settings()

        self.use_prompt = True

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)
            self.net_g_ema.eval()

        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None
        if train_opt.get('ssim_opt'):
            self.cri_ssim = build_loss(train_opt['ssim_opt']).to(self.device)
        else:
            self.cri_ssim = None

        if train_opt.get('fft_opt'):
            self.cri_fft = build_loss(train_opt['fft_opt']).to(self.device)
        else:
            self.cri_fft = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        self.init_multimodal_losses(train_opt)

        self.prompt_usage_stats = {
            'degradation': defaultdict(int)
        }
        self.stats_update_freq = train_opt.get('stats_update_freq', 5000)

        self.setup_optimizers()
        self.setup_schedulers()

        self.enable_prompt_analysis = self.opt['train'].get('enable_prompt_analysis', False)
        self.prompt_analysis_freq = self.opt['train'].get('prompt_analysis_freq', 10000) 
        self.prompt_analysis_save_path = self.opt['train'].get('prompt_analysis_save_path', './prompt_analysis')
        
        if self.enable_prompt_analysis:
            os.makedirs(self.prompt_analysis_save_path, exist_ok=True)
            logger = get_root_logger()
            logger.info(f"📊 Prompt analysis enabled, save to: {self.prompt_analysis_save_path}")

    def init_multimodal_losses(self, train_opt):

        self.prompt_diversity_weight = train_opt.get('prompt_diversity_weight', 0.01)


    def compute_prompt_diversity_loss(self):

        total_diversity_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)

        if hasattr(self.net_g, 'degradation_prompt_pool'):

            degradation_prompts = self.net_g.degradation_prompt_pool.prompt_values  # [pool_size, prompt_dim] 或 [pool_size, m, prompt_dim]
            if degradation_prompts.dim() == 3:

                degradation_prompts = degradation_prompts.view(degradation_prompts.size(0), -1)
            degradation_prompts_norm = F.normalize(degradation_prompts, dim=-1)
            
            similarity_matrix = torch.matmul(degradation_prompts_norm, degradation_prompts_norm.t())
            mask = torch.eye(similarity_matrix.size(0), device=self.device)
            similarity_matrix = similarity_matrix * (1 - mask)
            
            diversity_loss_values = torch.mean(torch.relu(similarity_matrix - self.diversity_temperature))

            degradation_keys = self.net_g.degradation_prompt_pool.prompt_keys  # [pool_size, prompt_dim]
            degradation_keys_norm = F.normalize(degradation_keys, dim=-1)
            
            similarity_matrix_keys = torch.matmul(degradation_keys_norm, degradation_keys_norm.t())
            similarity_matrix_keys = similarity_matrix_keys * (1 - mask)
            
            diversity_loss_keys = torch.mean(torch.relu(similarity_matrix_keys - self.diversity_temperature))

            degradation_diversity_loss = (diversity_loss_values + diversity_loss_keys) / 2.0
            total_diversity_loss = total_diversity_loss + degradation_diversity_loss.float()
        
        return total_diversity_loss

    def setup_optimizers(self):
        train_opt = self.opt['train']

        main_params = []
        prompt_params = []
        
        for name, param in self.net_g.named_parameters():
            if param.requires_grad:
                if 'prompt' in name.lower():
                    prompt_params.append(param)
                else:
                    main_params.append(param)
        
        optim_type = train_opt['optim_g'].pop('type')
        main_lr = train_opt['optim_g'].get('lr', 2e-4)
        
        prompt_lr = train_opt.get('prompt_lr', main_lr * 0.1) 
        
        param_groups = [
            {'params': main_params, 'lr': main_lr},
            {'params': prompt_params, 'lr': prompt_lr}
        ]
        
        self.optimizer_g = self.get_optimizer(optim_type, param_groups, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)
        
        logger = get_root_logger()
        logger.info(f'Main network params: {len(main_params)}, Prompt params: {len(prompt_params)}')
        logger.info(f'Main LR: {main_lr}, Prompt LR: {prompt_lr}')

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        self.gt_paths = data.get('gt_path', None)

    def optimize_parameters(self, current_iter):

        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        l_total = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        loss_dict = OrderedDict()

        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total = l_total + l_pix.float()
            loss_dict['l_pix'] = l_pix

        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total = l_total + l_percep.float()
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total = l_total + l_style.float()
                loss_dict['l_style'] = l_style

        if self.cri_ssim:
            l_ir_ssim= self.cri_ssim(self.output, self.gt)
            l_total += l_ir_ssim
            loss_dict['l_ssim'] = l_ir_ssim

        if self.cri_fft:
            l_ir_fft= self.cri_fft(self.output, self.gt)
            l_total += l_ir_fft
            loss_dict['l_fft'] = l_ir_fft

        for key, value in loss_dict.items():
            if not isinstance(value, torch.Tensor):
                logger = get_root_logger()
                logger.warning(f"Loss {key} is not a tensor: {type(value)}, converting...")
                loss_dict[key] = torch.tensor(float(value), device=self.device, dtype=torch.float32)

        if l_total.requires_grad:
            l_total.backward()
            self.optimizer_g.step()
        else:
            logger = get_root_logger()
            logger.warning("Total loss does not require grad, skipping backward pass")

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
    
    def test(self):
        """
        Upgraded test():
        - If EMA exists: keep as-is (full image forward), OR you can also apply sliding/TTA to EMA if you want.
        - Else:
            For each of 8 geometric TTAs:
                sliding-window inference (128, overlap 32) with Hann blending
                inverse transform back
            Aggregate (mean by default, can switch to median/trimmed-mean if you want)
        """
        patch = 256
        overlap = 16
        pad_mode = "reflect"
        use_hann = False

        # NOTE: assumes self.lq is [B,C,H,W] on correct device
        if hasattr(self, "net_g_ema"):
            self.net_g_ema.eval()
            with torch.no_grad():
                # Option A (simple): direct inference
                self.output = self.net_g_ema(self.lq)
                # Option B (stronger but slower): apply sliding + (optional) TTA to EMA as well
                # self.output = sliding_window_infer(self.net_g_ema, self.lq, patch, overlap, pad_mode, use_hann)
            return

        self.net_g.eval()
        with torch.no_grad():
            preds = []

            def infer_one(x_in):
                return sliding_window_infer(
                    self.net_g, x_in, patch=patch, overlap=overlap,
                    pad_mode=pad_mode, use_hann=use_hann
                )

            # 1) identity
            # self.output = infer_one(self.lq)
            pred = infer_one(self.lq)
            preds.append(pred)

            # 2) horizontal flip
            x_aug = torch.flip(self.lq, dims=[3])
            pred = infer_one(x_aug)
            preds.append(torch.flip(pred, dims=[3]))

            # 3) vertical flip
            x_aug = torch.flip(self.lq, dims=[2])
            pred = infer_one(x_aug)
            preds.append(torch.flip(pred, dims=[2]))

            # 4) rot90
            x_aug = torch.rot90(self.lq, k=1, dims=[2, 3])
            pred = infer_one(x_aug)
            preds.append(torch.rot90(pred, k=-1, dims=[2, 3]))

            # 5) rot180
            x_aug = torch.rot90(self.lq, k=2, dims=[2, 3])
            pred = infer_one(x_aug)
            preds.append(torch.rot90(pred, k=-2, dims=[2, 3]))

            # 6) rot270
            x_aug = torch.rot90(self.lq, k=3, dims=[2, 3])
            pred = infer_one(x_aug)
            preds.append(torch.rot90(pred, k=-3, dims=[2, 3]))

            # 7) hflip + rot90
            x_aug = torch.rot90(torch.flip(self.lq, dims=[3]), k=1, dims=[2, 3])
            pred = infer_one(x_aug)
            pred = torch.flip(torch.rot90(pred, k=-1, dims=[2, 3]), dims=[3])
            preds.append(pred)

            # 8) vflip + rot90
            x_aug = torch.rot90(torch.flip(self.lq, dims=[2]), k=1, dims=[2, 3])
            pred = infer_one(x_aug)
            pred = torch.flip(torch.rot90(pred, k=-1, dims=[2, 3]), dims=[2])
            preds.append(pred)

            stack = torch.stack(preds, dim=0)  # [T,B,C,H,W]

            # ---- Aggregation (default mean) ----
            # self.output = torch.mean(stack, dim=0)

            # # If you want to try median TTA aggregation (sometimes more robust, sometimes slightly worse):
            self.output = torch.median(stack, dim=0).values

        self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):

        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            self._initialize_best_metric_results(dataset_name)
        
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt

            # 清理内存
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                           f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                               f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                               f'{img_name}_{self.opt["name"]}.png')
                imwrite(sr_img, save_img_path)

            if with_metrics:
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
                
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)


    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                        f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)
                

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, 
                            param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)