from basicsr.utils.registry import LOSS_REGISTRY
import torch
import torch.nn as nn
# import torch.nn.functional as F
from torch.nn import functional as F
import numpy as np
from typing import Union, Tuple, Optional
# from basicsr.losses import l1
from .loss_util import weighted_loss
from math import exp
from torch.autograd import Variable

@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')

@weighted_loss
def smooth_l1_loss(pred, target):
    return F.smooth_l1_loss(pred, target, reduction='none')

@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


@weighted_loss
def charbonnier_loss(pred, target, eps=1e-12):
    return torch.sqrt((pred - target)**2 + eps)

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

class SSIM_loss(nn.Module):
    def __init__(self, window_size=11, channel=3, is_cuda=True, size_average=True):
        super(SSIM_loss, self).__init__()
        self.window_size = window_size
        self.channel = channel
        self.size_average = size_average
        self.window = create_window(window_size, channel)
        # self.loss_weight = loss_weight
        if is_cuda:
            self.window = self.window.cuda()


    def forward(self, img1, img2):
        mu1 = F.conv2d(img1, self.window, padding=self.window_size // 2, groups=self.channel)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size // 2, groups=self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, self.window, padding=self.window_size // 2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, self.window, padding=self.window_size // 2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.window, padding=self.window_size // 2, groups=self.channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        # return (-torch.log10(ssim_map.mean())) 
        return (1 - ssim_map.mean())


class TensorGuidedFilter(nn.Module):
    """
    Tensor版本的Guided Filter，专为深度学习训练设计
    支持PyTorch tensor，可微分，支持批处理和GPU加速
    """
    
    def __init__(self, radius: int = 8, epsilon: float = 0.1):
        """
        初始化Tensor Guided Filter
        
        Args:
            radius: 滤波窗口半径
            epsilon: 正则化参数
        """
        super(TensorGuidedFilter, self).__init__()
        self.radius = radius
        self.epsilon = epsilon
        
    def box_filter(self, x: torch.Tensor, r: int) -> torch.Tensor:
        """
        张量版本的box filter（均值滤波）
        
        Args:
            x: 输入张量 (B, C, H, W) 或 (C, H, W)
            r: 滤波半径
            
        Returns:
            滤波后的张量
        """
        # 确保输入至少是4维 (B, C, H, W)
        if x.dim() == 3:
            x = x.unsqueeze(0)
            squeeze_batch = True
        else:
            squeeze_batch = False
            
        # 创建box filter kernel
        kernel_size = 2 * r + 1
        kernel = torch.ones(1, 1, kernel_size, kernel_size, 
                           device=x.device, dtype=x.dtype) / (kernel_size ** 2)
        
        # 对每个通道分别进行卷积
        B, C, H, W = x.shape
        x_reshaped = x.view(B * C, 1, H, W)
        
        # 使用padding='same'效果的padding
        pad = r
        x_padded = F.pad(x_reshaped, (pad, pad, pad, pad), mode='reflect')
        filtered = F.conv2d(x_padded, kernel)
        
        # 恢复形状
        filtered = filtered.view(B, C, H, W)
        
        if squeeze_batch:
            filtered = filtered.squeeze(0)
            
        return filtered
    
    def guided_filter_single_channel(self, p: torch.Tensor, I: torch.Tensor, 
                                   eps: float) -> torch.Tensor:
        """
        单通道引导滤波
        
        Args:
            p: 输入图像 (B, 1, H, W) 或 (1, H, W)
            I: 引导图像 (B, 1, H, W) 或 (1, H, W)
            eps: 正则化参数
            
        Returns:
            滤波后的图像
        """
        # 确保是4维张量
        if p.dim() == 3:
            p = p.unsqueeze(0)
            I = I.unsqueeze(0)
            squeeze_batch = True
        else:
            squeeze_batch = False
            
        # 计算均值
        mean_I = self.box_filter(I, self.radius)
        mean_p = self.box_filter(p, self.radius)
        mean_Ip = self.box_filter(I * p, self.radius)
        
        # 计算协方差和方差
        cov_Ip = mean_Ip - mean_I * mean_p
        mean_II = self.box_filter(I * I, self.radius)
        var_I = mean_II - mean_I * mean_I
        
        # 计算系数a和b
        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I
        
        # 计算输出
        mean_a = self.box_filter(a, self.radius)
        mean_b = self.box_filter(b, self.radius)
        
        q = mean_a * I + mean_b
        
        if squeeze_batch:
            q = q.squeeze(0)
            
        return q
    
    def forward(self, input_img: torch.Tensor, guide_img: torch.Tensor,
                radius: Optional[int] = None, epsilon: Optional[float] = None) -> torch.Tensor:
        """
        RGB图像的通道分离引导滤波
        
        Args:
            input_img: 待滤波的RGB图像 (B, 3, H, W) 或 (3, H, W)
            guide_img: 引导RGB图像 (B, 3, H, W) 或 (3, H, W)
            radius: 滤波半径（可选）
            epsilon: 正则化参数（可选）
            
        Returns:
            滤波后的RGB图像，形状与输入相同
        """
        if radius is None:
            radius = self.radius
        if epsilon is None:
            epsilon = self.epsilon
            
        # 临时更新radius
        original_radius = self.radius
        self.radius = radius
        
        # 确保输入是4维张量 (B, C, H, W)
        if input_img.dim() == 3:
            input_img = input_img.unsqueeze(0)
            guide_img = guide_img.unsqueeze(0)
            squeeze_batch = True
        else:
            squeeze_batch = False
            
        # 检查输入形状
        assert input_img.shape == guide_img.shape, "输入图像和引导图像形状必须相同"
        assert input_img.shape[1] == 3, "输入图像必须是3通道RGB图像"
        
        B, C, H, W = input_img.shape
        filtered_channels = []
        
        # 对每个通道分别进行引导滤波
        for i in range(3):
            # 提取单个通道 (B, 1, H, W)
            input_channel = input_img[:, i:i+1, :, :]
            guide_channel = guide_img[:, i:i+1, :, :]
            
            # 执行引导滤波
            filtered_channel = self.guided_filter_single_channel(
                input_channel, guide_channel, epsilon
            )
            
            filtered_channels.append(filtered_channel)
        
        # 合并通道
        filtered_img = torch.cat(filtered_channels, dim=1)
        
        # 恢复原始batch维度
        if squeeze_batch:
            filtered_img = filtered_img.squeeze(0)
            
        # 恢复原始radius
        self.radius = original_radius
        
        return filtered_img


def guided_filter_channel_separated(input_img: torch.Tensor, 
                                   guide_img: torch.Tensor,
                                   radius: int = 8, 
                                   epsilon: float = 0.1) -> torch.Tensor:
    """
    函数式接口：RGB图像通道分离引导滤波
    
    Args:
        input_img: 待滤波的RGB图像 
                  - 形状: (B, 3, H, W) 或 (3, H, W)
                  - 值域: [0, 1] 或 [0, 255]
        guide_img: 引导RGB图像
                  - 形状: (B, 3, H, W) 或 (3, H, W) 
                  - 值域: [0, 1] 或 [0, 255]
        radius: 滤波窗口半径
        epsilon: 正则化参数
        
    Returns:
        filtered_img: 滤波后的RGB图像，形状与输入相同
        
    Example:
        >>> input_tensor = torch.randn(4, 3, 256, 256)  # 批量大小4
        >>> guide_tensor = torch.randn(4, 3, 256, 256)
        >>> filtered = guided_filter_channel_separated(input_tensor, guide_tensor)
        >>> print(filtered.shape)  # torch.Size([4, 3, 256, 256])
    """
    filter_module = TensorGuidedFilter(radius=radius, epsilon=epsilon)
    filter_module.eval()  # 设置为评估模式
    
    with torch.no_grad() if not (input_img.requires_grad or guide_img.requires_grad) else torch.enable_grad():
        return filter_module(input_img, guide_img)


def guided_filter_channel_separated_trainable(input_img: torch.Tensor, 
                                             guide_img: torch.Tensor,
                                             radius: int = 8, 
                                             epsilon: float = 0.1) -> torch.Tensor:
    """
    可训练版本：支持梯度传播的RGB图像通道分离引导滤波
    
    Args:
        input_img: 待滤波的RGB图像 (B, 3, H, W) 或 (3, H, W)
        guide_img: 引导RGB图像 (B, 3, H, W) 或 (3, H, W)
        radius: 滤波窗口半径
        epsilon: 正则化参数
        
    Returns:
        filtered_img: 滤波后的RGB图像，形状与输入相同，支持梯度传播
    """
    filter_module = TensorGuidedFilter(radius=radius, epsilon=epsilon)
    return filter_module(input_img, guide_img)

@LOSS_REGISTRY.register()
class GuidedFilterLoss(nn.Module):
    """
    使用Guided Filter的损失函数
    可以在训练过程中作为正则化项使用
    """
    
    def __init__(self, radius: int = 60, epsilon: float = 0.01, loss_weight: float = 1.0, reduction='mean'):
        super(GuidedFilterLoss, self).__init__()
        self.guided_filter = TensorGuidedFilter(radius=radius, epsilon=epsilon)
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.ssim = SSIM_loss()
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor, 
                guide: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算引导滤波损失
        
        Args:
            pred: 预测图像 (B, 3, H, W)
            target: 目标图像 (B, 3, H, W)
            guide: 引导图像 (B, 3, H, W)，如果为None则使用target作为引导
            
        Returns:
            loss: 损失值
        """
        if guide is None:
            guide = target
            
        # 对预测和目标都应用引导滤波
        filtered_pred = self.guided_filter(pred, guide)
        filtered_target = self.guided_filter(target, guide)
        
        # 计算L2损失
        loss = l1_loss(filtered_pred, filtered_target, reduction=self.reduction)
        # loss = self.ssim(filtered_pred, filtered_target)
        
        return self.loss_weight * loss


# 实用工具函数
def normalize_tensor(tensor: torch.Tensor, from_range: str = 'auto', 
                    to_range: Tuple[float, float] = (0.0, 1.0)) -> torch.Tensor:
    """
    标准化张量到指定范围
    
    Args:
        tensor: 输入张量
        from_range: 输入范围 ('auto', '0-1', '0-255', '-1-1')
        to_range: 输出范围
        
    Returns:
        标准化后的张量
    """
    if from_range == 'auto':
        min_val = tensor.min()
        max_val = tensor.max()
    elif from_range == '0-1':
        min_val, max_val = 0.0, 1.0
    elif from_range == '0-255':
        min_val, max_val = 0.0, 255.0
    elif from_range == '-1-1':
        min_val, max_val = -1.0, 1.0
    else:
        raise ValueError(f"Unsupported from_range: {from_range}")
    
    # 标准化到0-1
    normalized = (tensor - min_val) / (max_val - min_val)
    
    # 缩放到目标范围
    to_min, to_max = to_range
    result = normalized * (to_max - to_min) + to_min
    
    return result


def tensor_to_numpy_image(tensor: torch.Tensor) -> np.ndarray:
    """
    将tensor转换为numpy图像用于可视化
    
    Args:
        tensor: 输入张量 (3, H, W) 或 (B, 3, H, W)
        
    Returns:
        numpy图像 (H, W, 3) 或 (B, H, W, 3)
    """
    if tensor.dim() == 4:
        # (B, 3, H, W) -> (B, H, W, 3)
        tensor = tensor.permute(0, 2, 3, 1)
    elif tensor.dim() == 3:
        # (3, H, W) -> (H, W, 3)
        tensor = tensor.permute(1, 2, 0)
    
    # 转换到CPU并转为numpy
    numpy_img = tensor.detach().cpu().numpy()
    
    # 确保值在0-1范围内
    numpy_img = np.clip(numpy_img, 0, 1)
    
    return numpy_img


# 示例使用代码
def example_usage():
    """演示如何在深度学习中使用tensor版本的Guided Filter"""
    
    # 创建示例数据
    batch_size = 4
    height, width = 256, 256
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 模拟输入图像和引导图像
    input_imgs = torch.randn(batch_size, 3, height, width, device=device)
    guide_imgs = torch.randn(batch_size, 3, height, width, device=device)
    
    # 标准化到0-1范围
    input_imgs = torch.sigmoid(input_imgs)
    guide_imgs = torch.sigmoid(guide_imgs)
    
    print(f"输入图像形状: {input_imgs.shape}")
    print(f"引导图像形状: {guide_imgs.shape}")
    print(f"设备: {device}")
    
    # 方法1: 使用函数式接口（推荐用于推理）
    print("\n=== 函数式接口测试 ===")
    filtered_imgs = guided_filter_channel_separated(
        input_imgs, guide_imgs, radius=8, epsilon=0.1
    )
    print(f"滤波后图像形状: {filtered_imgs.shape}")
    
    # 方法2: 使用模块化接口（推荐用于训练）
    print("\n=== 模块化接口测试 ===")
    guided_filter = TensorGuidedFilter(radius=8, epsilon=0.1).to(device)
    filtered_imgs_module = guided_filter(input_imgs, guide_imgs)
    print(f"滤波后图像形状: {filtered_imgs_module.shape}")
    
    # 方法3: 在损失函数中使用
    print("\n=== 损失函数测试 ===")
    target_imgs = torch.randn(batch_size, 3, height, width, device=device)
    target_imgs = torch.sigmoid(target_imgs)
    
    gf_loss = GuidedFilterLoss(radius=8, epsilon=0.1, weight=0.5).to(device)
    loss_value = gf_loss(filtered_imgs, target_imgs, guide_imgs)
    print(f"引导滤波损失: {loss_value.item():.6f}")
    
    # 测试梯度传播
    print("\n=== 梯度传播测试 ===")
    input_imgs.requires_grad_(True)
    filtered_trainable = guided_filter_channel_separated_trainable(
        input_imgs, guide_imgs, radius=8, epsilon=0.1
    )
    loss = filtered_trainable.mean()
    loss.backward()
    print(f"输入图像梯度范数: {input_imgs.grad.norm().item():.6f}")
    
    print("\n=== 测试完成 ===")


if __name__ == "__main__":
    example_usage()