import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from basicsr.utils.registry import LOSS_REGISTRY
# from basicsr.archs.DINOv2Discriminator_arch import DINOv2DiscriminatorTorchHub

# def lecam_regularization(discriminator, real_samples, fake_samples, gamma=1.0):
#     """
#     计算LeCam正则化项
    
#     Args:
#         discriminator: 判别器网络
#         real_samples: 真实样本 [batch_size, ...]
#         fake_samples: 生成样本 [batch_size, ...]
#         gamma: 目标梯度范数 (默认1.0)
    
#     Returns:
#         lecam_loss: LeCam正则化损失
#     """
#     # 合并真实和虚假样本
#     samples = torch.cat([real_samples, fake_samples], dim=0)
#     samples.requires_grad_(True)
    
#     # 获取判别器输出
#     d_out = discriminator(samples)
    
#     # 计算梯度
#     gradients = autograd.grad(
#         outputs=d_out.sum(),
#         inputs=samples,
#         create_graph=True,
#         retain_graph=True,
#         only_inputs=True
#     )[0]
    
#     # 计算梯度范数
#     gradients = gradients.view(gradients.size(0), -1)
#     gradient_norm = torch.norm(gradients, p=2, dim=1)
    
#     # LeCam正则化项: (||∇_x D(x)||_2 - γ)²
#     lecam_loss = torch.mean((gradient_norm - gamma) ** 2)
    
#     return lecam_loss

def lecam_regularization(real_samples, fake_samples, real_out, fake_out, gamma=1.0):
    """
    计算LeCam正则化项
    
    Args:
        discriminator: 判别器网络
        real_samples: 真实样本 [batch_size, ...]
        fake_samples: 生成样本 [batch_size, ...]
        gamma: 目标梯度范数 (默认1.0)
    
    Returns:
        lecam_loss: LeCam正则化损失
    """
    # # 合并真实和虚假样本
    samples = torch.cat([real_samples, fake_samples], dim=0)
    samples.requires_grad_(True)
    
    # 获取判别器输出
    d_out = torch.cat([real_out, fake_out], dim=0)
    d_out.requires_grad_(True)
    
    # 计算梯度
    gradients = autograd.grad(
        outputs=d_out.sum(),
        inputs=samples,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # 计算梯度范数
    gradients = gradients.view(gradients.size(0), -1)
    gradient_norm = torch.norm(gradients, p=2, dim=1)
    
    # LeCam正则化项: (||∇_x D(x)||_2 - γ)²
    lecam_loss = torch.mean((gradient_norm - gamma) ** 2)
    
    return lecam_loss


def lecam_regularization_efficient(discriminator, samples, gamma=1.0):
    """
    更高效的LeCam正则化实现（避免重复前向传播）
    
    Args:
        discriminator: 判别器网络
        samples: 输入样本 [batch_size, ...]
        gamma: 目标梯度范数
    
    Returns:
        lecam_loss: LeCam正则化损失
    """
    samples.requires_grad_(True)
    
    # 判别器前向传播
    d_out = discriminator(samples)
    
    # 计算梯度
    gradients = autograd.grad(
        outputs=d_out.sum(),
        inputs=samples,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # 重塑并计算范数
    gradients = gradients.view(gradients.size(0), -1)
    gradient_norm = torch.norm(gradients, p=2, dim=1)
    
    # LeCam正则化损失
    lecam_loss = torch.mean((gradient_norm - gamma) ** 2)
    
    return lecam_loss


@LOSS_REGISTRY.register()
class lecam_regularizationLoss(nn.Module):

    def __init__(self, loss_weight=0.001, gamma=1.0):
        super(lecam_regularizationLoss, self).__init__()

        self.loss_weight = loss_weight
        # self.lecam_regularization = lecam_regularization(gamma=gamma)

    def forward(self, real_samples, fake_samples, real_out, fake_out, **kwargs):

        return self.loss_weight * lecam_regularization(real_samples, fake_samples, real_out, fake_out, gamma=1.0)

class GANTrainer:
    """
    集成LeCam正则化的GAN训练器
    """
    def __init__(self, generator, discriminator, lr=0.0002, lecam_lambda=0.001):
        self.generator = generator
        self.discriminator = discriminator
        self.lecam_lambda = lecam_lambda
        
        # 优化器
        self.g_optimizer = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
        self.d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        
        # 损失函数
        self.criterion = nn.BCEWithLogitsLoss()
    
    def train_discriminator(self, real_samples, noise):
        """
        训练判别器（包含LeCam正则化）
        """
        batch_size = real_samples.size(0)
        
        # 生成假样本
        with torch.no_grad():
            fake_samples = self.generator(noise)
        
        # 真实样本标签和预测
        real_labels = torch.ones(batch_size, 1, device=real_samples.device)
        real_pred = self.discriminator(real_samples)
        real_loss = self.criterion(real_pred, real_labels)
        
        # 假样本标签和预测
        fake_labels = torch.zeros(batch_size, 1, device=real_samples.device)
        fake_pred = self.discriminator(fake_samples)
        fake_loss = self.criterion(fake_pred, fake_labels)
        
        # 标准GAN损失
        gan_loss = real_loss + fake_loss
        
        # LeCam正则化
        lecam_loss = lecam_regularization(
            self.discriminator, 
            real_samples, 
            fake_samples,
            gamma=1.0
        )
        
        # 总损失
        total_loss = gan_loss + self.lecam_lambda * lecam_loss
        
        # 反向传播
        self.d_optimizer.zero_grad()
        total_loss.backward()
        self.d_optimizer.step()
        
        return {
            'discriminator_loss': total_loss.item(),
            'gan_loss': gan_loss.item(),
            'lecam_loss': lecam_loss.item()
        }
    
    def train_generator(self, noise):
        """
        训练生成器
        """
        batch_size = noise.size(0)
        
        # 生成假样本
        fake_samples = self.generator(noise)
        
        # 尝试欺骗判别器
        real_labels = torch.ones(batch_size, 1, device=noise.device)
        fake_pred = self.discriminator(fake_samples)
        g_loss = self.criterion(fake_pred, real_labels)
        
        # 反向传播
        self.g_optimizer.zero_grad()
        g_loss.backward()
        self.g_optimizer.step()
        
        return {'generator_loss': g_loss.item()}


# 使用示例
def example_usage():
    """
    LeCam正则化使用示例
    """
    # 假设的网络架构（简化）
    class SimpleGenerator(nn.Module):
        def __init__(self, noise_dim=100, output_dim=784):
            super().__init__()
            self.main = nn.Sequential(
                nn.Linear(noise_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, output_dim),
                nn.Tanh()
            )
        
        def forward(self, x):
            return self.main(x)
    
    class SimpleDiscriminator(nn.Module):
        def __init__(self, input_dim=784):
            super().__init__()
            self.main = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.LeakyReLU(0.2),
                nn.Linear(512, 256),
                nn.LeakyReLU(0.2),
                nn.Linear(256, 1)
            )
        
        def forward(self, x):
            return self.main(x.view(x.size(0), -1))
    
    # 初始化网络
    generator = SimpleGenerator()
    discriminator = SimpleDiscriminator()
    
    # 创建训练器
    trainer = GANTrainer(generator, discriminator, lecam_lambda=0.001)
    
    # 模拟训练循环
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator.to(device)
    discriminator.to(device)
    
    for epoch in range(10):
        # 模拟数据
        batch_size = 64
        real_samples = torch.randn(batch_size, 784, device=device)
        noise = torch.randn(batch_size, 100, device=device)
        
        # 训练判别器
        d_metrics = trainer.train_discriminator(real_samples, noise)
        
        # 训练生成器
        g_metrics = trainer.train_generator(noise)
        
        print(f"Epoch {epoch+1}:")
        print(f"  D Loss: {d_metrics['discriminator_loss']:.4f}")
        print(f"  G Loss: {g_metrics['generator_loss']:.4f}")
        print(f"  LeCam Loss: {d_metrics['lecam_loss']:.4f}")


# 超参数调整建议
LECAM_HYPERPARAMS = {
    "lecam_lambda": {
        "small_dataset": 0.001,  # 小数据集使用较小的权重
        "medium_dataset": 0.0001,
        "large_dataset": 0.00001
    },
    "gamma": {
        "default": 1.0,  # 通常使用1.0
        "stable_training": 0.5,  # 更稳定的训练
        "aggressive": 2.0  # 更强的约束
    }
}

if __name__ == "__main__":
    # 运行示例
    example_usage()