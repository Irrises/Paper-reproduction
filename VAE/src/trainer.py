import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from src.utils import visualize_float_result, visualize_latent_space

class Trainer:
    def __init__(self, config, model, train_loader, test_loader, device):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.logger = SummaryWriter(self.config['paths']['log_path'])
        self.optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
        self.image_size = config['image_size']
        self.loss_type = config.get('loss_type', 'bce')  # ✅ 新增

    # ✅ 原论文严格版损失函数（ELBO）
    def loss(self, src_image, recon_image, mu, log_var):
        src_image = src_image.view(src_image.size(0), -1)
        recon_image = recon_image.view(recon_image.size(0), -1)

        # 1️⃣ 重建误差：原论文采用二元交叉熵
        if self.loss_type == 'bce':
            recon_loss = F.binary_cross_entropy(recon_image, src_image, reduction='sum')
        elif self.loss_type == 'mse':
            recon_loss = F.mse_loss(recon_image, src_image, reduction='sum')
        else:
            raise ValueError(f"Invalid loss type: {self.loss_type}")

        # 2️⃣ KL散度：严格匹配论文推导
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        # ✅ ELBO = 重建项 + KL项
        return recon_loss, kl_div

    def train(self):
        num_epochs = self.config['num_epochs']
        global_step = 0

        for epoch in range(num_epochs):
            self.model.train()
            for i, (images, _) in enumerate(self.train_loader):
                images = images.to(self.device)
                self.optimizer.zero_grad()

                recon_x, mu, log_var, _ = self.model(images)
                recon_loss, kl_div = self.loss(images, recon_x, mu, log_var)
                total_loss = recon_loss + kl_div
                total_loss.backward()
                self.optimizer.step()

                # ✅ TensorBoard记录
                self.logger.add_scalar('loss/total', total_loss.item(), global_step)
                self.logger.add_scalar('loss/reconstruction', recon_loss.item(), global_step)
                self.logger.add_scalar('loss/kl_divergence', kl_div.item(), global_step)
                global_step += 1

                if i % 100 == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(self.train_loader)}], "
                          f"Recon: {recon_loss.item():.2f}, KL: {kl_div.item():.2f}")

            # ✅ 每10个epoch保存样本
            if (epoch + 1) % 10 == 0:
                self.save_samples(epoch)
            self.save_model(epoch)

    def save_model(self, epoch):
        ckpt_dir = self.config['paths']['ckpt_path']
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(ckpt_dir, f"model_epoch_{epoch + 1}.pth"))

    def save_samples(self, epoch):
        vis_dir = self.config['paths']['visualization_path']
        os.makedirs(vis_dir, exist_ok=True)
        with torch.no_grad():
            samples = self.model.sample(16)
            samples = samples.cpu().reshape(16, self.image_size, self.image_size)
        fig, axs = plt.subplots(4, 4, figsize=(8, 8))
        axs = visualize_float_result(samples, axs)
        plt.savefig(os.path.join(vis_dir, f"epoch_{epoch+1}_samples.png"), bbox_inches='tight')
        plt.close(fig)
