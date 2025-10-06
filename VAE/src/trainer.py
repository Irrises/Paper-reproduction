# src/trainer.py
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from src.utils import visualize_samples, visualize_latent_space

class Trainer:
    def __init__(self, model, loaders, config):
        self.model = model
        self.train_loader, self.test_loader = loaders
        self.config = config
        self.device = next(model.parameters()).device
        paths = config["paths"]
        os.makedirs(paths["ckpt_path"], exist_ok=True)
        os.makedirs(paths["log_path"], exist_ok=True)
        os.makedirs(paths["visualization_path"], exist_ok=True)
        self.writer = SummaryWriter(paths["log_path"])
        self.optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    def loss(self, src, recon, mu, log_var):
        src = src.reshape(src.shape[0], -1)
        recon = recon.reshape(recon.shape[0], -1)
        recon_loss = F.mse_loss(src, recon, reduction='sum')
        kl_div = torch.mean(0.5 * (log_var.exp() + mu.pow(2) - log_var - 1).sum(dim=1))
        return recon_loss, kl_div

    def train(self):
        num_epochs = self.config["num_epochs"]
        for epoch in range(num_epochs):
            self.model.train()
            for i, (images, labels) in enumerate(self.train_loader):
                images = images.to(self.device)
                self.optimizer.zero_grad()
                recon, mu, log_var, z = self.model(images)
                recon_loss, kl_div = self.loss(images, recon, mu, log_var)
                loss = 0.01 * recon_loss + kl_div
                loss.backward()
                self.optimizer.step()
                if i % 100 == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs}] Step [{i+1}/{len(self.train_loader)}] "
                          f"recon: {recon_loss.item():.2f}, kl: {kl_div.item():.2f}")
            self.save_model(epoch)
            self.evaluate(epoch)

    def save_model(self, epoch):
        path = os.path.join(self.config["paths"]["ckpt_path"], f"model_epoch_{epoch+1}.pth")
        torch.save(self.model.state_dict(), path)
        print(f"✅ 模型已保存至 {path}")

    def evaluate(self, epoch):
        self.model.eval()
        with torch.no_grad():
            samples = self.model.sample(16)
            visualize_samples(samples, epoch, self.config["paths"]["visualization_path"], self.config["image_size"])
            for (images, labels) in self.test_loader:
                images = images.to(self.device)
                _, _, _, latents = self.model(images)
                visualize_latent_space(latents, labels, epoch, self.writer)
                break
