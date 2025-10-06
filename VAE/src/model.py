# src/model.py
import torch
import torch.nn as nn

class VariationalAutoencoder(nn.Module):
    def __init__(self, image_size, input_channel, device, latent_dim=2):
        super(VariationalAutoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.device = device
        self.encoder = nn.Sequential(
            nn.Linear(input_channel, input_channel // 2),
            nn.ReLU(),
            nn.Linear(input_channel // 2, input_channel // 4),
            nn.ReLU(),
            nn.Linear(input_channel // 4, latent_dim * 2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_channel // 4),
            nn.ReLU(),
            nn.Linear(input_channel // 4, input_channel // 2),
            nn.ReLU(),
            nn.Linear(input_channel // 2, input_channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterization(mu, log_var)
        recon_x = self.decode(z)
        return recon_x, mu, log_var, z

    def encode(self, x):
        bs = x.shape[0]
        x = x.reshape(bs, -1)
        encoder_output = self.encoder(x)
        mu = encoder_output[:, :self.latent_dim]
        log_var = encoder_output[:, self.latent_dim:]
        return mu, log_var

    def reparameterization(self, mu, log_var):
        epsilon = torch.randn_like(mu)
        z = mu + torch.sqrt(log_var.exp()) * epsilon
        return z

    def decode(self, z):
        bs = z.shape[0]
        recon_x = self.decoder(z).reshape(bs, 1, self.image_size, self.image_size)
        return recon_x

    def sample(self, sample_num):
        z = torch.randn((sample_num, self.latent_dim)).to(self.device)
        sample_image = self.decode(z)
        return sample_image
