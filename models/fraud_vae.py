import torch
import torch.nn as nn
import torch.nn.functional as F


class FraudVAE(nn.Module):
    """
    Variational Autoencoder (VAE) used for fraud anomaly detection.
    """
    def __init__(self, input_dim=15, seq_len=12, latent_dim=16):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.flattened_dim = input_dim * seq_len  # e.g., 12 × 15 = 180

        # Encoder: [B, 180] → [B, latent_dim]
        self.encoder = nn.Sequential(
            nn.Linear(self.flattened_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

        # Decoder: [B, latent_dim] → [B, 180]
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, self.flattened_dim),
            nn.Sigmoid()  # normalized output
        )

    def encode(self, x: torch.Tensor):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor):
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        """
        x: [B, 12, 15] → flatten → encode → sample → decode → [B, 12, 15]
        """
        x_flat = x.view(x.size(0), -1)  # [B, 180]
        mu, logvar = self.encode(x_flat)
        z = self.reparameterize(mu, logvar)
        recon_flat = self.decode(z)
        recon = recon_flat.view(x.size())  # [B, 12, 15]
        return recon, mu, logvar

    def anomaly_score(self, x: torch.Tensor):
        """
        Compute MSE-based anomaly score per sample.
        """
        recon, _, _ = self.forward(x)
        return F.mse_loss(recon, x, reduction='none').mean(dim=(1, 2))  # [B]

    def vae_loss(self, x: torch.Tensor, beta: float = 1.0):
        """
        Computes VAE loss: reconstruction + β × KL divergence
        """
        recon, mu, logvar = self.forward(x)
        recon_loss = F.mse_loss(recon, x, reduction='mean')
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + beta * kl
