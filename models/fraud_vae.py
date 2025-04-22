import torch
import torch.nn as nn
import torch.nn.functional as F


class FraudVAE(nn.Module):
    """
    🕵️ Variational Autoencoder for fraud anomaly detection
    """
    def __init__(self, input_dim: int = 22, hidden_dim: int = 64, latent_dim: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def encode(self, x: torch.Tensor):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor):
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

    def anomaly_score(self, x: torch.Tensor):
        """
        Computes reconstruction MSE as fraud risk signal
        """
        x_recon, _, _ = self.forward(x)
        return F.mse_loss(x_recon, x, reduction='none').mean(dim=1)
