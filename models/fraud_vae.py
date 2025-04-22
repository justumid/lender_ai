import torch
import torch.nn as nn
import torch.nn.functional as F

class FraudVAE(nn.Module):
    def __init__(self, input_dim=10, seq_len=12, latent_dim=8):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        flattened = input_dim * seq_len  # 12×10 = 120 if combined salary+credit

        self.encoder = nn.Sequential(
            nn.Linear(flattened, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, flattened),
            nn.Sigmoid()  # use sigmoid for bounded output
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        # x: [B, 12, 10] → flatten
        x_flat = x.view(x.size(0), -1)
        mu, logvar = self.encode(x_flat)
        z = self.reparameterize(mu, logvar)
        recon_flat = self.decode(z)
        recon = recon_flat.view(x.size())  # [B, 12, 10]
        return recon, mu, logvar

    def anomaly_score(self, x):
        recon, _, _ = self.forward(x)
        loss = F.mse_loss(recon, x, reduction='none').mean(dim=(1, 2))  # mean over features + time
        return loss  # [B]
