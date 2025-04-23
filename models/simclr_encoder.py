import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPProjectionHead(nn.Module):
    def __init__(self, input_dim: int, projection_dim: int = 64):
        """
        Projection head used in SimCLR after encoder.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, projection_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimCLREncoder(nn.Module):
    def __init__(self, seq_len: int = 12, input_dim: int = 15, hidden_dim: int = 128, projection_dim: int = 64):
        """
        SimCLR encoder network.
        input_dim = 15: sequence features (e.g., 5 salary + 10 credit features)
        seq_len = 12: sequence length (time steps)
        """
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.flattened_dim = seq_len * input_dim  # e.g., 12 × 15 = 180

        self.encoder = nn.Sequential(
            nn.Linear(self.flattened_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.projection = MLPProjectionHead(hidden_dim, projection_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through encoder and projection head.

        Args:
            x (Tensor): Input tensor of shape [B, 12, 15]

        Returns:
            Tensor: Normalized embedding of shape [B, projection_dim]
        """
        x_flat = x.view(x.size(0), -1)       # Flatten [B, 12, 15] -> [B, 180]
        h = self.encoder(x_flat)             # Encoder output [B, hidden_dim]
        z = self.projection(h)               # Projected to [B, projection_dim]
        return F.normalize(z, dim=1)         # L2-normalize for contrastive loss
