import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPProjectionHead(nn.Module):
    def __init__(self, input_dim: int, projection_dim: int = 64):
        """
        Projection head for SimCLR — used after encoder.
        Applies a 2-layer MLP with normalization and activation.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, projection_dim),
            nn.BatchNorm1d(projection_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimCLREncoder(nn.Module):
    def __init__(self, seq_len: int = 12, input_dim: int = 15, hidden_dim: int = 128, projection_dim: int = 64):
        """
        SimCLR encoder for time-series data of shape [B, seq_len, input_dim].
        Example: [B, 12, 15] flattened to [B, 180] → encoded → projected → L2 norm
        """
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.flattened_dim = seq_len * input_dim

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
        Args:
            x: Tensor of shape [B, 12, 15]
        Returns:
            L2-normalized projection: [B, projection_dim]
        """
        x_flat = x.view(x.size(0), -1)
        h = self.encoder(x_flat)
        z = self.projection(h)
        return F.normalize(z, dim=1)
