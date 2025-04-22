import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPProjectionHead(nn.Module):
    def __init__(self, input_dim, projection_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, projection_dim)
        )

    def forward(self, x):
        return self.net(x)

class SimCLREncoder(nn.Module):
    def __init__(self, seq_len=12, input_dim=10, hidden_dim=128, projection_dim=64):
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
        # x: [B, 12, 10] → flatten
        x_flat = x.view(x.size(0), -1)  # [B, 120]
        h = self.encoder(x_flat)       # [B, hidden_dim]
        z = self.projection(h)         # [B, projection_dim]
        return F.normalize(z, dim=1)   # L2-normalized vector
