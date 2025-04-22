import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPProjectionHead(nn.Module):
    def __init__(self, input_dim: int, projection_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, projection_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimCLREncoder(nn.Module):
    """
    🔁 SimCLR Encoder for contrastive learning over tabular features.
    Output: L2-normalized embeddings.
    """
    def __init__(self, input_dim: int = 22, projection_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.projection = MLPProjectionHead(128, projection_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        z = self.projection(h)
        return F.normalize(z, dim=1)
