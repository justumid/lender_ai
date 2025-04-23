import torch
import torch.nn as nn
import torch.nn.functional as F


class RiskClassifierHead(nn.Module):
    """
    Predicts binary credit risk score (probability between 0 and 1).
    Used optionally in semi-supervised evaluation or downstream scoring.
    """
    def __init__(self, input_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Dropout(p=0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return torch.sigmoid(out).squeeze(1)  # [B]


class LoanLimitRegressor(nn.Module):
    """
    Predicts loan limit (regression head).
    Can be used with pseudo-labels or auto-learned representations.
    """
    def __init__(self, input_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(p=0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)  # [B]


class FraudDetectorHead(nn.Module):
    """
    Optional fraud detection head — predicts fraud score as probability.
    Can be connected to embeddings from encoder or VAE.
    """
    def __init__(self, input_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Dropout(p=0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return torch.sigmoid(out).squeeze(1)  # [B]
