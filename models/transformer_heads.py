import torch
import torch.nn as nn
import torch.nn.functional as F


class RiskClassifierHead(nn.Module):
    """
    Binary classifier for credit risk score — outputs value ∈ [0, 1].
    """
    def __init__(self, input_dim: int = 128, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)  # [B, 1]
        return torch.sigmoid(out).squeeze(1)  # [B]


class LoanLimitRegressor(nn.Module):
    """
    Regression head for predicting loan limit amount.
    """
    def __init__(self, input_dim: int = 128, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)  # [B]


class FraudDetectorHead(nn.Module):
    """
    Binary fraud detection head — outputs probability ∈ [0, 1].
    Can use encoder or VAE/SimCLR embeddings.
    """
    def __init__(self, input_dim: int = 128, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # input_dim is now 128
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(p=0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)  # [B, 1]
        return torch.sigmoid(out).squeeze(1)  # [B]
