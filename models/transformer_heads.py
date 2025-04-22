import torch
import torch.nn as nn
import torch.nn.functional as F


class RiskClassifierHead(nn.Module):
    """
    🎯 Takes [D] embedding → binary approval prediction
    """
    def __init__(self, input_dim=128):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return torch.sigmoid(self.classifier(x)).squeeze(-1)  # (B,)


class LoanLimitRegressor(nn.Module):
    """
    💰 Takes [D] embedding → continuous loan limit
    """
    def __init__(self, input_dim=128):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.regressor(x).squeeze(-1)  # (B,)


class FraudDetectorHead(nn.Module):
    """
    🕵️ Takes [D] embedding → fraud risk probability
    """
    def __init__(self, input_dim=128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return torch.sigmoid(self.model(x)).squeeze(-1)  # (B,)
