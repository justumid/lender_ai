
import torch
import torch.nn as nn

class SequenceFeatureEncoder(nn.Module):
    """
    📊 Learns self-represented features from salary/tax sequence using LSTM.
    Output: deep embedding [D] used for fraud, risk, limit, etc.
    """
    def __init__(self, input_dim=2, hidden_dim=64, embedding_dim=128):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.project = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, T, 2) — salary + tax
        _, (h_n, _) = self.lstm(x)          # (1, B, H)
        embedding = self.project(h_n[-1])   # (B, D)
        return embedding
