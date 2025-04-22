
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class DeepCreditEncoder(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128, n_heads=4, ff_dim=256, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=1,
                            batch_first=True, bidirectional=True)

        self.pos_encoder = PositionalEncoding(d_model=hidden_dim * 2)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim * 2,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_layer = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

    def forward(self, x, mask=None):
        lstm_out, _ = self.lstm(x)
        encoded = self.pos_encoder(lstm_out)

        if mask is not None:
            encoded = self.transformer(encoded, src_key_padding_mask=mask)
        else:
            encoded = self.transformer(encoded)

        cls_vector = encoded[:, 0]
        return self.output_layer(cls_vector)
