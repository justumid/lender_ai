import torch
import torch.nn as nn

class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attn_weights = None

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2, attn = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=True,
            average_attn_weights=False
        )
        self.attn_weights = attn
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        return self.norm2(src)

class PositionalEncoding(nn.Module):
    def __init__(self, dim_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, dim_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim_model, 2) * (-torch.log(torch.tensor(10000.0)) / dim_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, dim_model]

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class DeepCreditEncoder(nn.Module):
    def __init__(
        self,
        salary_input_dim=5,
        credit_input_dim=10,
        hidden_dim=128,
        nhead=4,
        nlayers=2,
        dropout=0.1,
        return_attention=False
    ):
        super().__init__()
        self.return_attention = return_attention

        # LSTMs
        self.salary_lstm = nn.LSTM(salary_input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.credit_lstm = nn.LSTM(credit_input_dim, hidden_dim, batch_first=True, bidirectional=True)

        # Positional encoding
        self.salary_pos = PositionalEncoding(hidden_dim * 2)
        self.credit_pos = PositionalEncoding(hidden_dim * 2)

        # Transformers
        self.salary_transformer = nn.ModuleList([
            CustomTransformerEncoderLayer(
                d_model=hidden_dim * 2,
                nhead=nhead,
                batch_first=True,
                dropout=dropout
            ) for _ in range(nlayers)
        ])
        self.credit_transformer = nn.ModuleList([
            CustomTransformerEncoderLayer(
                d_model=hidden_dim * 2,
                nhead=nhead,
                batch_first=True,
                dropout=dropout
            ) for _ in range(nlayers)
        ])

        # Fusion: salary + credit
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )

    def forward(self, salary_seq: torch.Tensor, credit_seq: torch.Tensor):
        # LSTM layers
        salary_out, _ = self.salary_lstm(salary_seq)
        credit_out, _ = self.credit_lstm(credit_seq)

        # Positional Encoding
        salary_encoded = self.salary_pos(salary_out)
        credit_encoded = self.credit_pos(credit_out)

        # Transformer layers
        for layer in self.salary_transformer:
            salary_encoded = layer(salary_encoded)
        for layer in self.credit_transformer:
            credit_encoded = layer(credit_encoded)

        # Global average pooling
        salary_pooled = salary_encoded.mean(dim=1)
        credit_pooled = credit_encoded.mean(dim=1)

        # Fusion
        fused = torch.cat([salary_pooled, credit_pooled], dim=1)
        output = self.fusion(fused)

        return output
