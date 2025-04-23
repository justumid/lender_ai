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
    def __init__(self, dim_model: int, max_len: int = 100):
        super().__init__()
        pe = torch.zeros(max_len, dim_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim_model, 2) * (-torch.log(torch.tensor(10000.0)) / dim_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # [1, max_len, dim_model]

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)


class DeepCreditEncoder(nn.Module):
    def __init__(
        self,
        salary_input_dim=5,
        credit_input_dim=10,
        hidden_dim=128,
        nhead=4,
        nlayers=2,
        return_attention=False
    ):
        super().__init__()
        self.return_attention = return_attention

        self.salary_lstm = nn.LSTM(salary_input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.credit_lstm = nn.LSTM(credit_input_dim, hidden_dim, batch_first=True, bidirectional=True)

        self.salary_pos = PositionalEncoding(dim_model=hidden_dim * 2)
        self.credit_pos = PositionalEncoding(dim_model=hidden_dim * 2)

        self.salary_transformer = nn.ModuleList([
            CustomTransformerEncoderLayer(d_model=hidden_dim * 2, nhead=nhead, batch_first=True)
            for _ in range(nlayers)
        ])
        self.credit_transformer = nn.ModuleList([
            CustomTransformerEncoderLayer(d_model=hidden_dim * 2, nhead=nhead, batch_first=True)
            for _ in range(nlayers)
        ])

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU()
        )

    def forward(self, salary_seq: torch.Tensor, credit_seq: torch.Tensor):
        # LSTM encoding
        salary_out, _ = self.salary_lstm(salary_seq)   # [B, T, 2H]
        credit_out, _ = self.credit_lstm(credit_seq)   # [B, T, 2H]

        # Positional encoding
        salary_encoded = self.salary_pos(salary_out)
        credit_encoded = self.credit_pos(credit_out)

        attn_s, attn_c = [], []

        for layer in self.salary_transformer:
            salary_encoded = layer(salary_encoded)
            if self.return_attention:
                attn_s.append(layer.attn_weights)

        for layer in self.credit_transformer:
            credit_encoded = layer(credit_encoded)
            if self.return_attention:
                attn_c.append(layer.attn_weights)

        # Use first token embedding (like CLS)
        salary_vec = salary_encoded[:, 0, :]  # [B, 2H]
        credit_vec = credit_encoded[:, 0, :]  # [B, 2H]

        fused = self.fusion(torch.cat([salary_vec, credit_vec], dim=1))  # [B, H]

        if self.return_attention:
            return fused, attn_s, attn_c
        return fused
