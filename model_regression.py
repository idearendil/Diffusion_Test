# model_regression.py
import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: [B, N, D]
        return x + self.pe[: x.size(1)].unsqueeze(0)


class RegressionTransformer(nn.Module):
    def __init__(
        self,
        n_tokens: int,
        in_dim: int,
        d_model: int,
        n_head: int,
        n_layers: int,
        d_ff: int,
        dropout: float,
    ):
        super().__init__()

        self.input_proj = nn.Linear(in_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.out = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: [B, N, F]
        h = self.input_proj(x)
        h = self.pos_enc(h)
        h = self.encoder(h)
        y = self.out(h).squeeze(-1)  # [B, N]
        return y