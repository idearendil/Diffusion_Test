# model.py
import math
import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: [B] int64
        returns: [B, dim]
        """
        half = self.dim // 2
        freqs = torch.exp(
            torch.linspace(0, math.log(10000), half, device=t.device, dtype=torch.float32) * (-1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # [B, half]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(t.shape[0], 1, device=t.device)], dim=1)
        return emb


class DiffusionTransformer(nn.Module):
    """
    Input tokens: [B, N, (F+1)]  where first dim is y_t (noisy target), rest is conditioning x
    Output: [B, N] (v_pred)
    """
    def __init__(self, n_tokens: int, in_dim: int, d_model: int, n_head: int, n_layers: int, d_ff: int, dropout: float):
        super().__init__()
        self.n_tokens = n_tokens
        self.in_dim = in_dim
        self.d_model = d_model

        self.in_proj = nn.Linear(in_dim, d_model)

        # Learned positional encoding (token position = ticker position)
        self.pos_emb = nn.Parameter(torch.zeros(1, n_tokens, d_model))
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)

        # Time embedding -> project to d_model and add to all tokens
        self.time_emb = SinusoidalTimeEmbedding(d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.out_norm = nn.LayerNorm(d_model)
        self.pred_head = nn.Linear(d_model, 1)

    def forward(self, tokens: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        tokens: [B, N, in_dim]
        t: [B] int64
        returns: v_pred [B, N]
        """
        h = self.in_proj(tokens)     # [B,N,d]
        h = h + self.pos_emb         # positional encoding

        te = self.time_mlp(self.time_emb(t))  # [B,d]
        h = h + te.unsqueeze(1)      # add to all tokens

        h = self.encoder(h)          # [B,N,d]
        h = self.out_norm(h)
        v = self.pred_head(h).squeeze(-1)  # [B,N]
        return v