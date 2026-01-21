import torch
import torch.nn as nn


class RegressionTransformer(nn.Module):
    def __init__(
        self,
        n_tokens,
        in_dim,
        d_model,
        n_head,
        n_layers,
        d_ff,
        dropout=0.1,
    ):
        super().__init__()

        self.n_tokens = n_tokens
        self.d_model = d_model

        # feature embedding
        self.feature_proj = nn.Linear(in_dim, d_model)

        # 🔥 stock id embedding (positional encoding 대체)
        self.stock_embedding = nn.Embedding(n_tokens, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )

        self.head = nn.Linear(d_model, 1)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.stock_embedding.weight, std=0.02)

    def forward(self, x):
        """
        x: [B, N, F]
        """
        B, N, _ = x.shape
        device = x.device

        h = self.feature_proj(x)  # [B, N, D]

        stock_ids = torch.arange(N, device=device)
        stock_emb = self.stock_embedding(stock_ids)[None, :, :]  # [1, N, D]

        h = h + stock_emb
        h = self.encoder(h)

        out = self.head(h).squeeze(-1)  # [B, N]
        return out