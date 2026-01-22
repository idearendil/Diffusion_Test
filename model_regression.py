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

        # feature projection
        self.feature_proj = nn.Linear(in_dim, d_model)

        # 🔥 Stock ID embedding (positional encoding 대체)
        self.stock_embedding = nn.Embedding(n_tokens, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)

        self.head = nn.Linear(d_model, 1)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.stock_embedding.weight, std=0.02)

    def forward(self, x, key_padding_mask=None):
        """
        x: [B, N, F]
        """
        B, N, _ = x.shape
        device = x.device

        # feature embedding
        h = self.feature_proj(x)  # [B, N, d_model]

        # stock id embedding
        stock_ids = torch.arange(N, device=device)
        stock_emb = self.stock_embedding(stock_ids)[None, :, :]  # [1, N, d_model]

        h = h + stock_emb

        h = self.encoder(h, src_key_padding_mask=key_padding_mask)

        out = self.head(h).squeeze(-1)  # [B, N]
        return out