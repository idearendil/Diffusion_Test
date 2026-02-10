import torch
import torch.nn as nn



class RegressionLSTMTransformer(nn.Module):
    def __init__(
        self,
        n_tokens,
        in_dim,
        d_model,
        n_head,
        n_layers,
        d_ff,
        lstm_layers=1,
        dropout=0.1,
    ):
        super().__init__()

        self.n_tokens = n_tokens
        self.d_model = d_model

        # 🔥 Temporal encoder (per-token)
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=d_model,
            num_layers=lstm_layers,
            batch_first=True,
        )

        # 🔥 Stock embedding (cross-sectional identity)
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
        self.confidence_head = nn.Linear(d_model, 1)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.stock_embedding.weight, std=0.02)

    def forward(self, x):
        """
        x: [B, T, N, F]
        """

        B, T, N, F = x.shape
        device = x.device

        # =========================
        # 1. Temporal LSTM (per token)
        # =========================
        x = x.permute(0, 2, 1, 3)          # [B, N, T, F]
        x = x.reshape(B * N, T, F)         # [B*N, T, F]

        _, (h_n, _) = self.lstm(x)         # h_n: [L, B*N, d_model]
        h = h_n[-1]                        # last layer → [B*N, d_model]

        h = h.view(B, N, self.d_model)     # [B, N, d_model]

        # =========================
        # 2. Add stock embedding
        # =========================
        stock_ids = torch.arange(N, device=device)
        stock_emb = self.stock_embedding(stock_ids)[None, :, :]
        h = h + stock_emb

        # =========================
        # 3. Cross-sectional Transformer
        # =========================
        h = self.encoder(h)

        # =========================
        # 4. Heads
        # =========================
        out = self.head(h).squeeze(-1)                   # [B, N]
        confidence = torch.softmax(
            self.confidence_head(h).squeeze(-1), dim=1
        )

        return out, confidence


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
        self.stock_embedding = nn.Embedding(n_tokens, in_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)

        self.head = nn.Linear(d_model, 1)
        self.confidence_head = nn.Linear(d_model, 1)
        
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.stock_embedding.weight, std=0.02)

    def forward(self, x):
        """
        x: [B, N, F]
        """
        B, N, _ = x.shape
        device = x.device

        # stock id embedding
        stock_ids = torch.arange(N, device=device)
        stock_emb = self.stock_embedding(stock_ids)[None, :, :]  # [1, N, in_dim]

        x = x + stock_emb

        # feature embedding
        h = self.feature_proj(x)  # [B, N, d_model]

        h = self.encoder(h)

        out = self.head(h).squeeze(-1)  # [B, N]
        confidence = torch.softmax(self.confidence_head(h).squeeze(-1), dim=1)  # [B, N]
        return out, confidence