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

    def forward(self, x, self_mask_prob):
        """
        x: [B, N, F]
        """
        B, N, _ = x.shape
        device = x.device

        # stock id embedding
        stock_ids = torch.arange(N, device=device)
        stock_emb = self.stock_embedding(stock_ids)[None, :, :]
        x = x + stock_emb

        # feature embedding
        h = self.feature_proj(x)  # [B, N, d_model]

        # =========================
        # 🔥 self-attention masking
        # =========================
        if self.training and self_mask_prob > 0:
            # [B, N]에서 각 token이 자기 자신을 볼지 말지 결정
            mask_flag = torch.rand(B, N, device=device) < self_mask_prob  # True면 막음

            # attention mask: [B, N, N]
            attn_mask = torch.zeros(B, N, N, device=device)

            # diagonal만 -inf 처리
            for b in range(B):
                idx = torch.arange(N, device=device)
                attn_mask[b, idx, idx] = torch.where(
                    mask_flag[b],
                    torch.tensor(float('-inf'), device=device),
                    torch.tensor(0.0, device=device)
                )

            # transformer expects [B*n_head, N, N] or [N,N]
            # -> batch별 다르게 하려면 expand 필요
            attn_mask = attn_mask.repeat_interleave(self.encoder.layers[0].self_attn.num_heads, dim=0)

            h = self.encoder(h, mask=attn_mask)

        else:
            h = self.encoder(h)

        # =========================
        # heads
        # =========================
        out1 = self.head(h).squeeze(-1)
        out2 = self.confidence_head(h).squeeze(-1)

        return out1, out2