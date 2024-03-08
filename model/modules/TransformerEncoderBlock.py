import torch
import torch.nn as nn


class TransformerEncoderBlock(nn.Module):
    def __init__(self, ff_inner_channels: int, embed_dims: int, num_heads: int):
        super(TransformerEncoderBlock, self).__init__()
        self.mha = torch.nn.MultiheadAttention(
            embed_dim=embed_dims,
            num_heads=num_heads,
            batch_first=True
        )
        self.layer_norm1 = torch.nn.LayerNorm(embed_dims)
        self.feed_forward = torch.nn.Sequential(
            nn.Linear(in_features=embed_dims, out_features=ff_inner_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=ff_inner_channels, out_features=embed_dims)
        )
        self.layer_norm2 = torch.nn.LayerNorm(embed_dims)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.layer_norm1(X + self.dropout(self.mha(query=X, key=X, value=X, need_weights=False)[0]))
        X = self.layer_norm2(X + self.dropout(self.feed_forward(X)))
        return X
