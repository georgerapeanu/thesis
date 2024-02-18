import torch
import torch.nn as nn

class TransformerDecoderBlock(nn.Module):
    def __init__(self, ff_inner_channels: int, encoder_embed_dims: int, decoder_embed_dims: int, num_heads: int, max_length: int):
        super(TransformerDecoderBlock, self).__init__()
        self.mha1 = torch.nn.MultiheadAttention(
            embed_dim=decoder_embed_dims,
            num_heads=num_heads,
            batch_first=True
        )
        self.layer_norm1 = torch.nn.LayerNorm(decoder_embed_dims)

        self.mha2 = torch.nn.MultiheadAttention(
            embed_dim=decoder_embed_dims,
            kdim=encoder_embed_dims,
            vdim=encoder_embed_dims,
            num_heads=num_heads,
            batch_first=True
        )

        self.layer_norm2 = torch.nn.LayerNorm(decoder_embed_dims)
        self.feed_forward = torch.nn.Sequential(
            nn.Linear(in_features=decoder_embed_dims, out_features=ff_inner_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=ff_inner_channels, out_features=decoder_embed_dims)
        )
        self.layer_norm3 = torch.nn.LayerNorm(decoder_embed_dims)
        attn_mask = (torch.tril(torch.ones(max_length, max_length)) == 0)
        self.register_buffer('attn_mask', attn_mask)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, encoder_output: torch.Tensor, X: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        X = self.layer_norm1(X + self.dropout(self.mha1(query=X, key=X, value=X, need_weights=False, key_padding_mask=pad_mask, attn_mask=self.attn_mask[:X.size(1), :X.size(1)], is_casual=True)[0]))
        X = self.layer_norm2(X + self.dropout(self.mha2(query=X, key=encoder_output, value=encoder_output, need_weights=False)[0]))
        X = self.layer_norm3(X + self.dropout(self.feed_forward(X)))
        return X
