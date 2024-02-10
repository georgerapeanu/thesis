import torch
import torch.nn as nn

from utils.configs import ModelConfig
from typing import *
import math

# Very good https://sungwookyoo.github.io/tips/study/Multihead_Attention/
# https://github.dev/karpathy/minGPT
# http://juditacs.github.io/2018/12/27/masked-attention.html

class PositionalEncoding1D(nn.Module):

    def __init__(self, max_len: int, d_model: int):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        self.Dropout = nn.Dropout(p=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.Dropout(x + self.pe[:x.size(1)])


# reference https://openjournals.uwaterloo.ca/index.php/vsl/article/view/3533/4579
class PositionalEncoding2D(nn.Module):

    def __init__(self, max_height: int, max_width: int, d_model: int):
        super().__init__()

        position_x = torch.arange(max_height).unsqueeze(1).broadcast_to((max_height, max_width)).unsqueeze(2)
        position_y = torch.arange(max_width).unsqueeze(0).broadcast_to((max_height, max_width)).unsqueeze(2)

        div_term = torch.exp(torch.arange(0, d_model, 4) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_height, max_width, d_model)
        pe[:, :, 0:2 * len(div_term):2] = torch.sin(position_x * div_term)
        pe[:, :, 1:2 * len(div_term):2] = torch.cos(position_x * div_term)
        pe[:, :, 2 * len(div_term)::2] = torch.sin(position_y * div_term)
        pe[:, :, 2 * len(div_term)::2] = torch.cos(position_y * div_term)
        self.register_buffer('pe', pe)
        self.Dropout = nn.Dropout(p=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.Dropout(x + self.pe[:x.size(1), :x.size(2)])


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, intermediary_channels: int):
        super(ResidualBlock, self).__init__()
        self.residual_layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=intermediary_channels, kernel_size=3, padding='same', bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=intermediary_channels, out_channels=in_channels, kernel_size=3, padding='same', bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(in_channels),
            nn.Dropout(p=0.1)
        )

    def forward(self, X):
        return self.residual_layer(X) + X


class EncoderBlock(nn.Module):
    def __init__(self, ff_inner_channels: int, embed_dims: int, num_heads: int):
        super(EncoderBlock, self).__init__()
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
        X = self.layer_norm1(X + self.dropout(self.mha(query=X, key=X, value=X, needs_weights=False)[0]))
        X = self.layer_norm2(X + self.dropout(self.feed_forward(X)))
        return X


class DecoderBlock(nn.Module):
    def __init__(self, ff_inner_channels: int, encoder_embed_dims: int, decoder_embed_dims: int, num_heads: int, max_length: int):
        super(DecoderBlock, self).__init__()
        self.mha1 = torch.nn.MultiheadAttention(
            embed_dim=decoder_embed_dims,
            num_heads=num_heads,
            batch_first=True
        )
        self.layer_norm1 = torch.nn.LayerNorm(decoder_embed_dims)

        self.mha2 = torch.nn.MultiheadAttention(
            embed_dim=decoder_embed_dims,
            kdim=encoder_embed_dims,
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
        X = self.layer_norm1(X + self.dropout(self.mha1(query=X, key=X, value=X, needs_weights=False, pad_mask=pad_mask, attn_mask=self.attn_mask[:X.size(1), :X.size(1)])[0]))
        X = self.layer_norm2(X + self.dropout(self.mha2(query=encoder_output, key=encoder_output, value=X, needs_weights=False, pad_mask=pad_mask)))
        X = self.layer_norm3(X + self.dropout(self.feed_forward(X)))
        return X


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super(Model, self).__init__()
        self.board_preparation = nn.Sequential(
            nn.Conv2d(in_channels=config['board_in_channels'], out_channels=config['board_embedding_size'], kernel_size=1, padding=0, bias=True),
            *[ResidualBlock(in_channels=config['board_embedding_size'], intermediary_channels=config['board_intermediary_channels']) for _ in range(config['conv_modules_count'])]
        )
        self.pe_board = PositionalEncoding2D(config['board_height'], config['board_width'], config['board_embedding_size'])
        self.pe_text = PositionalEncoding1D(config['data_config']['context_length'], config['text_embedding_size'])

        self.encoders = nn.ModuleList([EncoderBlock(ff_inner_channels=config['ff_inner_channels'], num_heads=config['num_heads'], embed_dims=config['board_embedding_size']) for _ in range(config['transformer_blocks'])])
        self.decoders = nn.ModuleList([DecoderBlock(ff_inner_channels=config['ff_inner_channels'], num_heads=config['num_heads'], decoder_embed_dims=config['text_embedding_size'], encoder_embed_dims=config['board_embedding_size']) for _ in range(config['transformer_blocks'])])
        self.linear = nn.Linear(in_features=config['text_embedding_size'], out_features=config['vocab_size'])

    def forward(self, X_board: torch.Tensor, X_text: torch.Tensor, padding_mask: torch.Tensor, targets: Optional[torch.Tensor] = None):
        X_board = self.board_preparation(X_board)
        X_board = X_board.permute(0, 2, 3, 1)
        b, _, _, ch = X_board.shape
        X_board = self.pe_board(X_board).view(b, -1, ch)
        X_text = self.pe_text(X_text)
        for decoder, encoder in zip(self.encoders, self.decoders):
            X_board = decoder(X_board)
            X_text = encoder(X_board, X_text, padding_mask)
        logits = self.linear(X_text)
        loss = None

        if targets is None:
            loss = torch.nn.functional.cross_entropy(logits, weight=(1 - padding_mask.int()))

        return logits, loss

