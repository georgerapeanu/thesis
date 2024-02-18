import torch
import torch.nn as nn

from model.modules.DepthwiseResidualBlock import DepthwiseResidualBlock
from model.modules.PositionalEncoding1D import PositionalEncoding1D
from model.modules.PositionalEncoding2D import PositionalEncoding2D
from model.modules.ResidualEncoder import ResidualEncoder
from model.modules.TransformerEncoderBlock import TransformerEncoderBlock
from model.modules.TransformerDecoderBlock import TransformerDecoderBlock
from utils.configs import ModelConfig, SharedConfig, MultiHeadConfig
from typing import *
import math

# Very good https://sungwookyoo.github.io/tips/study/Multihead_Attention/
# https://github.dev/karpathy/minGPT
# http://juditacs.github.io/2018/12/27/masked-attention.html



class Model(nn.Module):
    def __init__(self, config: ModelConfig, shared_config: SharedConfig):
        super(Model, self).__init__()
        self.board_preparation = nn.Sequential(
            nn.Conv2d(in_channels=config['board_in_channels'], out_channels=config['board_embedding_size'], kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(config['board_embedding_size']),
            nn.ReLU(inplace=True),
            *[DepthwiseResidualBlock(in_channels=config['board_embedding_size'], intermediary_channels=config['board_intermediary_channels']) for _ in range(config['conv_modules_count'])]
        )

        self.emb = torch.nn.Embedding(num_embeddings=shared_config['vocab_size'], embedding_dim=config['text_embedding_size'])

        self.pe_board = PositionalEncoding2D(config['board_height'], config['board_width'], config['board_embedding_size'])
        self.pe_text = PositionalEncoding1D(shared_config['context_length'], config['text_embedding_size'])

        self.encoders = nn.ModuleList([TransformerEncoderBlock(ff_inner_channels=config['ff_inner_channels'], num_heads=config['num_heads'], embed_dims=config['board_embedding_size']) for _ in range(config['transformer_blocks'])])
        self.decoders = nn.ModuleList([
            TransformerDecoderBlock(
                ff_inner_channels=config['ff_inner_channels'],
                num_heads=config['num_heads'],
                decoder_embed_dims=config['text_embedding_size'],
                encoder_embed_dims=config['board_embedding_size'],
                max_length=shared_config['context_length'])
            for _ in range(config['transformer_blocks'])
        ])
        self.linear = nn.Linear(in_features=config['text_embedding_size'], out_features=shared_config['vocab_size'])
        self.__config = config
        self.__shared_config = shared_config

    def forward(self, X_board: torch.Tensor, X_text: torch.Tensor, padding_mask: torch.Tensor, targets: Optional[torch.Tensor] = None):
        X_board = self.board_preparation(X_board)
        X_board = X_board.permute(0, 2, 3, 1)
        b, _, _, ch = X_board.shape
        X_board = self.pe_board(X_board).view(b, -1, ch)
        X_text = self.emb(X_text)
        X_text = self.pe_text(X_text)
        for encoder, decoder in zip(self.encoders, self.decoders):
            X_board = encoder(X_board)
            X_text = decoder(X_board, X_text, padding_mask)
        logits = self.linear(X_text)
        loss = None
        if targets is not None:
            log_logits = -torch.nn.functional.log_softmax(logits, dim=-1)
            log_logits = log_logits.masked_fill(padding_mask.unsqueeze(-1), 0)
            loss = torch.gather(log_logits, -1, targets.unsqueeze(-1)).sum() / (padding_mask == False).int().sum()

        return logits, loss

    def generate(self, X_board: torch.Tensor, X_text: torch.Tensor, max_new_tokens: int, device: str, temperature: float = 1.0, do_sample:bool = False) -> torch.Tensor:
        for _ in range(max_new_tokens):
            X_text_in = X_text if X_text.size(1) < self.__shared_config['context_length'] else X_text[:, -self.__shared_config['context_length']:]
            logits, _ = self(X_board, X_text_in, (torch.zeros(1, X_text_in.size(1)) == 1).to(device))
            logits = logits[:, -1, :] / temperature
            probs = torch.nn.functional.softmax(logits, dim=-1)
            if do_sample is not None:
                text_next = torch.multinomial(probs, num_samples=1)
            else:
                _, text_next = torch.topk(probs, k=1, dim=-1)
            X_text = torch.cat([X_text, text_next], dim=1)
            if text_next == self.__shared_config['eos_id']:
                break
        return X_text



class ModelResidualEncoder(torch.nn.Module):
    def __init__(self, config: ModelConfig, shared_config: SharedConfig):
        super(ModelResidualEncoder, self).__init__()
        self.__config = config
        self.__shared_config = shared_config

        self.emb = torch.nn.Embedding(num_embeddings=shared_config['vocab_size'], embedding_dim=config['text_embedding_size'])
        self.pe_text = PositionalEncoding1D(shared_config['context_length'], config['text_embedding_size'])
        self.pe_board = PositionalEncoding2D(config['board_height'], config['board_width'], config['board_embedding_size'])

        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=config['board_in_channels'], out_channels=config['board_embedding_size'], kernel_size=1, padding='same', bias=True),
                nn.BatchNorm2d(config['board_embedding_size']),
                nn.ReLU(inplace=True)
            ),
            *[ResidualEncoder(block_count=config['conv_modules_count'], in_channels=config['board_embedding_size'] * 2 ** i, intermediary_channels=config['board_intermediary_channels'] * 2 ** i) for i in range(config['transformer_blocks'] - 1)]
        ])

        self.decoders = nn.ModuleList([
            TransformerDecoderBlock(
                ff_inner_channels=config['ff_inner_channels'],
                num_heads=config['num_heads'],
                decoder_embed_dims=config['text_embedding_size'],
                encoder_embed_dims=config['board_embedding_size'] * 2 ** i,
                max_length=shared_config['context_length'])
            for i in range(config['transformer_blocks'])
        ])

        self.linear = nn.Linear(in_features=config['text_embedding_size'], out_features=shared_config['vocab_size'])

    def forward(self, X_board: torch.Tensor, X_text: torch.Tensor, padding_mask: torch.Tensor, targets: Optional[torch.Tensor] = None):
        X_text = self.emb(X_text)
        X_text = self.pe_text(X_text)

        for i, (encoder, decoder) in enumerate(zip(self.encoders, self.decoders)):
            X_board = encoder(X_board)
            if i == 0:
                X_board = X_board.permute(0, 2, 3, 1)
                X_board = self.pe_board(X_board).permute(0, 3, 1, 2)
            b, ch, _, _ = X_board.size()
            X_text = decoder(X_board.permute(0, 2, 3, 1).view(b, -1, ch), X_text, padding_mask) #test adding pe at each step

        logits = self.linear(X_text)
        loss = None
        if targets is not None:
            log_logits = -torch.nn.functional.log_softmax(logits, dim=-1)
            log_logits = log_logits.masked_fill(padding_mask.unsqueeze(-1), 0)
            loss = torch.gather(log_logits, -1, targets.unsqueeze(-1)).sum() / (padding_mask == False).int().sum()

        return logits, loss

    def generate(self, X_board: torch.Tensor, X_text: torch.Tensor, max_new_tokens: int, device: str, temperature: float = 1.0, do_sample:bool = False) -> torch.Tensor:
        for _ in range(max_new_tokens):
            X_text_in = X_text if X_text.size(1) < self.__shared_config['context_length'] else X_text[:, -self.__shared_config['context_length']:]
            logits, _ = self(X_board, X_text_in, (torch.zeros(1, X_text_in.size(1)) == 1).to(device))
            logits = logits[:, -1, :] / temperature
            probs = torch.nn.functional.softmax(logits, dim=-1)
            if do_sample is not None:
                text_next = torch.multinomial(probs, num_samples=1)
            else:
                _, text_next = torch.topk(probs, k=1, dim=-1)
            X_text = torch.cat([X_text, text_next], dim=1)
            if text_next == self.__shared_config['eos_id']:
                break
        return X_text


class MultipleHeadsModel(nn.Module):
    def __init__(self, config: MultiHeadConfig, shared_config: SharedConfig):
        super(MultipleHeadsModel, self).__init__()

        self.board_preparation = nn.Sequential(
            nn.Conv2d(in_channels=config['board_in_channels'],
                      out_channels=config['board_embedding_size'],
                      kernel_size=1,
                      padding=0,
                      bias=False
            ),
            nn.BatchNorm2d(config['board_embedding_size']),
            nn.ReLU(inplace=True),
        )

        self.__config = config
        self.__shared_config = shared_config

        self.emb = torch.nn.Embedding(num_embeddings=shared_config['vocab_size'], embedding_dim=config['text_embedding_size'])
        self.pe_text = PositionalEncoding1D(shared_config['context_length'], config['text_embedding_size'])
        self.pe_board = PositionalEncoding2D(config['board_height'], config['board_width'], config['board_embedding_size'])

        self.encoders = nn.ModuleList([
            TransformerEncoderBlock(ff_inner_channels=config['ff_inner_channels'],
                                    num_heads=config['num_heads'],
                                    embed_dims=config['board_embedding_size'])
            for _ in range(config['transformer_blocks'])
        ])

        self.decoders = nn.ModuleList([
            TransformerDecoderBlock(
                ff_inner_channels=config['ff_inner_channels'],
                num_heads=config['num_heads'],
                decoder_embed_dims=config['text_embedding_size'],
                encoder_embed_dims=config['board_embedding_size'],
                max_length=shared_config['context_length'])
            for i in range(config['transformer_blocks'])
        ])

        self.linears = nn.ModuleList([
            nn.Linear(in_features=config['text_embedding_size'], out_features=shared_config['vocab_size'])
            for _ in range(len(config['target_types_and_depth']))
        ])

        self.final_linear = nn.Linear(in_features=config['text_embedding_size'], out_features=shared_config['vocab_size'])

    def forward(self,
                X_board: torch.Tensor,
                X_text: torch.Tensor,
                padding_mask: torch.Tensor,
                targets: Optional[torch.Tensor] = None,
                is_type: Optional[torch.Tensor] = None #shape (b, types)
        ):
        X_board = self.board_preparation(X_board)
        X_board = X_board.permute(0, 2, 3, 1)
        b, _, _, ch = X_board.shape
        X_board = self.pe_board(X_board).view(b, -1, ch)

        X_text = self.emb(X_text)
        X_text = self.pe_text(X_text)

        decoder_outputs = []
        for i, (encoder, decoder) in enumerate(zip(self.encoders, self.decoders)):
            X_board = encoder(X_board)
            X_text = decoder(X_board, X_text, padding_mask) #test adding pe at each step
            decoder_outputs.append(X_text)

        final_logits = self.final_linear(X_text)
        loss = None
        count = (padding_mask == False).int().sum().item()
        if targets is not None:
            loss = torch.Tensor([0]).to(final_logits.device)
            for (type, depth) in self.__config['target_types_and_depth']:
                idx = is_type[:, type]
                my_logits = self.linears[type](decoder_outputs[depth][idx])
                my_log_logits = -torch.nn.functional.log_softmax(my_logits, dim=-1)
                my_log_logits = my_log_logits.masked_fill(padding_mask[idx].unsqueeze(-1), 0)
                loss += torch.gather(my_log_logits, -1, targets[idx].unsqueeze(-1)).sum() / count
            log_logits = -torch.nn.functional.log_softmax(final_logits, dim=-1)
            log_logits = log_logits.masked_fill(padding_mask.unsqueeze(-1), 0)
            loss += torch.gather(log_logits, -1, targets.unsqueeze(-1)).sum() / count

        return final_logits, loss

    def generate(self, X_board: torch.Tensor, X_text: torch.Tensor, max_new_tokens: int, device: str, temperature: float = 1.0, do_sample:bool = False) -> torch.Tensor:
        for _ in range(max_new_tokens):
            X_text_in = X_text if X_text.size(1) < self.__shared_config['context_length'] else X_text[:, -self.__shared_config['context_length']:]
            logits, _ = self(X_board, X_text_in, (torch.zeros(1, X_text_in.size(1)) == 1).to(device))
            logits = logits[:, -1, :] / temperature
            probs = torch.nn.functional.softmax(logits, dim=-1)
            if do_sample is not None:
                text_next = torch.multinomial(probs, num_samples=1)
            else:
                _, text_next = torch.topk(probs, k=1, dim=-1)
            X_text = torch.cat([X_text, text_next], dim=1)
            if text_next == self.__shared_config['eos_id']:
                break
        return X_text
