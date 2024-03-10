from io import BytesIO

import chess
import chess.svg
import torch
import torch.nn as nn
import torchmetrics
import wandb
from PIL import Image
from cairosvg import svg2png
from lightning.pytorch.core.module import MODULE_OPTIMIZERS
from lightning.pytorch.core.optimizer import LightningOptimizer
from lightning.pytorch.utilities.types import STEP_OUTPUT, LRSchedulerPLType, OptimizerLRScheduler
from omegaconf import DictConfig
from torch.optim import Optimizer
from tqdm import tqdm

from data.ActualBoardDataModule import ActualBoardDataModule
from model.modules.DepthwiseResidualBlock import DepthwiseResidualBlock
from model.modules.PositionalEncoding1D import PositionalEncoding1D
from model.modules.PositionalEncoding2D import PositionalEncoding2D
from model.modules.ResidualEncoder import ResidualEncoder
from model.modules.TransformerEncoderBlock import TransformerEncoderBlock
from model.modules.TransformerDecoderBlock import TransformerDecoderBlock
from model.predict import AlphaZeroPredictor, ActualBoardPredictor
from utils.configs import ModelConfig, SharedConfig, MultiHeadConfig
from typing import *
import math
import lightning as L

# Very good https://sungwookyoo.github.io/tips/study/Multihead_Attention/
# https://github.dev/karpathy/minGPT
# http://juditacs.github.io/2018/12/27/masked-attention.html


class AlphazeroTransformerModel(L.LightningModule):
    def __init__(
            self,
            board_in_channels: int,
            board_embedding_size: int,
            board_intermediary_channels: int,
            vocab_size: int,
            text_embedding_size: int,
            board_height: int,
            board_width: int,
            context_length: int,
            ff_inner_channels: int,
            num_heads: int,
            transformer_blocks: int,
            conv_modules_count: int,
            eos_id: int,
            optimizer: str,
            lr: float
    ):
        super(AlphazeroTransformerModel, self).__init__()
        self.save_hyperparameters()

        self.board_preparation = nn.Sequential(
            nn.Conv2d(in_channels=board_in_channels, out_channels=board_embedding_size, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(board_embedding_size),
            nn.ReLU(inplace=True),
            *[DepthwiseResidualBlock(in_channels=board_embedding_size, intermediary_channels=board_intermediary_channels) for _ in range(conv_modules_count)]
        )

        self.emb = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=text_embedding_size)

        self.pe_board = PositionalEncoding2D(board_height, board_width, board_embedding_size)
        self.pe_text = PositionalEncoding1D(context_length, text_embedding_size)

        self.encoders = nn.ModuleList([TransformerEncoderBlock(ff_inner_channels=ff_inner_channels, num_heads=num_heads, embed_dims=board_embedding_size) for _ in range(transformer_blocks)])
        self.decoders = nn.ModuleList([
            TransformerDecoderBlock(
                ff_inner_channels=ff_inner_channels,
                num_heads=num_heads,
                decoder_embed_dims=text_embedding_size,
                encoder_embed_dims=board_embedding_size,
                max_length=context_length
            )
            for _ in range(transformer_blocks)
        ])
        self.linear = nn.Linear(in_features=text_embedding_size, out_features=vocab_size)

        self.context_length = context_length
        self.eos_id = eos_id
        self.optimizer = optimizer
        self.lr = lr
        self.val_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=vocab_size)

        self.predictor = None
        self.to_predict = []
        self.to_predict_metadata = []

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

    def generate(self, X_board: torch.Tensor, X_text: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, do_sample:bool = False) -> torch.Tensor:
        for _ in range(max_new_tokens):
            X_text_in = X_text if X_text.size(1) < self.context_length else X_text[:, -self.context_length:]
            logits, _ = self(X_board, X_text_in, (torch.zeros(1, X_text_in.size(1)) == 1).to(X_board.device))
            logits = logits[:, -1, :] / temperature
            probs = torch.nn.functional.softmax(logits, dim=-1)
            if do_sample is not None:
                text_next = torch.multinomial(probs, num_samples=1)
            else:
                _, text_next = torch.topk(probs, k=1, dim=-1)
            X_text = torch.cat([X_text, text_next], dim=1)
            if text_next == self.eos_id:
                break
        return X_text

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> STEP_OUTPUT:
        (X_board, X_text, y_sequence, pad_mask, types) = batch
        _, loss = self(X_board, X_text, pad_mask, y_sequence)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> STEP_OUTPUT:
        (X_board, X_text, y_sequence, pad_mask, types) = batch
        logits, loss = self(X_board, X_text, pad_mask, y_sequence)

        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', self.val_acc(logits.view(-1, logits.size(-1))[pad_mask.flatten() == False], y_sequence.flatten()[pad_mask.flatten() == False]), on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> STEP_OUTPUT:
        (X_board, X_text, y_sequence, pad_mask, types) = batch
        _, loss = self(X_board, X_text, pad_mask, y_sequence)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        if self.optimizer == 'adam':
            return torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optimizer == 'sgd':
            return torch.optim.SGD(self.parameters(), lr=self.lr)
        else:
            raise ValueError(f'Unknown optimizer: {self.optimzer}')

    def set_predictors(self, sp, to_predict, to_predict_metadata):
        self.predictor = AlphaZeroPredictor(self.context_length, sp)
        self.to_predict = to_predict
        self.to_predict_metadata = to_predict_metadata

    def on_validation_end(self) -> None:
        wandb_table = wandb.Table(["past_board", "past_eval", "current_board", "current_eval", "actual_text", "predicted_text"])
        for ((X_board, y_tokens, _), (current_board, past_board, current_eval, past_eval)) in tqdm(zip(self.to_predict, self.to_predict_metadata), desc="Prediction"):
            predicted_text = self.predictor.predict(self, X_board, '', 1024)
            actual_text = self.predictor.tokens_to_string(y_tokens)
            wandb_table.add_data(
                wandb.Image(Image.open(BytesIO(
                    svg2png(chess.svg.board(None if past_board is None else chess.Board(past_board))))).convert(
                    'RGBA')),
                (0 if past_eval is None else past_eval),
                wandb.Image(
                    Image.open(BytesIO(svg2png(chess.svg.board(chess.Board(current_board))))).convert('RGBA')),
                current_eval,
                actual_text,
                predicted_text
            )
        wandb.log({
            'predictions': wandb_table
        })


class AlphazeroModelResidualEncoder(L.LightningModule):
    def __init__(
            self,
            board_in_channels: int,
            board_embedding_size: int,
            board_intermediary_channels: int,
            vocab_size: int,
            text_embedding_size: int,
            board_height: int,
            board_width: int,
            context_length: int,
            ff_inner_channels: int,
            num_heads: int,
            transformer_blocks: int,
            conv_modules_count: int,
            eos_id: int,
            optimizer: str,
            lr: float,
        ):
        super(AlphazeroModelResidualEncoder, self).__init__()
        self.save_hyperparameters()

        self.emb = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=text_embedding_size)
        self.pe_text = PositionalEncoding1D(max_len=context_length, d_model=text_embedding_size)
        self.pe_board = PositionalEncoding2D(board_height, board_width, board_embedding_size)

        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=board_in_channels, out_channels=board_embedding_size, kernel_size=1, padding='same', bias=True),
                nn.BatchNorm2d(board_embedding_size),
                nn.ReLU(inplace=True)
            ),
            *[ResidualEncoder(block_count=conv_modules_count, in_channels=board_embedding_size * 2 ** i, intermediary_channels=board_intermediary_channels * 2 ** i) for i in range(transformer_blocks - 1)]
        ])

        self.decoders = nn.ModuleList([
            TransformerDecoderBlock(
                ff_inner_channels=ff_inner_channels,
                num_heads=num_heads,
                decoder_embed_dims=text_embedding_size,
                encoder_embed_dims=board_embedding_size * 2 ** i,
                max_length=context_length)
            for i in range(transformer_blocks)
        ])

        self.linear = nn.Linear(in_features=text_embedding_size, out_features=vocab_size)

        self.context_length = context_length
        self.eos_id = eos_id
        self.optimizer = optimizer
        self.lr = lr
        self.val_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=vocab_size)

        self.predictor = None
        self.to_predict = []
        self.to_predict_metadata = []


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

    def generate(self, X_board: torch.Tensor, X_text: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, do_sample:bool = False) -> torch.Tensor:
        for _ in range(max_new_tokens):
            X_text_in = X_text if X_text.size(1) < self.context_length else X_text[:, -self.context_length:]
            logits, _ = self(X_board, X_text_in, (torch.zeros(1, X_text_in.size(1)) == 1).to(X_board.device))
            logits = logits[:, -1, :] / temperature
            probs = torch.nn.functional.softmax(logits, dim=-1)
            if do_sample is not None:
                text_next = torch.multinomial(probs, num_samples=1)
            else:
                _, text_next = torch.topk(probs, k=1, dim=-1)
            X_text = torch.cat([X_text, text_next], dim=1)
            if text_next == self.eos_id:
                break
        return X_text

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> STEP_OUTPUT:
        (X_board, X_text, y_sequence, pad_mask, types) = batch
        _, loss = self(X_board, X_text, pad_mask, y_sequence)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> STEP_OUTPUT:
        (X_board, X_text, y_sequence, pad_mask, types) = batch
        logits, loss = self(X_board, X_text, pad_mask, y_sequence)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', self.val_acc(logits.view(-1, logits.size(-1))[pad_mask.flatten() == False], y_sequence.flatten()[pad_mask.flatten() == False]), on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> STEP_OUTPUT:
        (X_board, X_text, y_sequence, pad_mask, types) = batch
        _, loss = self(X_board, X_text, pad_mask, y_sequence)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        if self.optimizer == 'adam':
            return torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optimizer == 'sgd':
            return torch.optim.SGD(self.parameters(), lr=self.lr)
        else:
            raise ValueError(f'Unknown optimizer: {self.optimzer}')


    def set_predictors(self, sp, to_predict, to_predict_metadata):
        self.predictor = AlphaZeroPredictor(self.context_length, sp)
        self.to_predict = to_predict
        self.to_predict_metadata = to_predict_metadata

    def on_validation_end(self) -> None:
        wandb_table = wandb.Table(["past_board", "past_eval", "current_board", "current_eval", "actual_text", "predicted_text"])
        for ((X_board, y_tokens, _), (current_board, past_board, current_eval, past_eval)) in tqdm(zip(self.to_predict, self.to_predict_metadata), desc="Prediction"):
            predicted_text = self.predictor.predict(self, X_board, '', 1024)
            actual_text = self.predictor.tokens_to_string(y_tokens)
            wandb_table.add_data(
                wandb.Image(Image.open(BytesIO(
                    svg2png(chess.svg.board(None if past_board is None else chess.Board(past_board))))).convert(
                    'RGBA')),
                (0 if past_eval is None else past_eval),
                wandb.Image(
                    Image.open(BytesIO(svg2png(chess.svg.board(chess.Board(current_board))))).convert('RGBA')),
                current_eval,
                actual_text,
                predicted_text
            )
        wandb.log({
            'predictions': wandb_table
        })


class AlphazeroMultipleHeadsModel(L.LightningModule):
    def __init__(
            self,
            board_in_channels: int,
            board_embedding_size: int,
            vocab_size: int,
            text_embedding_size: int,
            board_height: int,
            board_width: int,
            context_length: int,
            ff_inner_channels: int,
            num_heads: int,
            transformer_blocks: int,
            eos_id: int,
            target_types_and_depth: List[Tuple[int, int]],
            optimizer: str,
            lr: float,
    ):
        super(AlphazeroMultipleHeadsModel, self).__init__()
        self.save_hyperparameters()

        self.board_preparation = nn.Sequential(
            nn.Conv2d(in_channels=board_in_channels,
                      out_channels=board_embedding_size,
                      kernel_size=1,
                      padding=0,
                      bias=False
            ),
            nn.BatchNorm2d(board_embedding_size),
            nn.ReLU(inplace=True),
        )

        self.emb = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=text_embedding_size)
        self.pe_text = PositionalEncoding1D(context_length, text_embedding_size)
        self.pe_board = PositionalEncoding2D(board_height, board_width, board_embedding_size)

        self.encoders = nn.ModuleList([
            TransformerEncoderBlock(ff_inner_channels=ff_inner_channels,
                                    num_heads=num_heads,
                                    embed_dims=board_embedding_size)
            for _ in range(transformer_blocks)
        ])

        self.decoders = nn.ModuleList([
            TransformerDecoderBlock(
                ff_inner_channels=ff_inner_channels,
                num_heads=num_heads,
                decoder_embed_dims=text_embedding_size,
                encoder_embed_dims=board_embedding_size,
                max_length=context_length)
            for i in range(transformer_blocks)
        ])

        self.linears = nn.ModuleList([
            nn.Linear(in_features=text_embedding_size, out_features=vocab_size)
            for _ in range(len(target_types_and_depth))
        ])

        self.final_linear = nn.Linear(in_features=text_embedding_size, out_features=vocab_size)
        self.eos_id = eos_id
        self.context_length = context_length
        self.target_types_and_depth = target_types_and_depth
        self.optimizer = optimizer
        self.lr = lr
        self.val_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=vocab_size)

        self.predictor = None
        self.to_predict = []
        self.to_predict_metadata = []


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
            for (type, depth) in self.target_types_and_depth:
                idx = is_type[:, type]
                my_logits = self.linears[type](decoder_outputs[depth][idx])
                my_log_logits = -torch.nn.functional.log_softmax(my_logits, dim=-1)
                my_log_logits = my_log_logits.masked_fill(padding_mask[idx].unsqueeze(-1), 0)
                loss += torch.gather(my_log_logits, -1, targets[idx].unsqueeze(-1)).sum() / count
            log_logits = -torch.nn.functional.log_softmax(final_logits, dim=-1)
            log_logits = log_logits.masked_fill(padding_mask.unsqueeze(-1), 0)
            loss += torch.gather(log_logits, -1, targets.unsqueeze(-1)).sum() / count

        return final_logits, loss

    def generate(self, X_board: torch.Tensor, X_text: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, do_sample:bool = False) -> torch.Tensor:
        for _ in range(max_new_tokens):
            X_text_in = X_text if X_text.size(1) < self.context_length else X_text[:, -self.context_length:]
            logits, _ = self(X_board, X_text_in, (torch.zeros(1, X_text_in.size(1)) == 1).to(X_board.device))
            logits = logits[:, -1, :] / temperature
            probs = torch.nn.functional.softmax(logits, dim=-1)
            if do_sample is not None:
                text_next = torch.multinomial(probs, num_samples=1)
            else:
                _, text_next = torch.topk(probs, k=1, dim=-1)
            X_text = torch.cat([X_text, text_next], dim=1)
            if text_next == self.eos_id:
                break
        return X_text

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> STEP_OUTPUT:
        (X_board, X_text, y_sequence, pad_mask, types) = batch
        _, loss = self(X_board, X_text, pad_mask, y_sequence, types)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> STEP_OUTPUT:
        (X_board, X_text, y_sequence, pad_mask, types) = batch
        logits, loss = self(X_board, X_text, pad_mask, y_sequence, types)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', self.val_acc(logits.view(-1, logits.size(-1))[pad_mask.flatten() == False], y_sequence.flatten()[pad_mask.flatten() == False]), on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> STEP_OUTPUT:
        (X_board, X_text, y_sequence, pad_mask, types) = batch
        _, loss = self(X_board, X_text, pad_mask, y_sequence, types)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        if self.optimizer == 'adam':
            return torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optimizer == 'sgd':
            return torch.optim.SGD(self.parameters(), lr=self.lr)
        else:
            raise ValueError(f'Unknown optimizer: {self.optimzer}')


    def set_predictors(self, sp, to_predict, to_predict_metadata):
        self.predictor = AlphaZeroPredictor(self.context_length, sp)
        self.to_predict = to_predict
        self.to_predict_metadata = to_predict_metadata

    def on_validation_end(self) -> None:
        wandb_table = wandb.Table(["past_board", "past_eval", "current_board", "current_eval", "actual_text", "predicted_text"])
        for ((X_board, y_tokens, _), (current_board, past_board, current_eval, past_eval)) in tqdm(zip(self.to_predict, self.to_predict_metadata), desc="Prediction"):
            predicted_text = self.predictor.predict(self, X_board, '', 1024)
            actual_text = self.predictor.tokens_to_string(y_tokens)
            wandb_table.add_data(
                wandb.Image(Image.open(BytesIO(
                    svg2png(chess.svg.board(None if past_board is None else chess.Board(past_board))))).convert(
                    'RGBA')),
                (0 if past_eval is None else past_eval),
                wandb.Image(
                    Image.open(BytesIO(svg2png(chess.svg.board(chess.Board(current_board))))).convert('RGBA')),
                current_eval,
                actual_text,
                predicted_text
            )
        wandb.log({
            'predictions': wandb_table
        })


class ActualBoardTransformerModel(L.LightningModule):
    def __init__(
            self,
            count_past_boards: int,
            board_embedding_size: int,
            vocab_size: int,
            text_embedding_size: int,
            board_height: int,
            board_width: int,
            context_length: int,
            ff_inner_channels: int,
            num_heads: int,
            transformer_blocks: int,
            eos_id: int,
            optimizer: str,
            lr: float
    ):
        super(ActualBoardTransformerModel, self).__init__()
        self.save_hyperparameters()

        self.piece_embedding = nn.Embedding(num_embeddings=ActualBoardDataModule.get_board_token_size(), embedding_dim=board_embedding_size)
        self.pe_cell_embedding = nn.Parameter(data=torch.randn(board_width * board_height, board_embedding_size))
        self.pe_board_embedding = nn.Parameter(data=torch.randn(count_past_boards + 1, 1, board_embedding_size))
        self.strength_linear = nn.Linear(in_features=1, out_features=board_embedding_size)
        self.reps_linear = nn.Linear(in_features=1, out_features=board_embedding_size)
        self.state_ff = nn.Sequential(
            nn.Linear(in_features=7, out_features=board_embedding_size),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=board_embedding_size, out_features=board_embedding_size)
        ) #bias learns the positional embedding

        self.emb = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=text_embedding_size)
        self.pe_text = PositionalEncoding1D(context_length, text_embedding_size)

        self.encoders = nn.ModuleList([
            TransformerEncoderBlock(
                ff_inner_channels=ff_inner_channels,
                num_heads=num_heads,
                embed_dims=board_embedding_size)
            for _ in range(transformer_blocks)
        ])

        self.decoders = nn.ModuleList([
            TransformerDecoderBlock(
                ff_inner_channels=ff_inner_channels,
                num_heads=num_heads,
                decoder_embed_dims=text_embedding_size,
                encoder_embed_dims=board_embedding_size,
                max_length=context_length
            )
            for _ in range(transformer_blocks)
        ])
        self.linear = nn.Linear(in_features=text_embedding_size, out_features=vocab_size)

        self.context_length = context_length
        self.eos_id = eos_id
        self.optimizer = optimizer
        self.lr = lr
        self.val_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=vocab_size)

        self.predictor = None
        self.to_predict = []
        self.to_predict_metadata = []

    def forward(
            self,
            X_boards: torch.Tensor,                     # Batch x Boards x 64
            X_strength: torch.Tensor,                   # Batch x Boards x 1
            X_reps: torch.Tensor,                       # Batch x Boards x 1
            X_state: torch.Tensor,                      # Batch x Boards x 7
            X_text: torch.Tensor,                       # Batch x T
            padding_mask: torch.Tensor,                 # Batch x T
            targets: Optional[torch.Tensor] = None):    # Batch x T

        X_boards = self.piece_embedding(X_boards)
        X_boards = X_boards + self.pe_cell_embedding
        X_boards = X_boards + self.pe_board_embedding

        X_strength = self.strength_linear(X_strength).unsqueeze(2)
        X_reps = self.reps_linear(X_reps).unsqueeze(2)
        X_state = self.state_ff(X_state).unsqueeze(1)

        X_boards = torch.cat([X_boards, X_strength, X_reps], dim=2)
        X_boards = X_boards.view(X_boards.size(0), -1, X_boards.size(-1))
        X_boards = torch.cat([X_boards, X_state], dim=1)

        X_text = self.emb(X_text)
        X_text = self.pe_text(X_text)
        for encoder, decoder in zip(self.encoders, self.decoders):
            X_boards = encoder(X_boards)
            X_text = decoder(X_boards, X_text, padding_mask)
        logits = self.linear(X_text)
        loss = None
        if targets is not None:
            log_logits = -torch.nn.functional.log_softmax(logits, dim=-1)
            log_logits = log_logits.masked_fill(padding_mask.unsqueeze(-1), 0)
            loss = torch.gather(log_logits, -1, targets.unsqueeze(-1)).sum() / (padding_mask == False).int().sum()

        return logits, loss

    def generate(self, X_board: torch.Tensor, X_strength, X_reps, X_state, X_text: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, do_sample:bool = False) -> torch.Tensor:
        for _ in range(max_new_tokens):
            X_text_in = X_text if X_text.size(1) < self.context_length else X_text[:, -self.context_length:]
            logits, _ = self(X_board, X_strength, X_reps, X_state, X_text_in, (torch.zeros(1, X_text_in.size(1)) == 1).to(X_board.device))
            logits = logits[:, -1, :] / temperature
            probs = torch.nn.functional.softmax(logits, dim=-1)
            if do_sample is not None:
                text_next = torch.multinomial(probs, num_samples=1)
            else:
                _, text_next = torch.topk(probs, k=1, dim=-1)
            X_text = torch.cat([X_text, text_next], dim=1)
            if text_next == self.eos_id:
                break
        return X_text

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> STEP_OUTPUT:
        (X_board, X_strength, X_reps, X_state, X_text, y_sequence, pad_mask, _) = batch
        _, loss = self(X_board, X_strength, X_reps, X_state, X_text, pad_mask, y_sequence)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> STEP_OUTPUT:
        (X_board, X_strength, X_reps, X_state, X_text, y_sequence, pad_mask, _) = batch
        logits, loss = self(X_board, X_strength, X_reps, X_state, X_text, pad_mask, y_sequence)

        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', self.val_acc(logits.view(-1, logits.size(-1))[pad_mask.flatten() == False], y_sequence.flatten()[pad_mask.flatten() == False]), on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> STEP_OUTPUT:
        (X_board, X_strength, X_reps, X_state, X_text, y_sequence, pad_mask, _) = batch
        _, loss = self(X_board, X_strength, X_reps, X_state, X_text, pad_mask, y_sequence)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        if self.optimizer == 'adam':
            return torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optimizer == 'sgd':
            return torch.optim.SGD(self.parameters(), lr=self.lr)
        else:
            raise ValueError(f'Unknown optimizer: {self.optimzer}')

    def set_predictors(self, sp, to_predict, to_predict_metadata):
        self.predictor = ActualBoardPredictor(self.context_length, sp)
        self.to_predict = to_predict
        self.to_predict_metadata = to_predict_metadata

    def on_validation_end(self) -> None:
        wandb_table = wandb.Table(["past_board", "past_eval", "current_board", "current_eval", "actual_text", "predicted_text"])
        for ((X_board, X_strength, X_reps, X_state, y_tokens, _), (current_board, past_board, current_eval, past_eval)) in tqdm(zip(self.to_predict, self.to_predict_metadata), desc="Prediction"):
            predicted_text = self.predictor.predict(self, X_board, X_strength, X_reps, X_state, '', 1024)
            actual_text = self.predictor.tokens_to_string(y_tokens)
            wandb_table.add_data(
                wandb.Image(Image.open(BytesIO(
                    svg2png(chess.svg.board(None if past_board is None else chess.Board(past_board))))).convert(
                    'RGBA')),
                (0 if past_eval is None else past_eval),
                wandb.Image(
                    Image.open(BytesIO(svg2png(chess.svg.board(chess.Board(current_board))))).convert('RGBA')),
                current_eval,
                actual_text,
                predicted_text
            )
        wandb.log({
            'predictions': wandb_table
        })
