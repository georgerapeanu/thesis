import itertools

import chess
import chess.svg
import sentencepiece
import torch
import torch.nn as nn
import wandb

from model.model import Model
from model.model_checkpoint import ModelCheckpoint
from model.predict import Predictor
from utils.configs import DataConfig, ModelConfig, Optimizers, TrainConfig, SharedConfig
from data.CommentaryDataset import CommentaryDataset
from data.CommentaryDataloader import get_commentary_dataloader
from typing import *
from model.metrics import get_loss

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train(
        model_config: ModelConfig,
        train_config: TrainConfig,
        shared_config: SharedConfig,
        train_dl: torch.utils.data.DataLoader,
        val_dl: torch.utils.data.DataLoader,
        test_ds: CommentaryDataset,
        predictor: Predictor,
        model: Optional[Model] = None
) -> Model:

    if model is None:
        model = Model(model_config, shared_config)

    model = model.to(device)

    optimizer = (
        torch.optim.Adam(model.parameters(), lr=train_config['lr'])) if train_config['optimizer'] == Optimizers.ADAM \
        else torch.optim.SGD(model.parameters(), lr=train_config['lr'])

    to_predict = [test_ds[i] for i in range(train_config['predict_sentences'])]
    to_predict_metadata = [test_ds.get_raw_data(i) for i in range(train_config['predict_sentences'])]

    val_checkpoint = None
    train_checkpoint = None
    if train_config['with_wandb']:
        train_checkpoint = (ModelCheckpoint('train_loss', True, 1))
        val_checkpoint = (ModelCheckpoint('val_loss', True, 1))

    for epoch in range(train_config['num_epochs']):
        model.train()
        train_losses = []
        for batch in train_dl:
            optimizer.zero_grad()
            (X_board, X_text, y_sequence, pad_mask) = batch
            _, loss = model(X_board, X_text, pad_mask, y_sequence)
            train_losses.append(loss.item())
            loss.backward()
            optimizer.step()

        train_loss = sum(train_losses) / len(train_losses)
        val_loss = get_loss(model, val_dl, device)

        print(f"Epoch {epoch+1}/{train_config['num_epochs']}: train_loss: {train_loss}, val_loss: {val_loss}")

        wandb_table = None if not train_config['with_wandb'] else wandb.Table(["past_board", "past_eval", "current_board", "current_eval", "actual_text", "predicted_text"])
        # predictions
        for ((X_board, y_tokens), (current_board, past_board, current_eval, past_eval)) in zip(to_predict, to_predict_metadata):
            predicted_text = predictor.predict(model, X_board.to(device), '', 1024, device)
            actual_text = predictor.tokens_to_string(y_tokens)
            print("=" * 100)
            print(f"Past board {None if past_board is None else str(chess.Board(past_board))}")
            print(f"Past evaluation {0 if past_eval is None else past_eval}")
            print(f"Current board {str(chess.Board(current_board))}")
            print(f"Current evaluation {current_eval}")
            print("=" * 100)
            if train_config['with_wandb']:
                wandb_table.add_data([
                    wandb.Image(chess.svg.board(None if past_board is None else chess.Board(past_board))),
                    {0 if past_eval is None else past_eval},
                    wandb.Image(chess.svg.board(chess.Board(current_board))),
                    current_eval,
                    actual_text,
                    predicted_text
                ])
        if train_config['with_wandb']:
            wandb.log({
                'train_loss': train_loss,
                'val_loss': val_loss,
                'predictions': wandb_table
            })
            train_checkpoint(model, epoch, train_loss)
            val_checkpoint(model, epoch, val_loss)

    return model
