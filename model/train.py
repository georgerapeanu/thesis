import itertools

import sentencepiece
import torch
import torch.nn as nn
from model import Model
from utils.configs import DataConfig, ModelConfig, Optimizers
from data.CommentaryDataset import CommentaryDataset
from data.CommentaryDataloader import get_commentary_dataloader


def get_objects():
    train_data_config = {
        'split': 'train',
        'data_path': '../processed_data',
        'past_boards': 1,
        'context_length': 512,
        'sentencepiece_path': '../artifacts/sp8000.model',
        'stride_big_sequences': 5,
        'batch_size': 64,
        'num_workers': 4
    }

    val_data_config = {
        'split': 'valid',
        'data_path': '../processed_data',
        'past_boards': 1,
        'context_length': 512,
        'sentencepiece_path': '../artifacts/sp8000.model',
        'stride_big_sequences': 5,
        'batch_size': 64,
        'num_workers': 4
    }

    train_dl, vocab_size = get_commentary_dataloader(train_data_config)
    val_dl, _ = get_commentary_dataloader(val_data_config)

    model_config = {
        'text_embedding_size': 256,
        'conv_modules_count': 3,
        'transformer_blocks': 3,
        'board_intermediary_channels': 512,
        'board_in_channels': 494,
        'board_height': 8,
        'board_width': 8,
        'data_config': train_data_config,
        'board_embedding_size': 256,
        'ff_inner_channels': 1024,
        'num_heads': 8,
        'vocab_size': vocab_size,
        'optimizer': Optimizers.ADAM,
        'batches_per_epoch': 10,
        'val_batches_per_epoch': 10,
        'lr': 0.01
    }

    model = Model(model_config)
    print(f"Loaded model with {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters")
    optimizer = torch.optim.Adam(model.parameters(), lr=model_config['lr']) if model_config['optimizer'] == Optimizers.ADAM else torch.optim.SGD(model.parameters(), lr=model_config['lr'])

    return train_dl, val_dl, model, optimizer, model_config


if __name__ == '__main__':
    train_dl, val_dl, model, optimizer, model_config = get_objects()
    model.train()
    train_losses = []
    epoch = 0
    for i, (X_board, X_text, y_sequence, pad_mask) in zip(itertools.count(1), train_dl):
        optimizer.zero_grad()
        _, loss = model(X_board, X_text, pad_mask, y_sequence)
        train_losses.append(loss.item())
        optimizer.step()

        if i % model_config['batches_per_epoch'] == 0:
            model.eval()
            val_losses = [model(X_board, X_text, pad_mask, y_sequence)[1].item() for _, (X_board, X_text, y_sequence, pad_mask) in zip(range(model_config['batches_per_epoch']), val_dl)]

            model.train()

            epoch += 1
            print(f"Epoch {epoch} finished, train loss: {sum(train_losses) / len(train_losses)}, val loss: {sum(val_losses) / len(val_losses)}")
            train_losses = []

