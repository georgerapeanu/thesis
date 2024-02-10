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
        'context_length': 128,
        'sentencepiece_path': '../artifacts/sp800.model',
        'stride_big_sequences': 5,
        'batch_size': 512,
        'dl_num_workers': 2,
        'ds_num_workers': 8,
    }

    val_data_config = {
        'split': 'valid',
        'data_path': '../processed_data',
        'past_boards': 1,
        'context_length': 128,
        'sentencepiece_path': '../artifacts/sp800.model',
        'stride_big_sequences': 5,
        'batch_size': 512,
        'dl_num_workers': 2,
        'ds_num_workers': 8,
    }

    train_dl, vocab_size, bos_id, eos_id = get_commentary_dataloader(train_data_config)
    val_dl, _, _, _ = get_commentary_dataloader(val_data_config)

    model_config = {
        'text_embedding_size': 128,
        'conv_modules_count': 2,
        'transformer_blocks': 2,
        'board_intermediary_channels': 128,
        'board_in_channels': 110,
        'board_height': 8,
        'board_width': 8,
        'data_config': train_data_config,
        'board_embedding_size': 128,
        'ff_inner_channels': 128,
        'num_heads': 8,
        'vocab_size': vocab_size,
        'optimizer': Optimizers.ADAM,
        'batches_per_epoch': 100,
        'val_batches_per_epoch': 100,
        'lr': 0.01,
        'eos_id': eos_id,
        'bos_id': bos_id
    }

    model = Model(model_config)
    print(f"Loaded model with {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters")
    optimizer = torch.optim.Adam(model.parameters(), lr=model_config['lr']) if model_config['optimizer'] == Optimizers.ADAM else torch.optim.SGD(model.parameters(), lr=model_config['lr'])

    return train_dl, val_dl, model, optimizer, model_config

def generate(batch: torch.Tensor, model, model_config: ModelConfig):
    sp = sentencepiece.SentencePieceProcessor(model_file=model_config['data_config']['sentencepiece_path'])
    (X_board, X_text, y_sequence, pad_mask) = batch
    (X_board, X_text, y_sequence, pad_mask) = (X_board[:1], X_text[:1], y_sequence[:1], pad_mask[:1])
    model_text = model.generate(X_board, torch.ones(1, 1, dtype=torch.int) * model_config['bos_id'], max_new_tokens=1024)
    print(model_text.size(), model_text.tolist())
    print(X_text.size(), X_text.tolist())
    model_string = sp.decode(model_text.view(-1).tolist())
    X_string = sp.decode(X_text.view(-1).tolist())
    print(model_string)
    print(X_string)



if __name__ == '__main__':
    train_dl, val_dl, model, optimizer, model_config = get_objects()
    model.train()
    for epoch in itertools.count(1):
        train_losses = []
        model.train()
        for (X_board, X_text, y_sequence, pad_mask) in train_dl:
            optimizer.zero_grad()
            _, loss = model(X_board, X_text, pad_mask, y_sequence)
            train_losses.append(loss.item())
            optimizer.step()
        model.eval()
        val_losses = []
        with torch.no_grad():
            for (X_board, X_text, y_sequence, pad_mask) in val_dl:
                optimizer.zero_grad()
                _, loss = model(X_board, X_text, pad_mask, y_sequence)
                val_losses.append(loss.item())
                optimizer.step()
        print(
            f"Epoch {epoch} finished, train loss: {sum(train_losses) / len(train_losses)}, val loss: {sum(val_losses) / len(val_losses)}")
