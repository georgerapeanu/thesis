import time

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from data.CommentaryDataset import CommentaryDataset
from utils.configs import DataConfig


def get_commentary_dataloader(config: DataConfig):
    ds = CommentaryDataset(config)

    def collate_fn(data):
        board_data, sequences = torch.stack(list(map(lambda x: x[0], data))), pad_sequence(list(map(lambda x: x[1], data)), batch_first=True)
        X_sequence = sequences[:, :-1]
        y_sequence = sequences[:, 1:]
        return board_data, X_sequence, y_sequence

    dl = DataLoader(
        ds,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        collate_fn=collate_fn
    )

    return dl

if __name__ == '__main__':
    dl = get_commentary_dataloader({
        'split': 'train',
        'data_path': '../processed_data',
        'past_boards': 2,
        'context_length': 100,
        'sentencepiece_path': '../artifacts/sp8000.model',
        'stride_big_sequences': 1,
        'batch_size': 128,
        'num_workers': 8
    })
    a = time.time()
    data = next(iter(dl))
    print(data[0].shape, data[1].shape, data[2].shape)
    b = time.time()
    print(b - a)