import time

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from data.CommentaryDataset import CommentaryDataset
from utils.configs import DataConfig
from typing import *


def get_commentary_dataloader(config: DataConfig) -> Tuple[DataLoader, int, int, int]:
    ds = CommentaryDataset(config)

    def collate_fn(data):
        board_data = torch.stack(list(map(lambda x: x[0], data)))
        sequences = pad_sequence(list(map(lambda x: x[1], data)), batch_first=True, padding_value=ds.pad_id())
        next_pad_mask = pad_sequence([torch.ones(len(sequence)) for _, sequence in data], batch_first=True, padding_value=0)
        next_pad_mask = next_pad_mask[:, 1:]
        next_pad_mask = (next_pad_mask == 0)
        X_sequence = sequences[:, :-1]
        y_sequence = sequences[:, 1:]
        return board_data.float(), X_sequence, y_sequence, next_pad_mask

    dl = DataLoader(
        ds,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        collate_fn=collate_fn
    )

    return dl, ds.vocab_size(), ds.bos_id(), ds.eos_id(),


if __name__ == '__main__':
    dl = get_commentary_dataloader({
        'split': 'train',
        'data_path': '../processed_data',
        'past_boards': 2,
        'context_length': 1024,
        'sentencepiece_path': '../artifacts/sp8000.model',
        'stride_big_sequences': 1,
        'batch_size': 128,
        'num_workers': 8
    })
    a = time.time()
    print(next(iter(dl)))
    for batch in dl:
        print(batch[0].shape, batch[1].shape, batch[2].shape, batch[3].shape)
    b = time.time()
    print(b - a)