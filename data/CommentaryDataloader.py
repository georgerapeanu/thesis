import time

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, RandomSampler

from data.CommentaryDataset import CommentaryDataset
from utils.configs import DataConfig, SharedConfig
from typing import *


def get_commentary_dataloader(config: DataConfig, shared_config: SharedConfig) -> Tuple[DataLoader, SharedConfig]:
    ds = CommentaryDataset(config, shared_config)
    shared_config['pad_id'] = ds.get_pad_id()
    shared_config['bos_id'] = ds.get_bos_id()
    shared_config['eos_id'] = ds.get_eos_id()
    shared_config['vocab_size'] = ds.get_vocab_size()

    def collate_fn(data):
        board_data = torch.stack(list(map(lambda x: x[0], data)))
        sequences = pad_sequence(list(map(lambda x: x[1], data)), batch_first=True, padding_value=shared_config['pad_id'])
        X_sequence = sequences[:, :-1]
        y_sequence = sequences[:, 1:]
        next_pad_mask = (y_sequence == shared_config['pad_id'])
        return board_data.float(), X_sequence, y_sequence, next_pad_mask

    if config['dl_samples'] is not None:
        dl = DataLoader(
            ds,
            sampler=RandomSampler(ds, replacement=True, num_samples=config['dl_samples']),
            batch_size=config['batch_size'],
            shuffle=config['dl_shuffle'],
            num_workers=config['dl_num_workers'],
            collate_fn=collate_fn
        )
    else:
        dl = DataLoader(
            ds,
            batch_size=config['batch_size'],
            shuffle=config['dl_shuffle'],
            num_workers=config['dl_num_workers'],
            collate_fn=collate_fn
        )

    return dl, shared_config
