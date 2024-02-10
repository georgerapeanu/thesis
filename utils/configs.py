import enum
from typing import *


class EngineConfig(TypedDict):
    threads: int
    hash: int
    minimum_thinking_time: int
    location: str
    engine_depth: int
    mate_value: int
    processed_data_path: str
    raw_data_path: str


class DataConfig(TypedDict):
    split: str
    data_path: str
    past_boards: int
    context_length: int
    sentencepiece_path: str
    stride_big_sequences: int
    batch_size: int
    num_workers: int


class Optimizers(enum.Enum):
    ADAM = 0
    SGD = 1


class ModelConfig(TypedDict):
    text_embedding_size: int
    conv_modules_count: int
    transformer_blocks: int
    board_intermediary_channels: int
    board_in_channels: int
    board_height: int
    board_width: int
    data_config: DataConfig
    board_embedding_size: int
    ff_inner_channels: int
    num_heads: int
    vocab_size: int
    optimizer: Optimizers
    lr: int
    batches_per_epoch: int
    val_batches_per_epoch: int
    vocab_size: int
