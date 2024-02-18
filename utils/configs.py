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


class SharedConfig(TypedDict):
    context_length: int
    sentencepiece_path: str
    eos_id: int
    bos_id: int
    pad_id: int
    vocab_size: int
    target_types: List[int]


class DataConfig(TypedDict):
    split: str
    data_path: str
    past_boards: int
    stride_big_sequences: int
    batch_size: int
    dl_num_workers: int
    in_memory: bool
    dl_shuffle: bool
    dl_samples: Optional[int]


class Optimizers(enum.Enum):
    ADAM = 0
    SGD = 1


class Models(enum.Enum):
    MODEL = 0
    MODEL_RESIDUAL_ENCODER = 1
    MODEL_MULTIPLE_HEADS = 2


class ModelConfig(TypedDict):
    name: Models
    text_embedding_size: int
    conv_modules_count: int
    transformer_blocks: int
    board_intermediary_channels: int
    board_in_channels: int
    board_height: int
    board_width: int
    board_embedding_size: int
    ff_inner_channels: int
    num_heads: int


class MultiHeadConfig(TypedDict):
    name: Models
    text_embedding_size: int
    transformer_blocks: int
    board_intermediary_channels: int
    board_in_channels: int
    board_height: int
    board_width: int
    board_embedding_size: int
    ff_inner_channels: int
    num_heads: int
    target_types_and_depth: List[Tuple[int, int]]


class TrainConfig(TypedDict):
    optimizer: Optimizers
    lr: int
    with_wandb: bool
    num_epochs: int
    predict_sentences: int


class WandbConfig(TypedDict):
    model_name: str
    text_embedding_size: int
    conv_modules_count: int
    transformer_blocks: int
    board_intermediary_channels: int
    board_embedding_size: int
    ff_inner_channels: int
    num_heads: int
    lr: int
    optimizer: str
    num_epochs: int
    context_length: int
    sp_vocab: int
    batch_size: int
    past_boards: int
    stride_big_sequences: int
    samples_per_train_epoch: int
    predict_sentences: int
    target_types: List[int]
    target_types_and_depth: Optional[List[Tuple[int, int]]]
