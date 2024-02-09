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
