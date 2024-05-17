import multiprocessing
import os
import pickle

import hydra
import numpy as np
import polars as pl
import stockfish
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf


class Worker:
    def __init__(self, config: DictConfig):
        self.__config = config
        self.__engine = stockfish.Stockfish(self.__config['location'], parameters={
            "Threads": self.__config['threads'],
            "Hash": self.__config['hash'],
            "Minimum Thinking Time": self.__config['minimum_thinking_time']
        })
        self.__engine.set_depth(self.__config['engine_depth'])
        (self.vectorizer, self.classifiers) = pickle.load(open(os.path.join(self.__config['artifacts_path'], "svm.p"), "rb"))

    def __evaluation_to_value(self, evaluation):
        if evaluation['type'] == 'cp':
            return evaluation['value']
        else:
            return self.__config['mate_value'] if evaluation['value'] > 0 else -self.__config['mate_value']

    def __call__(self, file: str):
        try:
            raw_file_path = os.path.join(self.__config['raw_data_path'], file)
            df = pl.read_parquet(raw_file_path)
            answer = []
            for past_board, current_board, commentary in df.rows():
                self.__engine.set_fen_position(past_board)
                past_strength = self.__evaluation_to_value(self.__engine.get_evaluation())
                self.__engine.set_fen_position(current_board)
                current_strength = self.__evaluation_to_value(self.__engine.get_evaluation())

                vectorized_commentary = self.vectorizer.transform([commentary])

                if len(commentary.strip()) == 0:
                    continue

                answer.append({
                    k: v for k, v in [
                        ('past_board', past_board),
                        ('current_board', current_board),
                        ('commentary', commentary),
                        ('past_strength', past_strength),
                        ('current_strength', current_strength),
                        *[(f"is_type_{i}", self.classifiers[i].predict(vectorized_commentary)[0]) for i in range(len(self.classifiers))]
                    ]
                })
            if len(answer) > 0:
                pl.DataFrame(answer).write_parquet(os.path.join(self.__config['processed_path'], file))
            print(f"Done {file}")
        except pl.exceptions.ComputeError as e:
            pass


worker = None
def init_worker(config):
    global worker
    worker = Worker(config)


def process(file):
    global worker
    worker(file)


def process_raw_data(engine_config: DictConfig):
    # config: EngineConfig = {
    #     'location': os.path.join(artifacts_path, 'stockfish-ubuntu-x86-64-avx2'),
    #     'threads': 4,
    #     'hash': 128,
    #     'minimum_thinking_time': 1,
    #     'raw_data_path': raw_data_path,
    #     'processed_data_path': processed_data_path,
    #     'engine_depth': 5,
    #     'mate_value': 10000
    # }

    files = []

    for split in ['train', 'test', 'valid']:
        for file in os.listdir(os.path.join(engine_config['raw_data_path'], split)):
            files.append(os.path.join(split, file))

    # init_worker(engine_config)
    # for file in files:
    #     process(file)

    with multiprocessing.get_context("spawn").Pool(8, initializer=init_worker, initargs=[engine_config]) as p:
        p.map(process, files)

if __name__ == '__main__':
    config = {
        'artifacts_path': "../artifacts",
        'location': os.path.join("../artifacts", 'stockfish-ubuntu-x86-64-avx2'),
        'threads': 4,
        'hash': 128,
        'minimum_thinking_time': 1,
        'raw_data_path': "../raw_data",
        'processed_path': "../processed_data",
        'engine_depth': 5,
        'mate_value': 10000
    }

    process_raw_data(config)