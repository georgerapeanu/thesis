import multiprocessing
import os

import numpy as np
import polars as pl
import stockfish
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer

from data.create_data_type_train_file_cli import TYPES
from utils.configs import EngineConfig

class Worker:
    def __init__(self, config: EngineConfig):
        self.__config = config
        self.__engine = stockfish.Stockfish(self.__config['location'], parameters={
            "Threads": self.__config['threads'],
            "Hash": self.__config['hash'],
            "Minimum Thinking Time": self.__config['minimum_thinking_time']
        })
        self.__engine.set_depth(self.__config['engine_depth'])

        data = pl.read_parquet("../artifacts/commentary_types.parquet")
        commentary = [(x['commentary'].strip()).lower() for x in data.rows(named=True)]
        cnt_samples = len(commentary)
        types = np.zeros((cnt_samples, len(TYPES)), dtype=np.int32)
        for i, x in enumerate(data.rows(named=True)):
            for type in x['type'].split(","):
                types[i, int(type)] = 1

        self.vectorizer = TfidfVectorizer()
        commentary = self.vectorizer.fit_transform(commentary)

        self.classifiers = [svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto') for _ in range(len(TYPES))]

        for i in range(len(self.classifiers)):
            self.classifiers[i].fit(commentary, types[:, i])

        for i in range(len(self.classifiers)):
            predictions = self.classifiers[i].predict(commentary)
            data_matrix = np.zeros((2, 2))

            for x, y in zip(types[:, i], predictions):
                data_matrix[x, y] += 1

            print(f"{TYPES[i]}")
            print(f"Training percentage: {types[:, i].sum() / cnt_samples}")
            print(f"Accuracy: {(data_matrix[0][0] + data_matrix[1][1]) / (data_matrix.sum())}")
            pr = (data_matrix[1][1]) / (data_matrix[1][1] + data_matrix[0][1])
            rc = (data_matrix[1][1]) / (data_matrix[1][1] + data_matrix[1][0])
            print(f"Precision: {pr}")
            print(f"Recall: {rc}")
            print(f"F1: {2 * pr * rc / (pr + rc)}")
            print("")

        counts = np.zeros(len(TYPES))

    def __evaluation_to_value(self, evaluation):
        if evaluation['type'] == 'cp':
            return evaluation['value']
        else:
            return self.__config['mate_value'] if evaluation['value'] > 0 else -self.__config['mate_value']

    def __call__(self, file: str):
        try:
            df = pl.read_parquet(os.path.join(self.__config['raw_data_path'], file))
            answer = []
            for past_board, current_board, commentary in df.rows():
                self.__engine.set_fen_position(past_board)
                past_strength = self.__evaluation_to_value(self.__engine.get_evaluation())
                self.__engine.set_fen_position(current_board)
                current_strength = self.__evaluation_to_value(self.__engine.get_evaluation())

                vectorized_commentary = self.vectorizer.transform([commentary])
                answer.append({
                    k: v for k, v in [
                        ('past_board', past_board),
                        ('current_board', current_board),
                        ('commentary', commentary),
                        ('past_strength', past_strength),
                        ('current_strength', current_strength),
                        *[(f"is_type_{i}", self.classifiers[i].predict(vectorized_commentary)) for i in range(len(self.classifiers))]
                    ]
                })
            pl.DataFrame(answer).write_parquet(os.path.join(self.__config['processed_data_path'], file))
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

if __name__ == '__main__':
    config: EngineConfig = {
        'location': '../artifacts/stockfish-ubuntu-x86-64-avx2',
        'threads': 4,
        'hash': 128,
        'minimum_thinking_time': 1,
        'raw_data_path': '../raw_data',
        'processed_data_path': '../processed_data',
        'engine_depth': 5,
        'mate_value': 10000
    }

    files = []

    for split in ['train', 'test', 'valid']:
        for file in os.listdir(os.path.join(config['raw_data_path'], split)):
            files.append(os.path.join(split, file))

    with multiprocessing.Pool(8, initializer=init_worker, initargs=[config]) as p:
        p.map(process, files)
