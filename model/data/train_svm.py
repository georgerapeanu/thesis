import os
import pickle

import polars as pl
import numpy as np
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer

from data.create_data_type_train_file_cli import TYPES


def train_svm(artifacts_path: str):
    data = pl.read_parquet(os.path.join(artifacts_path, "commentary_types.parquet"))
    commentary = [(x['commentary'].strip()).lower() for x in data.rows(named=True)]
    cnt_samples = len(commentary)
    types = np.zeros((cnt_samples, len(TYPES)), dtype=np.int32)
    for i, x in enumerate(data.rows(named=True)):
        for type in x['type'].split(","):
            types[i, int(type)] = 1

    vectorizer = TfidfVectorizer()
    commentary = vectorizer.fit_transform(commentary)

    classifiers = [svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto') for _ in range(len(TYPES))]

    for i in range(len(classifiers)):
        classifiers[i].fit(commentary, types[:, i])

    for i in range(len(classifiers)):
        predictions = classifiers[i].predict(commentary)
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

    with open(os.path.join(artifacts_path, "svm.p"), "wb") as f:
        pickle.dump((vectorizer, classifiers), f)
