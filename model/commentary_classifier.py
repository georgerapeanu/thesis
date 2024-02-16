import polars as pl
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, svm
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    data = pl.read_parquet("../artifacts/commentary_types.parquet")
    commentary = [(x['commentary'].strip()).lower() for x in data.rows(named=True)]
    types = [x['type'] for x in data.rows(named=True)]

    encoder = LabelEncoder()
    vectorizer = TfidfVectorizer()

    commentary = vectorizer.fit_transform(commentary)
    types = encoder.fit_transform(types)

    SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    SVM.fit(commentary, types)

    predictions_SVM = SVM.predict(commentary)

    data_stats = defaultdict(lambda: [0, 0])

    for x, y in zip(predictions_SVM, types):
        data_stats[y][1] += 1
        if x == y:
            data_stats[x][0] += 1

    print("Accuracy: ", accuracy_score(predictions_SVM, types) * 100)

    for i in sorted(data_stats.keys()):
        print(f"Accuracy for class {i}: {data_stats[i][0] / data_stats[i][1] * 100}")


    with open('../artifacts/commentaries.txt') as f:
        commentaries = [line for line in f]
        commentaries_vectorized = vectorizer.transform(commentaries)
        predctions = SVM.predict(commentaries_vectorized)

        print("\n".join(list(map(lambda x: x[0], filter(lambda x: x[1] == 5, zip(commentaries, predctions))))[:100]))