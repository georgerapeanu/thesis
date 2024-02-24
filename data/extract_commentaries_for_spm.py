import pickle

if __name__ == '__main__':
    vectorizer, classifiers = pickle.load(open('../artifacts/svm.p', 'rb'))
    with open("../artifacts/commentaries.txt", "w") as f:
        lines = []
        with open("../artifacts/commentaries_raw.txt") as g:
            for line in g:
                lines.append(line)
        vectorized_lines = vectorizer.transform(lines)
        predictions = []
        for classifier in classifiers:
            predictions.append(classifier.predict(vectorized_lines))

        for i, line in enumerate(lines):
            if predictions[-1][i] == 1:
                continue
            take = False
            for j in range(len(classifiers)):
                if predictions[j][i]:
                    take = True
                    break
            if take:
                f.write(line)
