import numpy as np


class FeatureExtractor:
    def __init__(self, max_features=1000):
        self.max_features = max_features
        self.features_test = None
        self.features_train = None
        self.vocabulary_ = {}
        self.idf_ = None

    def _fit_vocabulary(self, X_train):
        vocab = {}
        for document in X_train:
            for word in document.split():
                if word in vocab:
                    vocab[word] += 1
                else:
                    vocab[word] = 1

        sorted_vocab = sorted(vocab.items(), key=lambda item: item[1], reverse=True)
        self.vocabulary_ = {term: index for index, (term, _) in enumerate(sorted_vocab[:self.max_features])}

    def _calculate_idf(self, X_train):
        idf = np.zeros(len(self.vocabulary_))
        N = len(X_train)

        for doc in X_train:
            seen_terms = set()
            for term in doc.split():
                if term in self.vocabulary_ and term not in seen_terms:
                    idf[self.vocabulary_[term]] += 1
                    seen_terms.add(term)

        self.idf_ = np.log(N / (idf + 1)) + 1

    def _transform(self, X, fit=False):
        if fit:
            self._fit_vocabulary(X)
            self._calculate_idf(X)

        rows, cols, values = [], [], []
        for idx, doc in enumerate(X):
            term_count = {}
            for term in doc.split():
                if term in self.vocabulary_:
                    if term in term_count:
                        term_count[term] += 1
                    else:
                        term_count[term] = 1

            for term, count in term_count.items():
                if term in self.vocabulary_:
                    term_idx = self.vocabulary_[term]
                    rows.append(idx)
                    cols.append(term_idx)
                    # Calculate TF-IDF
                    tfidf = (count / len(doc.split())) * self.idf_[term_idx]
                    values.append(tfidf)

        return np.array(rows), np.array(cols), np.array(values)

    def fit_transform(self, X_train):
        rows, cols, values = self._transform(X_train, fit=True)
        self.features_train = np.zeros((len(X_train), len(self.vocabulary_)))
        self.features_train[rows, cols] = values
        return self.features_train

    def transform(self, X_test):
        rows, cols, values = self._transform(X_test)
        self.features_test = np.zeros((len(X_test), len(self.vocabulary_)))
        self.features_test[rows, cols] = values
        return self.features_test

    def transform_single(self, preprocessed_email):
        return self.transform([preprocessed_email])
