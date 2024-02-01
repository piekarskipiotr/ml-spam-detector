import numpy as np


class NaiveBayes:
    def __init__(self):
        self.class_log_prior_ = None
        self.feature_log_prob_ = None
        self.classes_ = None

    def fit(self, X, y):
        count_sample = X.shape[0]
        separated = [[x for x, t in zip(X, y) if t == c] for c in np.unique(y)]
        self.classes_ = np.unique(y)
        count_classes = len(self.classes_)

        self.class_log_prior_ = np.log(np.array([len(i) for i in separated]) / count_sample)
        count = np.array([np.array(i).sum(axis=0) for i in separated]) + 1.0  # Laplace smoothing
        self.feature_log_prob_ = np.log(count / count.sum(axis=1)[np.newaxis].T)
        return self

    def predict_log_proba(self, X):
        return [(self.feature_log_prob_ * x).sum(axis=1) + self.class_log_prior_ for x in X]

    def predict(self, X):
        return np.argmax(self.predict_log_proba(X), axis=1)


class SpamDetector:
    def __init__(self):
        self.model = NaiveBayes()

    def fit(self, features_train, y_train):
        self.model.fit(features_train, y_train)

    def predict(self, features):
        return self.model.predict(features)

    @staticmethod
    def evaluate(y_test, y_pred):
        accuracy = np.mean(y_pred == y_test)
        print("Accuracy:", accuracy)
