import numpy as np


class NaiveBayes:
    def __init__(self):
        # Inicjalizacja zmiennych do przechowywania logarytmów prawdopodobieństwa,
        # logarytmów prawdopodobieństwa cech oraz listy klas.
        self.class_log_prior_ = None
        self.feature_log_prob_ = None
        self.classes_ = None

    def fit(self, X, y):
        # Metoda do trenowania modelu na podstawie danych X (cechy) i y (etykiety).
        count_sample = X.shape[0]
        # Oddzielenie próbek dla każdej klasy.
        separated = [[x for x, t in zip(X, y) if t == c] for c in np.unique(y)]
        self.classes_ = np.unique(y)

        # Obliczenie logarytmu prawdopodobieństwa a priori dla każdej klasy.
        self.class_log_prior_ = np.log(np.array([len(i) for i in separated]) / count_sample)
        # Obliczenie sumy cech dla każdej klasy i dodanie 1 dla uniknięcia dzielenia przez zero (wygładzanie Laplace'a).
        count = np.array([np.array(i).sum(axis=0) for i in separated]) + 1.0
        # Obliczenie logarytmów prawdopodobieństwa cech.
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
        # Trenowanie modelu Naive Bayes na podstawie danych treningowych.
        self.model.fit(features_train, y_train)

    def predict(self, features):
        # Predykcja, czy dane wejściowe są spamem czy nie.
        return self.model.predict(features)

    @staticmethod
    def evaluate(y_test, y_pred):
        # Obliczenie i wyświetlenie dokładności modelu na podstawie danych testowych i przewidywań.
        accuracy = np.mean(y_pred == y_test)
        print("Accuracy:", accuracy)
