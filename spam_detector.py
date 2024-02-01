import numpy as np


class NaiveBayes:
    # Inicjalizacja klasyfikatora
    def __init__(self):
        self.class_log_prior_ = None  # Logarytm prawdopodobieństwa a priori dla każdej klasy
        self.feature_log_prob_ = None  # Logarytm prawdopodobieństwa cech przy danej klasie
        self.classes_ = None  # Lista unikalnych klas

    # Trenowanie klasyfikatora na podstawie danych treningowych X oraz etykiet y
    def fit(self, X, y):
        count_sample = X.shape[0]  # Całkowita liczba próbek
        separated = [[x for x, t in zip(X, y) if t == c] for c in np.unique(y)]  # Separacja próbek według klas
        self.classes_ = np.unique(y)  # Zapisanie unikalnych klas
        count_classes = len(self.classes_)  # Liczba unikalnych klas

        # Obliczanie logarytmu prawdopodobieństwa a priori dla każdej klasy
        self.class_log_prior_ = np.log(np.array([len(i) for i in separated]) / count_sample)
        # Obliczanie sumy cech dla każdej klasy z wygładzaniem Laplace'a
        count = np.array([np.array(i).sum(axis=0) for i in separated]) + 1.0
        # Obliczanie logarytmu prawdopodobieństwa cech przy danej klasie
        self.feature_log_prob_ = np.log(count / count.sum(axis=1)[np.newaxis].T)
        return self

    # Obliczanie logarytmu prawdopodobieństwa przynależności próbki do klas
    def predict_log_proba(self, X):
        return [(self.feature_log_prob_ * x).sum(axis=1) + self.class_log_prior_ for x in X]

    # Predykcja klas dla podanych próbek
    def predict(self, X):
        return np.argmax(self.predict_log_proba(X), axis=1)


class SpamDetector:
    # Inicjalizacja detektora spamu z modelem naiwnego Bayesa
    def __init__(self):
        self.model = NaiveBayes()

    # Trenowanie modelu na podstawie cech i etykiet treningowych
    def fit(self, features_train, y_train):
        self.model.fit(features_train, y_train)

    # Predykcja klas na podstawie cech
    def predict(self, features):
        return self.model.predict(features)

    # Statyczna metoda do oceny dokładności modelu
    @staticmethod
    def evaluate(y_test, y_pred):
        accuracy = np.mean(y_pred == y_test)  # Obliczenie dokładności jako średniej wartości prawidłowych predykcji
        print("Accuracy:", accuracy)
