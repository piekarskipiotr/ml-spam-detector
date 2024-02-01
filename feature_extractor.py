import numpy as np


class FeatureExtractor:
    # Inicjalizacja klasy z opcjonalnym argumentem określającym maksymalną liczbę cech
    def __init__(self, max_features=1000):
        self.max_features = max_features  # Maksymalna liczba cech
        self.features_test = None  # Cechy dla zbioru testowego
        self.features_train = None  # Cechy dla zbioru treningowego
        self.vocabulary_ = {}  # Słownik z mapowaniem słów na indeksy
        self.idf_ = None  # Wartości IDF dla słów

    # Budowanie słownika na podstawie zbioru treningowego
    def _fit_vocabulary(self, X_train):
        vocab = {}  # Tymczasowy słownik do zliczania wystąpień słów
        for document in X_train:
            for word in document.split():
                if word in vocab:
                    vocab[word] += 1
                else:
                    vocab[word] = 1

        # Wybór `max_features` najczęściej występujących słów
        sorted_vocab = sorted(vocab.items(), key=lambda item: item[1], reverse=True)
        self.vocabulary_ = {term: index for index, (term, _) in enumerate(sorted_vocab[:self.max_features])}

    # Obliczanie IDF dla każdego słowa w słowniku
    def _calculate_idf(self, X_train):
        idf = np.zeros(len(self.vocabulary_))  # Zainicjowanie tablicy IDF wartościami zerowymi
        N = len(X_train)  # Całkowita liczba dokumentów

        for doc in X_train:
            seen_terms = set()  # Zbiór unikalnych słów w dokumencie
            for term in doc.split():
                if term in self.vocabulary_ and term not in seen_terms:
                    idf[self.vocabulary_[term]] += 1
                    seen_terms.add(term)

        # Obliczenie IDF zgodnie ze wzorem
        self.idf_ = np.log(N / (idf + 1)) + 1

    # Transformacja zbioru dokumentów na reprezentację TF-IDF
    def _transform(self, X, fit=False):
        if fit:
            self._fit_vocabulary(X)  # Budowanie słownika i obliczanie IDF podczas fazy treningowej
            self._calculate_idf(X)

        rows, cols, values = [], [], []
        for idx, doc in enumerate(X):
            term_count = {}  # Słownik zliczający wystąpienia słów w dokumencie
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
                    # Obliczenie wartości TF-IDF dla słowa
                    tfidf = (count / len(doc.split())) * self.idf_[term_idx]
                    values.append(tfidf)

        # Zwrócenie trzech tablic: indeksy wierszy, kolumn i wartości TF-IDF
        return np.array(rows), np.array(cols), np.array(values)

    # Transformacja zbioru treningowego i zapisanie wyników w `features_train`
    def fit_transform(self, X_train):
        rows, cols, values = self._transform(X_train, fit=True)
        self.features_train = np.zeros((len(X_train), len(self.vocabulary_)))  # Inicjalizacja macierzy cech
        self.features_train[rows, cols] = values  # Przypisanie wartości TF-IDF
        return self.features_train

    # Transformacja zbioru testowego bez dodatkowego dopasowywania słownika
    def transform(self, X_test):
        rows, cols, values = self._transform(X_test)
        self.features_test = np.zeros(
            (len(X_test), len(self.vocabulary_)))

    def transform_single(self, preprocessed_email):
        # Inicjalizacja pustych list do przechowywania indeksów wierszy, kolumn i wartości reprezentacji TF-IDF
        rows, cols, values = [], [], []

        # Inicjalizacja pustego słownika do zliczania częstotliwości występowania terminów w pojedynczym e-mailu
        term_count = {}

        # Podział przetworzonego tekstu e-maila na terminy
        for term in preprocessed_email.split():
            # Sprawdzenie, czy termin znajduje się w słowniku
            if term in self.vocabulary_:
                # Inkrementacja licznika terminów lub dodanie go do słownika, jeśli jeszcze go nie ma
                term_count[term] = term_count.get(term, 0) + 1

        # Iteracja przez terminy i ich liczniki w e-mailu
        for term, count in term_count.items():
            if term in self.vocabulary_:
                # Pobranie indeksu terminu ze słownika
                term_idx = self.vocabulary_[term]
                # Dodanie indeksu do listy wierszy (0, ponieważ jest to tylko jeden e-mail)
                rows.append(0)
                # Dodanie indeksu terminu do listy kolumn
                cols.append(term_idx)
                # Obliczenie wartości TF-IDF i dodanie jej do listy wartości
                tfidf = (count / len(preprocessed_email.split())) * self.idf_[term_idx]
                values.append(tfidf)

        # Utworzenie macierzy rzadkiej dla wartości TF-IDF pojedynczego e-maila
        feature_vector = np.zeros((1, len(self.vocabulary_)))
        feature_vector[rows, cols] = values

        # Zwrócenie wektora cech TF-IDF dla pojedynczego e-maila
        return feature_vector
