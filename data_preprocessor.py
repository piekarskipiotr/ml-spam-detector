import csv
import numpy as np
import string
import nltk

# Pobranie listy słów nieistotnych z biblioteki NLTK
nltk.download('stopwords')
from nltk.corpus import stopwords


class DataPreprocessor:
    # Inicjalizacja klasy z podaną ścieżką do pliku oraz wczytaniem słów nieistotnych
    def __init__(self, filepath):
        self.filepath = filepath  # Ścieżka do pliku z danymi
        self.stopwords = set(stopwords.words('english'))  # Lista słów nieistotnych w języku angielskim
        self.X_train = None  # Zbiór danych treningowych (tekst)
        self.X_test = None  # Zbiór danych testowych (tekst)
        self.y_train = None  # Etykiety dla zbioru treningowego
        self.y_test = None  # Etykiety dla zbioru testowego

    # Wczytywanie danych z pliku CSV
    def load_data(self):
        with open(self.filepath, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)  # Użycie czytnika słownikowego do łatwego dostępu do kolumn
            data = [row for row in reader]  # Przeczytanie wszystkich wierszy
        return data

    # Czyszczenie tekstu z znaków interpunkcyjnych i słów nieistotnych
    def clean_text(self, text):
        text = "".join(char for char in text if char not in string.punctuation)  # Usunięcie znaków interpunkcyjnych
        text = text.lower()  # Zamiana tekstu na małe litery
        text = " ".join(word for word in text.split() if word not in self.stopwords)  # Usunięcie słów nieistotnych
        return text

    # Przetwarzanie danych: wczytanie, oczyszczenie i podział na zbiory treningowe i testowe
    def preprocess(self):
        data = self.load_data()  # Wczytanie danych
        processed_texts = [self.clean_text(row['text']) for row in data]  # Oczyszczenie tekstu w każdym wierszu
        labels = [row['label_num'] for row in data]  # Wczytanie etykiet

        texts_np = np.array(processed_texts)  # Konwersja tekstów na tablicę NumPy
        labels_np = np.array(labels, dtype=int)  # Konwersja etykiet na tablicę NumPy

        # Przemieszanie danych
        indices = np.arange(texts_np.shape[0])
        np.random.shuffle(indices)
        texts_np = texts_np[indices]
        labels_np = labels_np[indices]

        # Podział na zbiory treningowe i testowe
        test_size = 0.2  # Rozmiar zbioru testowego to 20% całkowitych danych
        split_index = int(len(texts_np) * (1 - test_size))  # Obliczenie indeksu podziału
        self.X_train, self.X_test = texts_np[:split_index], texts_np[split_index:]  # Podział tekstów
        self.y_train, self.y_test = labels_np[:split_index], labels_np[split_index:]  # Podział etykiet

        return self.X_train, self.X_test, self.y_train, self.y_test
