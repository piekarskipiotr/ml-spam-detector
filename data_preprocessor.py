import csv
import numpy as np
import string
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords


class DataPreprocessor:
    def __init__(self, filepath):
        self.filepath = filepath
        self.stopwords = set(stopwords.words('english'))
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self):
        with open(self.filepath, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            data = [row for row in reader]
        return data

    def clean_text(self, text):
        text = "".join(char for char in text if char not in string.punctuation)
        text = text.lower()
        text = " ".join(word for word in text.split() if word not in self.stopwords)
        return text

    def preprocess(self):
        data = self.load_data()
        processed_texts = [self.clean_text(row['text']) for row in data]
        labels = [row['label_num'] for row in data]

        texts_np = np.array(processed_texts)
        labels_np = np.array(labels, dtype=int)

        indices = np.arange(texts_np.shape[0])
        np.random.shuffle(indices)
        texts_np = texts_np[indices]
        labels_np = labels_np[indices]

        test_size = 0.2
        split_index = int(len(texts_np) * (1 - test_size))
        self.X_train, self.X_test = texts_np[:split_index], texts_np[split_index:]
        self.y_train, self.y_test = labels_np[:split_index], labels_np[split_index:]

        return self.X_train, self.X_test, self.y_train, self.y_test
