import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
import string
import nltk

nltk.download('stopwords')


class DataPreprocessor:
    def __init__(self, filepath):
        self.y_test = None
        self.y_train = None
        self.X_test = None
        self.X_train = None
        self.data = None
        self.filepath = filepath

    def load_data(self):
        self.data = pd.read_csv(self.filepath)

    @staticmethod
    def clean_text(text):
        text = "".join([char for char in text if char not in string.punctuation])
        text = text.lower()
        text = " ".join([word for word in text.split() if word not in stopwords.words("english")])
        return text

    def preprocess(self):
        self.load_data()
        self.data['processed_text'] = self.data['text'].apply(self.clean_text)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.data['processed_text'], self.data['label_num'], test_size=0.2, random_state=42
        )
        return self.X_train, self.X_test, self.y_train, self.y_test
