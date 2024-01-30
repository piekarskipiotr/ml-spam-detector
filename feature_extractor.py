from sklearn.feature_extraction.text import TfidfVectorizer


class FeatureExtractor:
    def __init__(self, max_features=1000):
        self.features_test = None
        self.features_train = None
        self.vectorizer = TfidfVectorizer(max_features=max_features)

    def fit_transform(self, X_train):
        self.features_train = self.vectorizer.fit_transform(X_train)
        return self.features_train

    def transform(self, X_test):
        self.features_test = self.vectorizer.transform(X_test)
        return self.features_test

    def transform_single(self, preprocessed_email):
        return self.vectorizer.transform([preprocessed_email])
