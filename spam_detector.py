from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


class SpamDetector:
    def __init__(self):
        self.model = MultinomialNB()

    def fit(self, features_train, y_train):
        self.model.fit(features_train, y_train)

    def predict(self, features):
        return self.model.predict(features)

    @staticmethod
    def evaluate(y_test, y_pred):
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))
