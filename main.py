from data_preprocessor import DataPreprocessor
from feature_extractor import FeatureExtractor
from spam_detector import SpamDetector


def main():
    preprocessor = DataPreprocessor(filepath='spam_ham_dataset.csv')
    X_train, X_test, y_train, y_test = preprocessor.preprocess()

    feature_extractor = FeatureExtractor(max_features=1000)
    features_train = feature_extractor.fit_transform(X_train)
    features_test = feature_extractor.transform(X_test)

    spam_detector = SpamDetector()
    spam_detector.fit(features_train, y_train)
    predictions = spam_detector.predict(features_test)
    spam_detector.evaluate(y_test, predictions)

    while True:
        input_email = input("Enter an email to classify or 'exit' to quit:\n")
        if input_email.lower() == 'exit':
            break
        preprocessed_email = preprocessor.clean_text(input_email)
        email_features = feature_extractor.transform_single(preprocessed_email)
        prediction = spam_detector.predict(email_features)
        prediction_label = 'Spam' if prediction[0] == 1 else 'Not Spam'
        print(f"The email is classified as: {prediction_label}")


if __name__ == "__main__":
    main()
