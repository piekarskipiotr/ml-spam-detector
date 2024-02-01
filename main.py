from data_preprocessor import DataPreprocessor
from feature_extractor import FeatureExtractor
from spam_detector import SpamDetector


def main():
    # Inicjalizacja i przetwarzanie danych przy użyciu klasy DataPreprocessor
    preprocessor = DataPreprocessor(filepath='spam_ham_dataset.csv')  # Wczytanie danych z pliku CSV
    X_train, X_test, y_train, y_test = preprocessor.preprocess()  # Przetwarzanie danych i podział na zbiory

    # Ekstrakcja cech z przetworzonych danych tekstowych
    feature_extractor = FeatureExtractor(max_features=1000)  # Ustawienie maksymalnej liczby cech
    features_train = feature_extractor.fit_transform(X_train)  # Transformacja zbioru treningowego
    features_test = feature_extractor.transform(X_test)  # Transformacja zbioru testowego

    # Inicjalizacja i trenowanie detektora spamu
    spam_detector = SpamDetector()
    spam_detector.fit(features_train, y_train)  # Trenowanie modelu na zbiorze treningowym
    predictions = spam_detector.predict(features_test)  # Predykcja na zbiorze testowym
    spam_detector.evaluate(y_test, predictions)  # Ocena modelu

    # Interaktywne klasyfikowanie nowych e-maili
    while True:
        input_email = input("Enter an email to classify or 'exit' to quit:\n")  # Pobranie e-maila od użytkownika
        if input_email.lower() == 'exit':  # Możliwość wyjścia z pętli
            break
        preprocessed_email = preprocessor.clean_text(input_email)  # Czyszczenie tekstu e-maila
        email_features = feature_extractor.transform_single(preprocessed_email)  # Ekstrakcja cech z e-maila
        prediction = spam_detector.predict(email_features)  # Predykcja klasy e-maila
        prediction_label = 'Spam' if prediction[0] == 1 else 'Not Spam'  # Przypisanie etykiety na podstawie predykcji
        print(f"The email is classified as: {prediction_label}")  # Wyświetlenie wyniku klasyfikacji


if __name__ == "__main__":
    main()
