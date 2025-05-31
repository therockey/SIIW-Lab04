import os
import pickle
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def ratio(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{x} is not a valid float")
    if x < 0.0 or x >= 1.0:
        raise argparse.ArgumentTypeError(f"{x} not in range [0, 1)")
    return x


def getargs():
    parser = argparse.ArgumentParser(description="CTG Classifier CLI")
    parser.add_argument("--data", type=str, required=True, help="Path to the input CSV file")
    parser.add_argument("--ratio", type=ratio,
                        help="Split ratio between training and testing data. Higher values = more test data, less training data",
                        default=0.2)
    subparsers = parser.add_subparsers(dest="operation", help="Operation to perform")

    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("model", type=str, choices=["NB", "DT"], help="Choice of model")
    train_parser.add_argument("output", type=str, help="Path to save the trained model")

    classify_parser = subparsers.add_parser("classify", help="Classify using a trained model")
    classify_parser.add_argument("input", type=str, help="Path to the trained model")
    classify_parser.add_argument("--result_output", type=str, help="Path to save classification results")

    analyze_parser = subparsers.add_parser("analyze", help="Perform data mining operations")

    return parser.parse_args()


def load_data(filepath):
    df = pd.read_csv(filepath)
    df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.fillna(df.mean())
    return df


def preprocess_data(df, ratio):
    X = df.drop("CLASS", axis=1)
    y = df["CLASS"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if ratio == 1:
        return [], X_scaled, [], y
    if ratio == 0:
        return X_scaled, [], y, []

    return train_test_split(X_scaled, y, test_size=ratio, random_state=42)


def check_path(path, is_input=True):
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path) and is_input:
        raise FileNotFoundError(f"The path {path} does not exist.")
    if not os.path.exists(os.path.dirname(abs_path)):
        raise FileNotFoundError(f"The directory for the path {path} does not exist.")


def check_extension(path, ext):
    if not path.endswith(ext):
        raise ValueError(f"The file {path} does not have the required extension {ext}.")


def save_model(model, output_path):
    with open(output_path, 'wb') as f:
        pickle.dump(model, f)


def load_model(input_path):
    with open(input_path, 'rb') as f:
        model = pickle.load(f)
    return model


def save_results(x, y, y_pred, output_path):
    results = pd.DataFrame({
        "Actual": y,
        "Predicted": y_pred
    })
    results.to_csv(output_path, index=False)


def main():
    try:
        args = getargs()

        print("Loading data...")
        check_path(args.data)
        check_extension(args.data, ".csv")
        data = load_data(args.data)
        print("Data loaded successfully.")

        if args.operation == "analyze":
            print("Analyzing dataset...")

            # Podstawowe statystyki
            print("\nBasic statistics:")
            print(data.describe(include='all'))

            # Liczba brakujących wartości
            print("\nMissing values per column:")
            print(data.isnull().sum())

            # Dystrybucja klas
            print("\nClass distribution (CLASS):")
            print(data['CLASS'].value_counts().sort_index())

            # Korelacje
            print("\nCorrelation matrix (first 5 rows):")
            print(data.corr().round(2).head())

            # Wariancja cech
            print("\nFeature variances:")
            print(data.drop("CLASS", axis=1).var())

            print("\nAnalysis complete.")

        if args.operation == "train":
            print("Checking model output path...")
            check_path(args.output, is_input=False)
            check_extension(args.output, ".pkl")

            print("Preprocessing data...")
            X_train, X_test, y_train, y_test = preprocess_data(data, args.ratio)

            if X_train.size == 0 and y_train.size == 0:
                raise ValueError("No training data available after preprocessing. Try changing the split ratio.")

            model = None

            if args.model == "NB":
                model = GaussianNB()
            elif args.model == "DT":
                model = DecisionTreeClassifier()

            print("Training model... This may take a while.")
            model.fit(X_train, y_train)

            print("Model training completed.")
            print("Saving model...")
            save_model(model, args.output)
            print(f"Model saved to \"{os.path.abspath(args.output)}\" successfully.")

        if args.operation == "classify":
            print("Checking model input path...")
            check_path(args.input)
            check_extension(args.input, ".pkl")

            print("Loading model...")
            model = load_model(args.input)
            print("Model loaded successfully.")

            print("Preprocessing data for classification...")
            X_train, X_test, y_train, y_test = preprocess_data(data, args.ratio)
            if X_test.size == 0 and y_test.size == 0:
                raise ValueError("No test data available after preprocessing. Try changing the split ratio.")

            print("Classifying data...")
            y_pred = model.predict(X_test)
            print("Classification completed.")

            print("Calculating metrics...")
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')

            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")
            if args.result_output:
                print("Saving classification results...")
                save_results(X_test, y_test, y_pred, args.result_output)
                print(f"Results saved to \"{os.path.abspath(args.result_output)}\" successfully.")

    except Exception as e:
        print(f"Error: {e}")
        return


if __name__ == "__main__":
    main()
