import math
import os
import pickle
import argparse
import pandas as pd
import numpy as np
from typing import Tuple
import sklearn.base
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

pd.options.display.max_columns = 22

def ratio(x: str) -> float:
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{x} is not a valid float")
    if x < 0.0 or x >= 1.0:
        raise argparse.ArgumentTypeError(f"{x} not in range [0, 1)")
    return x


def getargs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CTG Classifier CLI")
    parser.add_argument("--data", type=str, required=True, help="Path to the input CSV file")
    parser.add_argument("--ratio", type=ratio,
                        help="Split ratio between training and testing data. Higher values = more test data, less training data",
                        default=0.2)
    parser.add_argument("--scaler", type=str, choices=["Standardization", "Normalization", "None"],
                        default="Normalization",
                        help="Scaler to use for preprocessing data")
    subparsers = parser.add_subparsers(dest="operation", help="Operation to perform")

    # Train models
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("output", type=str, help="Path to save the trained model")
    train_subparsers = train_parser.add_subparsers(dest="model", help="Choice of model to train")

    # Naive Bayes model
    nb_subparser = train_subparsers.add_parser("NB", help="Train a Naive Bayes model")
    nb_subparsers = nb_subparser.add_subparsers(dest="type", help="Type of Naive Bayes model")

    # Naive Bayes Gaussian
    nb_gaussian_subparser = nb_subparsers.add_parser("Gaussian", help="Train a Gaussian Naive Bayes model")
    nb_gaussian_subparser.add_argument("--smoothing", type=float, default=1e-9,
                                       help="Smoothing parameter for Gaussian Naive Bayes")

    # Naive Bayes Multinomial
    nb_multinomial_subparser = nb_subparsers.add_parser("Multinomial", help="Train a Multinomial Naive Bayes model")
    nb_multinomial_subparser.add_argument("--alpha", type=float, default=1.0,
                                          help="Smoothing parameter for Multinomial Naive Bayes")

    # Naive Bayes Bernoulli
    nb_bernoulli_subparser = nb_subparsers.add_parser("Bernoulli", help="Train a Bernoulli Naive Bayes model")
    nb_bernoulli_subparser.add_argument("--alpha", type=float, default=1.0,
                                        help="Smoothing parameter for Bernoulli Naive Bayes")
    nb_bernoulli_subparser.add_argument("--binarize", type=float, default=0.0,
                                        help="Binarization threshold for Bernoulli Naive Bayes")

    # Decision Tree model
    dt_subparser = train_subparsers.add_parser("DT", help="Train a Decision Tree model")
    dt_subparser.add_argument("--max_depth", type=int, default=None, help="Maximum depth of the tree")
    dt_subparser.add_argument("--min_samples_split", type=int, default=2,
                              help="Minimum number of samples required to split an internal node")
    dt_subparser.add_argument("--min_samples_leaf", type=int, default=1,
                              help="Minimum number of samples required to be at a leaf node")
    dt_subparser.add_argument("--max_leaf_nodes", type=int, default=None,
                              help="Grow a tree with max_leaf_nodes in best-first fashion")
    dt_subparser.add_argument("--criterion", type=str, choices=["gini", "entropy"], default="gini",
                              help="Criterion to measure the quality of a split")
    dt_subparser.add_argument("--splitter", type=str, choices=["best", "random"], default="best",
                              help="Strategy used to choose the split at each node")
    dt_subparser.add_argument("--max_features", type=str, choices=["auto", "sqrt", "log2", None], default=None,
                              help="Number of features to consider when looking for the best split")

    # Classify using a trained model
    classify_parser = subparsers.add_parser("classify", help="Classify using a trained model")
    classify_parser.add_argument("input", type=str, help="Path to the trained model")
    classify_parser.add_argument("--result_output", type=str, help="Path to save classification results")

    # Analyze dataset
    analyze_parser = subparsers.add_parser("analyze", help="Perform data mining operations")

    return parser.parse_args()


def get_model(args: argparse.Namespace) -> sklearn.base.ClassifierMixin:
    if args.model == "NB":
        if args.type == "Gaussian":
            return GaussianNB(var_smoothing=args.smoothing)
        if args.type == "Multinomial":
            return MultinomialNB(alpha=args.alpha)
        if args.type == "Bernoulli":
            return BernoulliNB(alpha=args.alpha, binarize=args.binarize)
    if args.model == "DT":
        return DecisionTreeClassifier(
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            min_samples_leaf=args.min_samples_leaf,
            max_leaf_nodes=args.max_leaf_nodes,
            criterion=args.criterion,
            splitter=args.splitter,
            max_features=args.max_features
        )

    raise ValueError(f"Unknown model type: {args.model} with subtype {args.type if hasattr(args, 'type') else ''}")


def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    return df


def preprocess_data(df: pd.DataFrame, ratio: float, scaler_choice: str) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.fillna(df.mean())
    X = df.drop("CLASS", axis=1)
    y = df["CLASS"]
    scaler = None

    if scaler_choice == "Normalization":
        scaler = Normalizer()
    elif scaler_choice == "Standardization":
        scaler = StandardScaler()

    if scaler is not None:
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X.values

    if ratio == 1:
        return np.array([]), X_scaled, np.array([]), y
    if ratio == 0:
        return X_scaled, np.array([]), y, np.array([])

    return train_test_split(X_scaled, y, test_size=ratio, random_state=42)


def check_path(path: str, is_input=True) -> None:
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path) and is_input:
        raise FileNotFoundError(f"The path {path} does not exist.")
    if not os.path.exists(os.path.dirname(abs_path)):
        raise FileNotFoundError(f"The directory for the path {path} does not exist.")


def check_extension(path: str, ext: str) -> None:
    if not path.endswith(ext):
        raise ValueError(f"The file {path} does not have the required extension {ext}.")


def save_model(model: sklearn.base.ClassifierMixin, output_path: str) -> None:
    with open(output_path, 'wb') as f:
        pickle.dump(model, f)


def load_model(input_path: str) -> sklearn.base.ClassifierMixin:
    with open(input_path, 'rb') as f:
        model = pickle.load(f)
    return model


def save_results(x, y, y_pred, output_path) -> None:
    results = pd.DataFrame({
        "Actual": y,
        "Predicted": y_pred
    })
    results.to_csv(output_path, index=False)


def main() -> None:
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
            print(data.corr().round(2))

            # Wariancja cech
            print("\nFeature variances:")
            print(data.drop("CLASS", axis=1).var())

            print("\nAnalysis complete.")

        if args.operation == "train":
            print("Checking model output path...")
            check_path(args.output, is_input=False)
            check_extension(args.output, ".pkl")

            print("Preprocessing data...")
            X_train, X_test, y_train, y_test = preprocess_data(data, args.ratio, args.scaler)

            if args.model == "NB" and args.type == "Multinomial":
                min_val = min(X_train.min(), X_test.min())
                if min_val < 0:
                    X_train -= min_val
                    X_test -= min_val

            if X_train.size == 0 and y_train.size == 0:
                raise ValueError("No training data available after preprocessing. Try changing the split ratio.")

            model = get_model(args)

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
            X_train, X_test, y_train, y_test = preprocess_data(data, args.ratio, args.scaler)
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
