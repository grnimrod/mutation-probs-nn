import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from joblib import dump
import argparse


def train_baseline(data_version):
    """
    Load in splits, write function to direct encode response variable
    (for now, until I write custom function to calculate accuracy on vectors)
    """

    filepath = "/faststorage/project/MutationAnalysis/Nimrod/data/splits/"

    if data_version == "fA":
        filenames = ["X_train_A", "y_train_A", "X_val_A", "y_val_A", "X_test_A", "y_test_A"]
        X_train, y_train, X_val, y_val, X_test, y_test = [np.load(os.path.join(filepath, filename + ".npy")) for filename in filenames]

    elif data_version == "fC":
        filenames = ["X_train_C", "y_train_C", "X_val_C", "y_val_C", "X_test_C", "y_test_C"]
        X_train, y_train, X_val, y_val, X_test, y_test = [np.load(os.path.join(filepath, filename + ".npy")) for filename in filenames]

    elif data_version == "sA":
        filenames = ["X_train_subset_A", "y_train_subset_A", "X_val_subset_A", "y_val_subset_A", "X_test_subset_A", "y_test_subset_A"]
        X_train, y_train, X_val, y_val, X_test, y_test = [np.load(os.path.join(filepath, filename + ".npy")) for filename in filenames]

    elif data_version == "sC":
        filenames = ["X_train_subset_C", "y_train_subset_C", "X_val_subset_C", "y_val_subset_C", "X_test_subset_C", "y_test_subset_C"]
        X_train, y_train, X_val, y_val, X_test, y_test = [np.load(os.path.join(filepath, filename + ".npy")) for filename in filenames]

    else:
        print("Invalid file version specification")
        exit(1)

    # Direct encode response variable (temporal solution)
    def direct_encode(array):
        return np.argmax(array, axis=1).reshape(-1)

    y_train = direct_encode(y_train)
    y_val = direct_encode(y_val)
    y_test = direct_encode(y_test)

    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)

    class_probs = rf.predict_proba(X_val)
    print("Example probability distributions: ", class_probs[:5]) # Probability distribution of mutations of first five samples in X_val

    y_pred = rf.predict(X_val)

    accuracy = accuracy_score(y_val, y_pred)
    print("Accuracy: ", accuracy)

    cm = confusion_matrix(y_val, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    timestamp = datetime.now().strftime("%m-%d_%H-%M-%S")
    plt.savefig(f"/faststorage/project/MutationAnalysis/Nimrod/results/figures/conf_matrix_rf_{timestamp}.png")
    plt.close()
    print("Confusion matrix saved to file")

    dump(rf, "/faststorage/project/MutationAnalysis/Nimrod/results/models/baseline_rf.joblib") # TODO: model name should reflect data version


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_version",
        type=str,
        choices=["fA", "fC", "sA", "sC"],
        required=True,
        help="Specify version of the data requested (full or subset, A or C as reference nucleotide)"
    )

    args = parser.parse_args()
    train_baseline(args.data_version)