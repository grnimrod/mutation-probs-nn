import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from joblib import dump
import argparse

from utils.load_splits import load_splits


def train_baseline(data_version):
    """
    Load in splits, write function to direct encode response variable
    (for now, until I write custom function to calculate accuracy on vectors)
    """

    X_train, y_train, X_val, y_val, X_test, y_test = load_splits(data_version)

    # Direct encode response variable (temporal solution)
    def direct_encode(array):
        return np.argmax(array, axis=1).reshape(-1)

    y_train = direct_encode(y_train)
    y_val = direct_encode(y_val)
    y_test = direct_encode(y_test)

    print("Label examples:", y_val[:5])
    print("Label shape", y_val.shape)

    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)

    class_probs = rf.predict_proba(X_val)
    print("Example probability distributions: ", class_probs[:5]) # Probability distribution of mutations of first five samples in X_val

    y_pred = rf.predict(X_val) # Predicts label, not distribution

    accuracy = accuracy_score(y_val, y_pred)
    print("Accuracy: ", accuracy)

    # Log likelihood of the validation set
    loss = log_loss(y_val, class_probs)
    print(f"Log loss: {loss}")

    plots_dir = "/faststorage/project/MutationAnalysis/Nimrod/results/figures/baseline"
    os.makedirs(plots_dir, exist_ok=True)

    cm = confusion_matrix(y_val, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    timestamp = datetime.now().strftime("%m-%d_%H-%M-%S")
    plt.savefig(f"{plots_dir}/conf_matrix_rf_{timestamp}.png")
    plt.close()
    print("Confusion matrix saved to file")

    model_dir = f"/faststorage/project/MutationAnalysis/Nimrod/results/models/baseline/{data_version}"
    os.makedirs(model_dir, exist_ok=True)
    dump(rf, f"{model_dir}/baseline_rf.joblib")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_version",
        type=str,
        choices=["3fA", "3fC", "3sA", "3sC", "15fA", "15fC", "15sA", "15sC", "experiment"],
        required=True,
        help="Specify version of the data requested (full or subset, A or C as reference nucleotide)"
    )

    args = parser.parse_args()
    train_baseline(args.data_version)