import os
import argparse

from utils.load_splits import load_splits
from utils.decode_kmer import decode_kmer
from model_definitions import KmerCountsModel


def train_counts_model(data_version):
    """
    Import and train simple counts-based model to obtain probability distribution
    of mutation outcomes. Counts mutation outcome class occurrences, normalizes
    counts to correspond to probability distributions. Requires one-hot data to
    be decoded into labels for the .fit() method to work.
    """

    print(f"Version of the data: {data_version}")

    X_local_train, y_train, X_local_val, y_val = load_splits(data_version, requested_splits=["train", "val"], convert_to_tensor=False)

    X_train = decode_kmer(X_local_train)
    y_train = decode_kmer(y_train)
    X_val = decode_kmer(X_local_val)
    y_val = decode_kmer(y_val)

    print("Examples of labels:", y_train[:5])

    model = KmerCountsModel()
    model.fit(X_train, y_train)
    
    example_kmer = model.prob_table.index[0]
    pred = model.predict(example_kmer)
    print(f"Example prediction ({example_kmer}):\n{pred}")

    print("Log loss over validation set:", model.evaluate(X_val, y_val))

    model_dir = f"/faststorage/project/MutationAnalysis/Nimrod/results/models/counts_benchmark/{data_version}"
    os.makedirs(model_dir, exist_ok=True)
    model.save(f"{model_dir}/model.pkl")
    print(f"Model saved to {model_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_version",
        type=str,
        choices=[
            "3fA", "3fC", "3sA", "3sC", "5fA", "5fC", "7fA", "7fC", "9fA", "9fC",
            "11fA", "11fC", "13fA", "13fC", "15fA", "15fC", "15sA", "15sC",
            "experiment_full", "experiment_subset"
            ],
        required=True,
        help="Specify version of the data requested (kmer length, full or subset, A or C as reference nucleotide)"
    )

    args = parser.parse_args()
    train_counts_model(args.data_version)