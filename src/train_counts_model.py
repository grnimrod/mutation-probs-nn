import pandas as pd
import numpy as np
from sklearn.metrics import log_loss
import argparse

from utils.load_splits import load_splits


def counts_model(data_version):
    print(f"Version of the data: {data_version}")

    X_train, y_train, X_val, y_val, X_test, y_test = load_splits(data_version, convert_to_tensor=False)

    def decode_kmer(kmer):
        """Transform one-hot encoded version back to labels"""
        
        kmer_length = kmer.shape[1] // 4
        reshaped = kmer.reshape(-1, kmer_length, 4)
        indices = np.argmax(reshaped, axis=2)
        nucleotides = np.array(["A", "C", "G", "T"])
        decoded = nucleotides[indices]
        return np.array([''.join(nucs) for nucs in decoded])
    

    X_train = decode_kmer(X_train)
    X_val = decode_kmer(X_val)
    X_test = decode_kmer(X_test)
    y_train = decode_kmer(y_train)
    y_val = decode_kmer(y_val)
    y_test = decode_kmer(y_test)

    print(y_train[:5])

    train_df = pd.DataFrame({
        "context": X_train,
        "mut_type": y_train
    })
    val_df = pd.DataFrame({
        "context": X_val,
        "mut_type": y_val
    })

    print(train_df.head())

    mutation_counts = train_df.groupby(["context", "mut_type"]).size().unstack(fill_value=0)

    mutation_counts = mutation_counts.div(mutation_counts.sum(axis=1), axis=0)

    print(f"Mutation rates:\n{mutation_counts}")
    print(mutation_counts.index)

    class KmerMutationModel:
        def __init__(self, prob_table):
            self.prob_table = prob_table # prob_table is a pandas dataframe
        
        def predict(self, kmer):
            return self.prob_table.loc[[kmer]].iloc[0] if kmer in self.prob_table.index else pd.Series(0, index=self.prob_table.columns)


    def evaluate_model(model, test_df):
        y_true = []
        y_pred = []
        for _, row in test_df.iterrows():
            kmer = row["context"]
            mut = row["mut_type"]
            probs = model.predict(kmer)
            y_true.append(mut)
            y_pred.append(probs)
        pred_df = pd.DataFrame(y_pred)
        return log_loss(y_true, pred_df, labels=pred_df.columns)


    model = KmerMutationModel(mutation_counts)
    pred = model.predict("ACA")
    print(pred)

    print("Log loss over validation set:", evaluate_model(model, val_df))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_version",
        type=str,
        choices=["3fA", "3fC", "3sA", "3sC", "15fA", "15fC", "15sA", "15sC", "experiment_full", "experiment_subset"],
        required=True,
        help="Specify version of the data requested (kmer length, full or subset, A or C as reference nucleotide)"
    )

    args = parser.parse_args()
    counts_model(args.data_version)