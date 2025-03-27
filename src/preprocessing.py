import os
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import argparse


def preprocess_data(tsv_data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
    # Read in tsv file
    df = pd.read_csv(tsv_data, sep="\t")

    # Create train val test splits
    temp_dataset, val_dataset = train_test_split(df, test_size=val_ratio, random_state=random_state, stratify=df["mut"])
    train_dataset, test_dataset = train_test_split(
        temp_dataset,
        test_size=test_ratio/(train_ratio + test_ratio),
        random_state=random_state,
        stratify=temp_dataset["mut"]
        )

    # One-hot encode
    alphabet = {
        'A': [1, 0, 0, 0],
        'C': [0, 1, 0, 0],
        'G': [0, 0, 1, 0],
        'T': [0, 0, 0, 1]
        }

    def one_hot_encode(dataframe):
        # Initialize empty lists for results
        kmer_list = []
        res_list = []

        for index in range(len(dataframe)):
            kmer = dataframe.iloc[index]["context"]
            kmer_encoded = np.array([alphabet[nucl] for nucl in kmer])
            kmer_list.append(kmer_encoded.flatten())

            res = dataframe.iloc[index]["type"][-1]
            res_encoded = np.array(alphabet[res])
            res_list.append(res_encoded)
        
        X = np.array(kmer_list)
        y = np.array(res_list)

        return X, y

    X_train, y_train = one_hot_encode(train_dataset)
    X_val, y_val = one_hot_encode(val_dataset)
    X_test, y_test = one_hot_encode(test_dataset)
    
    # Save one-hot encoded splits
    os.makedirs("/faststorage/project/MutationAnalysis/Nimrod/data/splits", exist_ok=True)

    filenames = ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"]
    files = [X_train, y_train, X_val, y_val, X_test, y_test]

    # Naming convention
    reference_nucl = tsv_data.split(".")[0][-1] # Nucleotide at site is last character of the original filename

    if "subset" in tsv_data:
        for name, data in zip(filenames, files):
            np.save(f"/faststorage/project/MutationAnalysis/Nimrod/data/splits/{name}_subset_{reference_nucl}.npy", data)
    else:
        for name, data in zip(filenames, files):
            np.save(f"/faststorage/project/MutationAnalysis/Nimrod/data/splits/{name}_{reference_nucl}.npy", data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create splits with corresponding naming")
    parser.add_argument("tsv_data", type=str, help="Name of the tsv file that splits should be created of")

    args = parser.parse_args()    
    preprocess_data(args.tsv_data)