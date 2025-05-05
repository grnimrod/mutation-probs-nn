import os
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import argparse


def preprocess_data(tsv_data, train_ratio=0.7, random_state=42):
    """
    Preprocessing function. Takes in tsv file, one-hot encodes
    local feature and label columns, takes expanded feature as
    average mutation rate in the bin, then creates train val test
    local feature, expanded feature and label arrays separately,
    nine files in total. Handles experiment files separated from
    regular files.
    """

    # Read in tsv file
    df = pd.read_csv(tsv_data, sep="\t")

    # Create train val test splits
    train_dataset, temp_dataset = train_test_split(df, train_size=train_ratio, random_state=random_state, stratify=df["mut"])
    val_dataset, test_dataset = train_test_split(temp_dataset, test_size=0.5, random_state=random_state, stratify=temp_dataset["mut"])
    
    def one_hot_encode(dataframe):
        # Convert arrays to 2D arrays of characters ("ACA" -> ["A", "C", "A"])
        kmer_array = np.array([list(kmer) for kmer in dataframe["context"].to_numpy()]) # to_numpy(): np representation of the column
        
        vocab = np.array(["A", "C", "G", "T"])
        char_to_index = {char: idx for idx, char in enumerate(vocab)}

        # Map characters to indices
        index_array = np.vectorize(char_to_index.get)(kmer_array) # np.vectorize takes a function and applies it to every element of an array

        # One-hot encode indices
        X = np.eye(4)[index_array] # np.eye(4) is a 4x4 identity matrix

        X = X.reshape(X.shape[0], -1)

        res_char = np.vectorize(char_to_index.get)(dataframe["mut_res"].to_numpy())
        y = np.eye(4)[res_char]
        
        return X, y


    X_local_train, y_train = one_hot_encode(train_dataset)
    X_local_val, y_val = one_hot_encode(val_dataset)
    X_local_test, y_test = one_hot_encode(test_dataset)

    # array_total_length = len(y_train) + len(y_val) + len(y_test)
    # print(f"Total length of splits: {array_total_length}")

    X_region_train = train_dataset["avg_mut_1mb"].to_numpy()
    X_region_val = val_dataset["avg_mut_1mb"].to_numpy()
    X_region_test = test_dataset["avg_mut_1mb"].to_numpy()
    
    # Save one-hot encoded splits
    filepath = "/faststorage/project/MutationAnalysis/Nimrod/data/splits"
    os.makedirs(filepath, exist_ok=True)

    splitnames = ["X_local_train", "X_region_train", "y_train", "X_local_val", "X_region_val", "y_val", "X_local_test", "X_region_test", "y_test"]
    splits = [X_local_train, X_region_train, y_train, X_local_val, X_region_val, y_val, X_local_test, X_region_test, y_test]

    # Naming convention
    if "experiment" not in tsv_data:
        filename = tsv_data.split("/")[-1].split(".")[0] # Name of input file with path leading to it and file extension stripped
        parts = filename.split("_") # Naming of possible input files: kmer_fullorsubset_refnucl
        if len(parts) != 3:
            raise ValueError(f"Unexpected filename format: {filename}")
        kmer_name, size_info, ref_nucl = parts

        folder_name = f"{kmer_name}_{size_info}_{ref_nucl}"
        os.makedirs(f"{filepath}/{folder_name}", exist_ok=True)

        for name, data in zip(splitnames, splits):
            np.save(f"{filepath}/{folder_name}/{name}_{kmer_name}_{size_info}_{ref_nucl}.npy", data)

    elif "experiment" in tsv_data:
        filename = tsv_data.split("/")[-1].split(".")[0]
        size_info = filename.split("_")[-1]

        folder_name = f"experiment_{size_info}"
        os.makedirs(f"{filepath}/{folder_name}", exist_ok=True)
        for name, data in zip(splitnames, splits):
            np.save(f"{filepath}/{folder_name}/{name}_experiment_{size_info}.npy", data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create splits with corresponding naming")
    parser.add_argument("tsv_data", type=str, help="Name of the tsv file that splits should be created of")

    args = parser.parse_args()
    preprocess_data(args.tsv_data)