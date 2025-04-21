import os
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import argparse


def preprocess_data(tsv_data, train_ratio=0.7, random_state=42):
    """
    Preprocessing function. Takes in tsv file, one-hot encodes
    feature and label columns, then creates train val test
    feature and label arrays separately, six files in total.
    Handles experiment files separated from regular files.
    """

    # Read in tsv file
    df = pd.read_csv(tsv_data, sep="\t")

    # Create train val test splits
    train_dataset, temp_dataset = train_test_split(df, train_size=train_ratio, random_state=random_state, stratify=df["mut"])
    val_dataset, test_dataset = train_test_split(temp_dataset, test_size=0.5, random_state=random_state, stratify=temp_dataset["mut"])

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
    filepath = "/faststorage/project/MutationAnalysis/Nimrod/data/splits"
    os.makedirs(filepath, exist_ok=True)

    splitnames = ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"]
    splits = [X_train, y_train, X_val, y_val, X_test, y_test]

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

        folder_name = "experiment"
        os.makedirs(f"{filepath}/{folder_name}", exist_ok=True)
        for name, data in zip(splitnames, splits):
            np.save(f"{filepath}/{folder_name}/{name}_{size_info}_experiment.npy", data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create splits with corresponding naming")
    parser.add_argument("tsv_data", type=str, help="Name of the tsv file that splits should be created of")

    args = parser.parse_args()
    preprocess_data(args.tsv_data)