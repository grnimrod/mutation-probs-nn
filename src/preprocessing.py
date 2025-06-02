import os
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import argparse


def preprocess_data(tsv_data, train_ratio=0.7, random_state=42):
    """
    Preprocessing function. Takes in tsv file as dataframe, creates bin ID
    columns over specified bin sizes, creates average mutation rate per bin
    columns, splits dataframe into train val test, one-hot encodes local
    context feature and label arrays, then saves all arrays created.
    Handles experiment files separated from regular files.
    """

    # Read in tsv file
    df = pd.read_csv(tsv_data, sep="\t")

    # Encode mb_bin column to integer labels
    def natural_key(s):
        """
        Helper function to be used as key for natural sorting
        """
        return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', s)] # Transforms chr10_3 into ['chr', 10, '_', 3, ''], with digits being integers

    bins = ["100kb", "500kb", "1mb"] # Specify
    bin_sizes = [100_000, 500_000, 1_000_000]

    # Create derived feature columns
    for label, size in zip(bins, bin_sizes):
        # Create bins
        df[f"bin_{label}"] = df["chrom"] + "_" + (df["pos"] // size).astype(str)

        sorted_bins = sorted(df[f"bin_{label}"].unique(), key=natural_key)
        label_map = {bin: index for index, bin in enumerate(sorted_bins)}

        # Integer encode bins
        df[f"bin_id_{label}"] = df[f"bin_{label}"].map(label_map)

        # Compute average mutation rates in bins
        # df[f"avg_mut_{label}"] = df.groupby(f"bin_id_{label}")["mut"].transform("mean")

    # Create train val test splits
    train_dataset, temp_dataset = train_test_split(df, train_size=train_ratio, random_state=random_state, stratify=df["mut"])
    val_dataset, test_dataset = train_test_split(temp_dataset, test_size=0.5, random_state=random_state, stratify=temp_dataset["mut"])

    # Add both overall and outcome-specific average mutation rate per bin columns
    def add_avg_mut_per_bin_columns(df):
        for label in bins:
            # Add overall average mutation rate
            df[f"avg_mut_{label}"] = df.groupby(f"bin_id_{label}")["mut"].transform("mean")

            # Count all rows per group
            group_counts = df.groupby(f"bin_id_{label}").size()
            for base in ["A", "C", "G", "T"]:
                # Filter down data to specific mutation outcome
                df_filtered = df[df["mut_res"] == base]

                # Count filtered rows per group
                filtered_group_counts = df_filtered.groupby(f"bin_id_{label}").size()

                # Obtain ratio of specific outcome to all outcomes
                type_spec_mut_ratio_per_bin = (filtered_group_counts / group_counts).fillna(0.0)

                # Map returned Series on original dataframe, add this as new column
                df[f"avg_mut_{label}_{base}"] = df[f"bin_id_{label}"].map(type_spec_mut_ratio_per_bin)
        return df

    train_dataset = add_avg_mut_per_bin_columns(train_dataset)
    val_dataset = add_avg_mut_per_bin_columns(val_dataset)
    test_dataset = add_avg_mut_per_bin_columns(test_dataset)
    
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

    splitnames = ["X_local_train", "y_train", "X_local_val", "y_val", "X_local_test", "y_test"]
    splits = [X_local_train, y_train, X_local_val, y_val, X_local_test, y_test]

    for label in bins:
        for split_name, dataset in zip(["train", "val", "test"], [train_dataset, val_dataset, test_dataset]):

            # Add bin id columns to splits to be saved
            split_bin_id = f"X_bin_id_{label}_{split_name}"
            splitnames.append(split_bin_id)
            splits.append(dataset[f"bin_id_{label}"].to_numpy().reshape(-1, 1))

            # Add avg mut rate columns to splits to be saved
            split_avg_mut = f"X_avg_mut_{label}_{split_name}"
            splitnames.append(split_avg_mut)
            splits.append(dataset[f"avg_mut_{label}"].to_numpy().reshape(-1, 1))

            # Add outcome-specific avg mut rate columns to splits to be saved (as one array)
            split_avg_mut_per_type = f"X_avg_mut_{label}_pertype_{split_name}"
            splitnames.append(split_avg_mut_per_type)
            splits.append(dataset[[f"avg_mut_{label}_{base}" for base in ["A", "C", "G", "T"]]].to_numpy())
    
    filepath = "/faststorage/project/MutationAnalysis/Nimrod/data/splits"
    os.makedirs(filepath, exist_ok=True)

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