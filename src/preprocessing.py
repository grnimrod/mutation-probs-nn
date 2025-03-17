import os
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


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
    os.makedirs("./../data/splits", exist_ok=True)

    # TODO: naming should reflect which file the splits are created of
    np.save("./../data/splits/X_train.npy", X_train)
    np.save("./../data/splits/y_train.npy", y_train)
    np.save("./../data/splits/X_val.npy", X_val)
    np.save("./../data/splits/y_val.npy", y_val)
    np.save("./../data/splits/X_test.npy", X_test)
    np.save("./../data/splits/y_test.npy", y_test)


preprocess_data("./../data/DNM_15mer_v1_subset_A.tsv")