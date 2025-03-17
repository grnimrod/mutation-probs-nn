from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import h5py


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

        # array_lean = np.column_stack((kmer_list_flattened, res_list))

        return X, y

    X_train, y_train = one_hot_encode(train_dataset)
    X_val, y_val = one_hot_encode(val_dataset)
    X_test, y_test = one_hot_encode(test_dataset)
    
    # Save one-hot encoded splits using h5py
    with h5py.File("./../data/dataset.hdf5", "w") as f: # TODO: add naming convention that reflects version of the data (A or C, subset or full)
        group = f.create_group("group")

        group.create_dataset("X_train", data=X_train)
        group.create_dataset("y_train", data=y_train)
        group.create_dataset("X_val", data=X_val)
        group.create_dataset("y_val", data=y_val)
        group.create_dataset("X_test", data=X_test)
        group.create_dataset("y_test", data=y_test)


preprocess_data("./../data/15mer_A.tsv")