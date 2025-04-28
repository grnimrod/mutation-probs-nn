import os
import numpy as np
import torch


def load_splits(data_version, convert_to_tensor=True):
    """
    Load specified version of splits for model training
    """

    filepath = "/faststorage/project/MutationAnalysis/Nimrod/data/splits/"

    version_map = {
        "3fA": "3mer_full_A",
        "3fC": "3mer_full_C",
        "3sA": "3mer_subset_A",
        "3sC": "3mer_subset_C",
        "5fA": "5mer_full_A",
        "5fC": "5mer_full_C",
        "7fA": "7mer_full_A",
        "7fC": "7mer_full_C",
        "9fA": "9mer_full_A",
        "9fC": "9mer_full_C",
        "11fA": "11mer_full_A",
        "11fC": "11mer_full_C",
        "13fA": "13mer_full_A",
        "13fC": "13mer_full_C",
        "15fA": "15mer_full_A",
        "15fC": "15mer_full_C",
        "15sA": "15mer_subset_A",
        "15sC": "15mer_subset_C",
        "experiment_full": "experiment_full",
        "experiment_subset": "experiment_subset"
    }

    if data_version not in version_map:
        raise ValueError(f"Invalid file version specification: {data_version}")
    
    folder = version_map[data_version]
    suffix = folder # Suffix is the same as the folder name

    splits = ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"]
    filepaths = [os.path.join(filepath, folder, f"{split}_{suffix}.npy") for split in splits]

    if convert_to_tensor:
        tensors = [torch.as_tensor(np.load(file), dtype=torch.float32) for file in filepaths]
        return tuple(tensors)
    
    if not convert_to_tensor:
        arrays = [np.load(file) for file in filepaths]
        return tuple(arrays)