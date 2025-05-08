import os
import numpy as np
import torch


def load_splits(data_version: str, requested_splits: list, bin_size: str = None, requested_features: list = None, convert_to_tensor: bool = True):
    """
    Load specified version of splits for model training.
    Order of the files returned: X_local_split, X_derived_split, y_split
    'split' should follow 'train', 'val', 'test' order
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
    
    if requested_features is None:
        requested_features = []
    
    if requested_features:
        if bin_size not in ["100kb", "500kb", "1mb"]:
            raise ValueError("Non-existent size specification")
    
    allowed_features = {"bin_id", "avg_mut"}
    if not set(requested_features).issubset(allowed_features):
        raise ValueError(f"Invalid feature(s) requested: {set(requested_features) - allowed_features}")
    
    allowed_splits = {"train", "val", "test"}
    if not set(requested_splits).issubset(allowed_splits):
        raise ValueError(f"Invalid split(s) requested: {set(requested_splits) - allowed_splits}")
    
    folder = version_map[data_version]
    suffix = folder # Suffix is the same as the folder name

    splits = []
    for split_name in requested_splits:
        splits.append(f"X_local_{split_name}")
        for feature in requested_features:
                splits.append(f"X_{feature}_{bin_size}_{split_name}")
        splits.append(f"y_{split_name}")

    filepaths = [os.path.join(filepath, folder, f"{split}_{suffix}.npy") for split in splits]

    if convert_to_tensor:
        tensors = [torch.as_tensor(np.load(file), dtype=torch.float32) if "bin_id" not in file else torch.as_tensor(np.load(file), dtype=torch.long) for file in filepaths] # Embedding requires 'long' type
        return tuple(tensors)
    
    if not convert_to_tensor:
        arrays = [np.load(file) for file in filepaths]
        return tuple(arrays)