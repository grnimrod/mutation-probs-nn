import os
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import numpy as np


def create_splits(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
    mut_labels = dataset.get_mut_labels()

    temp_indices, val_indices, temp_mut, _ = train_test_split(
        range(len(dataset)), # Indices of data items in the dataset
        mut_labels, # List of labels for stratified sampling
        test_size=val_ratio,
        random_state=random_state,
        stratify=mut_labels
    )

    temp_dataset = Subset(dataset, temp_indices)
    val_dataset = Subset(dataset, val_indices)

    train_indices, test_indices, _, _ = train_test_split(
        range(len(temp_dataset)),
        temp_mut,
        test_size=test_ratio/(train_ratio+test_ratio),
        random_state=random_state,
        stratify=temp_mut
    )

    train_dataset = Subset(temp_dataset, train_indices)
    test_dataset = Subset(temp_dataset, test_indices)

    os.makedirs("splits", exist_ok=True)

    np.save("splits/train_indices.npy", train_indices)
    np.save("splits/val_indices.npy", val_indices)
    np.save("splits/test_indices.npy", test_indices)

    return train_dataset, val_dataset, test_dataset