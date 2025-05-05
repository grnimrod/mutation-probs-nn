import os
from datetime import datetime
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import argparse

from utils.load_splits import load_splits
from model_definitions import LocalModule, ExpandedModule, CombinedMLP, MixedInputModel
from utils.early_stopping import EarlyStopping


def train_combined(data_version):
    """
    Train model with separate modules for handling input
    features, then head module for combining them
    """

    print("Training with both the sequence context and expanded region as features")
    print(f"Version of the data: {data_version}")

    X_local_train, X_region_train, y_train, X_local_val, X_region_val, y_val, X_local_test, X_region_test, y_test = load_splits(data_version)
    print(f"X_local_train shape: {X_local_train.shape}")
    print(f"X_region_train shape: {X_region_train.shape}")

    label_encoder = LabelEncoder()

    X_region_train = torch.as_tensor(label_encoder.fit_transform(X_region_train), dtype=torch.long)
    X_region_val = torch.as_tensor(label_encoder.fit_transform(X_region_val), dtype=torch.long)
    X_region_test = torch.as_tensor(label_encoder.fit_transform(X_region_test), dtype=torch.long)

    all_bins = np.concatenate((X_region_train, X_region_val, X_region_test), axis=0)
    unique_bins = set(all_bins)
    n_bins = len(unique_bins)
    print("Unique bin IDs:", n_bins)
    print(all_bins.min(), all_bins.max())

    train_dataset = TensorDataset(X_local_train, X_region_train, y_train)
    val_dataset = TensorDataset(X_local_val, X_region_val, y_val)
    test_dataset = TensorDataset(X_local_test, X_region_test, y_test)

    lr = 0.001
    epochs = 100
    bs = 512
    train_losses, val_losses = [], []

    early_stopping = EarlyStopping()

    print(f"Parameter values:\nlr: {lr},\nbatch size: {bs}")

    local_model = LocalModule()
    expanded_model = ExpandedModule(num_embeddings=n_bins, embedding_dim=52)
    combined_model = CombinedMLP(input_dim=64, output_dim=4)  # 32+32 -> 4 classes

    model = MixedInputModel(local_model, expanded_model, combined_model)

    with torch.no_grad():
        model((X_local_train[:2], X_region_train[:2]))
    
    # Choose device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)
    print(f"Using device: {device}")

    opt = optim.Adam(model.parameters(), lr=lr)

    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False)

    loss_func = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        for xb_local, xb_region, yb in train_loader:
            xb_input = (xb_local, xb_region) # Model expects a tuple for input
            pred = model(xb_input)
            loss = loss_func(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running_loss += loss.item() * yb.size(0) # Default reduction of CrossEntropyLoss() is 'mean' -> multiply by batch size to get loss per batch
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for xb_local, xb_region, yb in val_loader:
                xb_input = (xb_local, xb_region)
                pred = model(xb_input)
                loss = loss_func(pred, yb)
                val_running_loss += loss.item() * yb.size(0)
        val_loss = val_running_loss / len(val_loader.dataset)
        val_losses.append(val_loss)

        print(f"Epoch {epoch + 1}/{epochs} train loss: {train_loss:.6f} val loss: {val_loss:.6f}")

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    early_stopping.load_best_model(model)

    plots_dir = f"/faststorage/project/MutationAnalysis/Nimrod/results/figures/combined/{data_version}"
    os.makedirs(plots_dir, exist_ok=True)

    plt.plot(train_losses, label="Training loss")
    plt.plot(val_losses, label="Validation loss")
    plt.legend()
    plt.title(f"Loss over epochs (data version: {data_version})")
    plt.xlabel("Epoch")
    plt.ylabel("Cross Entropy Loss")
    timestamp = datetime.now().strftime("%m-%d_%H-%M-%S")
    print(f"Figure timestamp: {timestamp}")
    plt.savefig(f"{plots_dir}/loss_curves_fc_{timestamp}.png")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_version",
        type=str,
        choices=[
            "3fA", "3fC", "3sA", "3sC", "5fA", "5fC", "7fA", "7fC", "9fA", "9fC",
            "11fA", "11fC", "13fA", "13fC", "15fA", "15fC", "15sA", "15sC",
            "experiment_full", "experiment_subset"
            ],
        required=True,
        help="Specify version of the data requested (kmer length, full or subset, A or C as reference nucleotide)"
    )

    args = parser.parse_args()
    train_combined(args.data_version)