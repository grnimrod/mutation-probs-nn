import os
from datetime import datetime
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import argparse

from utils.load_splits import load_splits
from model_definitions import ModularModel
from utils.early_stopping import EarlyStopping


def train_combined(data_version):
    """
    Train combined model
    """

    print("Training on average mutation rate, bin ID and normalized bin ID")
    print(f"Version of the data: {data_version}")

    bin_sizes = ["100kb", "500kb", "1mb"]

    for bin_size in bin_sizes:
        print(f"Training using data with bin size {bin_size}")
        
        X_local_train, X_avg_mut_1mb_train, X_bin_id_1mb_train, y_train, X_local_val, X_avg_mut_1mb_val, X_bin_id_1mb_val, y_val, _, _, X_bin_id_1mb_test, _ = load_splits(data_version, requested_splits=["train", "val", "test"], bin_size=bin_size, requested_features=["avg_mut", "bin_id"])

        all_bins = np.concatenate((X_bin_id_1mb_train, X_bin_id_1mb_val, X_bin_id_1mb_test), axis=0)
        unique_bins = set(all_bins)
        n_bins = len(unique_bins)

        X_bin_id_1mb_normalized_train = X_bin_id_1mb_train.float() / n_bins
        X_bin_id_1mb_normalized_val = X_bin_id_1mb_val.float() / n_bins

        X_avg_mut_1mb_train = X_avg_mut_1mb_train.unsqueeze(1)
        X_avg_mut_1mb_val = X_avg_mut_1mb_val.unsqueeze(1)
        X_bin_id_1mb_train = X_bin_id_1mb_train.unsqueeze(1)
        X_bin_id_1mb_val = X_bin_id_1mb_val.unsqueeze(1)
        X_bin_id_1mb_normalized_train = X_bin_id_1mb_normalized_train.unsqueeze(1)
        X_bin_id_1mb_normalized_val = X_bin_id_1mb_normalized_val.unsqueeze(1)

        print(f"X_local_train shape: {X_local_train.shape}")
        print(f"X_avg_mut_1mb_train shape: {X_bin_id_1mb_train.shape}")

        train_dataset = TensorDataset(X_local_train, X_avg_mut_1mb_train, X_bin_id_1mb_train, X_bin_id_1mb_normalized_train, y_train)
        val_dataset = TensorDataset(X_local_val, X_avg_mut_1mb_val, X_bin_id_1mb_val, X_bin_id_1mb_normalized_val, y_val)

        lr = 0.001
        epochs = 100
        bs = 64
        train_losses, val_losses = [], []

        early_stopping = EarlyStopping()

        print(f"Parameter values:\nlr: {lr},\nbatch size: {bs}")

        model = ModularModel(use_avg_mut=True, use_bin_id_embed=True, use_bin_id_norm=True, num_bins=n_bins, embed_dim=64)

        with torch.no_grad():
            model(local_context=X_local_train[:2], avg_mut=X_avg_mut_1mb_train[:2], bin_id=X_bin_id_1mb_train[:2], bin_id_norm=X_bin_id_1mb_normalized_train[:2])
        
        # Choose device
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
        model.to(device)
        print(f"Using device: {device}")

        opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

        train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False)

        loss_func = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            # Training phase
            model.train()
            running_loss = 0.0
            for xb_local, xb_avg_mut, xb_bin_id, xb_bin_id_norm, yb in train_loader:
                pred = model(local_context=xb_local, avg_mut=xb_avg_mut, bin_id=xb_bin_id, bin_id_norm=xb_bin_id_norm)
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
                for xb_local, xb_avg_mut, xb_bin_id, xb_bin_id_norm, yb in val_loader:
                    pred = model(local_context=xb_local, avg_mut=xb_avg_mut, bin_id=xb_bin_id, bin_id_norm=xb_bin_id_norm)
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

        model_dir = f"/faststorage/project/MutationAnalysis/Nimrod/results/models/combined/{data_version}/avgmut_binid_binnorm"
        os.makedirs(model_dir, exist_ok=True)
        torch.save(model.state_dict(), f"{model_dir}/model_{bin_size}.pth")

        plots_dir = f"/faststorage/project/MutationAnalysis/Nimrod/results/figures/combined/{data_version}/avgmut_binid_binnorm"
        os.makedirs(plots_dir, exist_ok=True)

        plt.plot(train_losses, label="Training loss")
        plt.plot(val_losses, label="Validation loss")
        plt.legend()
        plt.title(f"Loss over epochs (data version: {data_version})\nWeight decay")
        plt.xlabel("Epoch")
        plt.ylabel("Cross Entropy Loss")
        timestamp = datetime.now().strftime("%m-%d_%H-%M-%S")
        print(f"Figure timestamp: {timestamp}")
        plt.savefig(f"{plots_dir}/loss_curves_{bin_size}_{timestamp}.png")
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