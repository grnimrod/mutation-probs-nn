import os
from datetime import datetime
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import argparse

from utils.load_splits import load_splits
from utils.early_stopping import EarlyStopping
from model_definitions import ModularModel


def train_combined(data_version):
    """
    Train models on all combinations of expanded region
    features.
    """

    BIN_SIZES = ["100kb", "500kb", "1mb"]
    MODEL_VARIANTS = {
        "avgmut": {"use_avg_mut": True, "use_bin_id_embed": False, "use_bin_id_norm": False},
        "binid": {"use_avg_mut": False, "use_bin_id_embed": True, "use_bin_id_norm": False},
        "binnorm": {"use_avg_mut": False, "use_bin_id_embed": False, "use_bin_id_norm": True},
        "avgmut_binid": {"use_avg_mut": True, "use_bin_id_embed": True, "use_bin_id_norm": False},
        "avgmut_binnorm": {"use_avg_mut": True, "use_bin_id_embed": False, "use_bin_id_norm": True},
        "binid_binnorm": {"use_avg_mut": False, "use_bin_id_embed": True, "use_bin_id_norm": True},
        "avgmut_binid_binnorm": {"use_avg_mut": True, "use_bin_id_embed": True, "use_bin_id_norm": True},
    }

    MODEL_BASE_PATH = f"/faststorage/project/MutationAnalysis/Nimrod/results/models/combined"

    for bin_size in BIN_SIZES:
        # Load data
        X_local_train, X_avg_mut_train, X_bin_id_train, y_train, \
        X_local_val, X_avg_mut_val, X_bin_id_val, y_val, \
        _, _, X_bin_id_test, _ = load_splits(data_version, requested_splits=["train", "val", "test"], bin_size=bin_size, requested_features=["avg_mut", "bin_id"])

        # Infer nr of bins across whole data (for normalization and embedding)
        all_bins = np.concatenate((X_bin_id_train, X_bin_id_val, X_bin_id_test), axis=0)
        unique_bins = set(all_bins.squeeze(1).tolist())
        n_bins = len(unique_bins)

        # Normalize integer bin IDs
        X_bin_id_normalized_train = X_bin_id_train.float() / n_bins
        X_bin_id_normalized_val = X_bin_id_val.float() / n_bins

        # Set model parameters
        lr = 0.0001
        epochs = 100
        bs = 64
        train_losses, val_losses = [], []

        # Train each model
        for variant_name, config in MODEL_VARIANTS.items():
            print(f"Training {variant_name} model variant on {data_version} data and {bin_size} bin size")

            early_stopping = EarlyStopping()

            train_inputs = [X_local_train]
            val_inputs = [X_local_val]

            if config["use_avg_mut"]:
                train_inputs.append(X_avg_mut_train)
                val_inputs.append(X_avg_mut_val)
            if config["use_bin_id_embed"]:
                train_inputs.append(X_bin_id_train)
                val_inputs.append(X_bin_id_val)
            if config["use_bin_id_norm"]:
                train_inputs.append(X_bin_id_normalized_train)
                val_inputs.append(X_bin_id_normalized_val)
            
            train_dataset = TensorDataset(*train_inputs, y_train)
            val_dataset = TensorDataset(*val_inputs, y_val)

            train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False)

            model_kwargs = dict(config)
            if config.get("use_bin_id_embed"):
                model_kwargs["num_bins"] = n_bins
                model_kwargs["embed_dim"] = 64
            
            model = ModularModel(**model_kwargs)

            # Pass in samples to initialize weights for lazy layers
            with torch.no_grad():
                dummy_inputs = {
                    "local_context": X_local_train[:2]
                }

                if config.get("use_avg_mut"):
                    dummy_inputs["avg_mut"] = X_avg_mut_train[:2]
                if config.get("use_bin_id_embed"):
                    dummy_inputs["bin_id"] = X_bin_id_train[:2]
                if config.get("use_bin_id_norm"):
                    dummy_inputs["bin_id_norm"] = X_bin_id_normalized_train[:2]
                
                model(**dummy_inputs)
            
            # Choose device
            device = "cpu"
            if torch.cuda.is_available():
                device = "cuda:0"
                if torch.cuda.device_count() > 1:
                    model = nn.DataParallel(model)
            model.to(device)
            print(f"Using device: {device}")

            opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

            loss_func = nn.CrossEntropyLoss()

            for epoch in range(epochs):
                # Training phase
                model.train()
                train_running_loss = 0.0
                for batch in train_loader:
                    *features, label = batch

                    idx = 0
                    local_context = features[idx]
                    idx += 1

                    if config["use_avg_mut"]:
                        avg_mut = features[idx]
                        idx += 1
                    else:
                        avg_mut = None
                    
                    if config["use_bin_id_embed"]:
                        bin_id = features[idx]
                        idx += 1
                    else:
                        bin_id = None
                    
                    if config["use_bin_id_norm"]:
                        bin_id_norm = features[idx]
                        idx += 1
                    else:
                        bin_id_norm = None
                    
                    pred = model(local_context=local_context, avg_mut=avg_mut, bin_id=bin_id, bin_id_norm=bin_id_norm)
                    loss = loss_func(pred, label)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    train_running_loss += loss.item() * label.size(0) # Default reduction of CrossEntropyLoss() is 'mean' -> multiply by batch size to get loss per batch
                train_loss = train_running_loss / len(train_loader.dataset)
                train_losses.append(train_loss)
                
                # Validation phase
                model.eval()
                val_running_loss = 0.0
                with torch.no_grad():
                    for batch in val_loader:
                        *features, label = batch

                        idx = 0
                        local_context = features[idx]
                        idx += 1

                        if config["use_avg_mut"]:
                            avg_mut = features[idx]
                            idx += 1
                        else:
                            avg_mut = None
                        
                        if config["use_bin_id_embed"]:
                            bin_id = features[idx]
                            idx += 1
                        else:
                            bin_id = None
                        
                        if config["use_bin_id_norm"]:
                            bin_id_norm = features[idx]
                            idx += 1
                        else:
                            bin_id_norm = None

                        pred = model(local_context=local_context, avg_mut=avg_mut, bin_id=bin_id, bin_id_norm=bin_id_norm)
                        loss = loss_func(pred, label)
                        val_running_loss += loss.item() * label.size(0)
                val_loss = val_running_loss / len(val_loader.dataset)
                val_losses.append(val_loss)

                print(f"Epoch {epoch + 1}/{epochs} train loss: {train_loss:.6f} val loss: {val_loss:.6f}")

                early_stopping(val_loss, model)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

            early_stopping.load_best_model(model)

            model_dir = f"{MODEL_BASE_PATH}/{data_version}/{variant_name}"
            os.makedirs(model_dir, exist_ok=True)
            torch.save(model.state_dict(), f"{model_dir}/model_{bin_size}.pth")

            plots_dir = f"/faststorage/project/MutationAnalysis/Nimrod/results/figures/combined/{data_version}/{variant_name}"
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