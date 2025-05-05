import os
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import argparse

from utils.load_splits import load_splits
from model_definitions import FullyConnectedNN, FCNNEmbedding
from utils.early_stopping import EarlyStopping


def train_fc(data_version):
    """
    Load in splits, set up model architecture, train model, save loss curves figure
    """

    print("Training on expanded version alone")
    print(f"Version of the data: {data_version}")

    _, X_region_train, y_train, _, X_region_val, y_val, _, X_region_test, y_test = load_splits(data_version)

    # label_encoder = LabelEncoder()

    # X_region_train = torch.as_tensor(label_encoder.fit_transform(X_region_train), dtype=torch.long)
    # X_region_val = torch.as_tensor(label_encoder.fit_transform(X_region_val), dtype=torch.long)
    # X_region_test = torch.as_tensor(label_encoder.fit_transform(X_region_test), dtype=torch.long)

    X_train = X_region_train.unsqueeze(1)
    X_val = X_region_val.unsqueeze(1)
    X_test = X_region_test.unsqueeze(1)

    # all_bins = np.concatenate((X_region_train, X_region_val, X_region_test), axis=0)
    # unique_bins = set(all_bins)
    # n_bins = len(unique_bins)

    y_train = torch.argmax(y_train, dim=1).long()
    y_val = torch.argmax(y_val, dim=1).long()
    y_test = torch.argmax(y_test, dim=1).long()

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    # Inspect example feature-label pair
    feature, label = train_dataset[100]
    print(f"Example feature: {feature}\nexample label: {label}")

    # Set model parameters
    lr = 0.001
    epochs = 100
    bs = 32
    train_losses, val_losses = [], []

    early_stopping = EarlyStopping()

    print(f"Parameter values:\nlr: {lr},\nbatch size: {bs}")

    model = FullyConnectedNN()
    # model = FCNNEmbedding(num_embeddings=n_bins, embedding_dim=52)
    # print(f"Embedding dimension: {model.embedding}")
    with torch.no_grad():
        model(X_train[:2]) # So that weights are initialized before moving model to different device (required due to use of LazyLinear)

    # Choose device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)
    print(f"Using device: {device}")

    opt = optim.Adam(model.parameters(), lr=lr) # TODO: possibly add weight decay (weight_decay=1e-4)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', patience=3, factor=0.5, verbose=True)

    # Wrap DataLoader iterator around our custom dataset(s)
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)

    loss_func = nn.CrossEntropyLoss()

    # Calculating accuracy will not be our focus
    def accuracy(out, yb):
        preds = torch.argmax(out, dim=1) # To make it compatible with batches (where shape is [bs, nr_classes]), use dim=1
        return (preds == torch.argmax(yb, dim=1)).float().mean()
    

    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            pred = model(xb)
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
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = loss_func(pred, yb)
                val_running_loss += loss.item() * yb.size(0)
        val_loss = val_running_loss / len(val_loader.dataset)
        val_losses.append(val_loss)

        # scheduler.step(val_loss)

        print(f"Epoch {epoch + 1}/{epochs} train loss: {train_loss:.6f} val loss: {val_loss:.6f}")

        # early_stopping(val_loss, model)
        # if early_stopping.early_stop:
        #     print("Early stopping")
        #     break

    # early_stopping.load_best_model(model)

    model_dir = f"/faststorage/project/MutationAnalysis/Nimrod/results/models/fc/{data_version}"
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), f"{model_dir}/model.pth")


    plots_dir = f"/faststorage/project/MutationAnalysis/Nimrod/results/figures/fc/{data_version}"
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

    example_out = model(feature.unsqueeze(0)) # unsqueeze(0) to add a "batch" dimension of 1 at position 1
    print(f"Example of model output: {example_out}, {example_out.shape}")


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
    train_fc(args.data_version)