import os
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import argparse

from utils.load_splits import load_splits


def train_fc(data_version):
    """
    Load in splits, set up model architecture, train model, save loss curves figure
    """

    X_train, y_train, X_val, y_val, X_test, y_test = load_splits(data_version)

    y_train = torch.argmax(y_train, dim=1).long()
    y_val = torch.argmax(y_val, dim=1).long()
    y_test = torch.argmax(y_test, dim=1).long()

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    # Inspect example feature-label pair
    feature, label = train_dataset[100]
    print(f"Example feature: {feature}\nexample label: {label}")

    # Set up network architecture
    class FullyConnectedNN(nn.Module):
        def __init__(self):
            super().__init__()

            self.linear_relu_seq = nn.Sequential(
                nn.LazyLinear(128),
                nn.ReLU(),
                # nn.Dropout(p=0.3),
                nn.LazyLinear(64),
                nn.ReLU(),
                # nn.Dropout(p=0.3),
                nn.LazyLinear(4)
            )
        
        def forward(self, x):
            x = self.linear_relu_seq(x)
            return x
        
        def predict_proba(self, x):
            logits = self.linear_relu_seq(x)
            return F.softmax(logits, dim=-1)


    # Set model parameters
    lr = 0.001
    epochs = 40
    bs = 64
    train_losses, val_losses = [], []

    model = FullyConnectedNN()
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

    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

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
            # yb = yb.argmax(dim=1) # CrossEntropyLoss() expects class index as target

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
        running_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                # yb = yb.argmax(dim=1)

                pred = model(xb)
                loss = loss_func(pred, yb)
                running_loss += loss.item() * yb.size(0)
            val_loss = running_loss / len(val_loader.dataset)
            val_losses.append(val_loss)
        
        print(f"Epoch {epoch + 1}/{epochs} train loss: {train_loss}, validation loss: {val_loss}")

    plots_dir = "/faststorage/project/MutationAnalysis/Nimrod/results/figures/fc"
    os.makedirs(plots_dir, exist_ok=True)

    plt.plot(train_losses, label="Training loss")
    plt.plot(val_losses, label="Validation loss")
    plt.legend()
    plt.title("Loss over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Cross Entropy Loss")
    timestamp = datetime.now().strftime("%m-%d_%H-%M-%S")
    plt.savefig(f"{plots_dir}/loss_curves_fc_{timestamp}.png")
    plt.close()

    example_out = model(feature.unsqueeze(0)) # unsqueeze(0) to add a "batch" dimension of 1 at position 1
    print(f"Example of model output: {example_out}, {example_out.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_version",
        type=str,
        choices=["3fA", "3fC", "3sA", "3sC", "15fA", "15fC", "15sA", "15sC", "experiment"],
        required=True,
        help="Specify version of the data requested (kmer length, full or subset, A or C as reference nucleotide)"
    )

    args = parser.parse_args()
    train_fc(args.data_version)