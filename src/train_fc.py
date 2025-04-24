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

    print(f"Version of the data: {data_version}")

    X_train, y_train, X_val, y_val, X_test, y_test = load_splits(data_version)

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
    epochs = 100
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

    opt = optim.Adam(model.parameters(), lr=lr) # TODO: possibly add weight decay
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
        val_correct = torch.zeros(yb.shape[1], device=device)
        val_total = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = loss_func(pred, yb)
                val_running_loss += loss.item() * yb.size(0)

                probs = torch.sigmoid(pred)
                pred_labels = (probs >= 0.5).float()
                correct = (pred_labels == yb).float().sum(dim=0)
                val_correct += correct
                val_total += yb.size(0)

        val_loss = val_running_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        val_acc_per_label = val_correct / val_total
        mean_val_acc = val_acc_per_label.mean().item()

        # scheduler.step(val_loss)

        print(f"Epoch {epoch + 1}/{epochs} val loss: {val_loss:.4f}, val acc: {mean_val_acc:.4f}")


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
        choices=["3fA", "3fC", "3sA", "3sC", "15fA", "15fC", "15sA", "15sC", "experiment_full", "experiment_subset"],
        required=True,
        help="Specify version of the data requested (kmer length, full or subset, A or C as reference nucleotide)"
    )

    args = parser.parse_args()
    train_fc(args.data_version)