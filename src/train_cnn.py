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


def train_cnn(data_version):
    print(f"Version of the data: {data_version}")

    X_train, y_train, X_val, y_val, X_test, y_test = load_splits(data_version)

    expl_files = [X_train, X_val, X_test]
    expl_files = [file.view(file.size(0), file.size(1) // 4, 4).transpose(1, 2) for file in expl_files]
    X_train, X_val, X_test = expl_files

    y_train = torch.argmax(y_train, dim=1).long()
    y_val = torch.argmax(y_val, dim=1).long()
    y_test = torch.argmax(y_test, dim=1).long()

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    # Inspect example feature-label pair
    feature, label = train_dataset[100]
    print(f"Example feature: {feature}\nexample label: {label}")

    # Choose device
    device = "cpu" # CPU for now, when working with full dataset, include option to utilize GPU

    # Set up network architecture
    class CustomCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.LazyConv1d(out_channels=16, kernel_size=3, padding=1), nn.LazyBatchNorm1d(), nn.ReLU(),
                # nn.AvgPool1d(kernel_size=2, stride=2),
                nn.LazyConv1d(out_channels=64, kernel_size=3, padding=1), nn.LazyBatchNorm1d(), nn.ReLU(),
                # nn.AvgPool1d(kernel_size=2, stride=2),
                nn.LazyConv1d(out_channels=256, kernel_size=3, padding=1), nn.LazyBatchNorm1d(), nn.ReLU(),
                nn.Flatten(),
                nn.LazyLinear(32), nn.LazyBatchNorm1d(), nn.ReLU(),
                nn.LazyLinear(4)
            )
        
        def forward(self, x):
            x = self.net(x)
            return x
        
        def predict_proba(self, x):
            logits = self.forward(x)
            return F.softmax(logits, dim=-1)
        
        def layer_summary(self, x_shape):
            x = torch.randn(*x_shape)
            for layer in self.net:
                x = layer(x)
                print(layer.__class__.__name__, "output shape:\t", x.shape)


    # Set model parameters
    lr = 0.001
    epochs = 40
    bs = 256
    print(f"Learning rate: {lr}\nEpochs: {epochs}\nBatch size: {bs}")

    train_losses, val_losses = [], []

    print(f"Parameter values:\nlr: {lr},\nbatch size: {bs}")

    model = CustomCNN()
    with torch.no_grad():
        model(X_train[:2])
    
    # model.layer_summary((2, 4, 15))
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)

    # Wrap DataLoader iterator around our custom dataset(s)
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)

    loss_func = nn.CrossEntropyLoss()

    # example_out = model.predict_proba(feature)
    # print(f"Example of model output: {example_out}, {example_out.shape}")

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

    plots_dir = f"/faststorage/project/MutationAnalysis/Nimrod/results/figures/cnn/{data_version}"
    os.makedirs(plots_dir, exist_ok=True)

    plt.plot(train_losses, label="Training loss")
    plt.plot(val_losses, label="Validation loss")
    plt.legend()
    plt.title(f"Loss over epochs (data version: {data_version})")
    timestamp = datetime.now().strftime("%m-%d_%H-%M-%S")
    print(f"Figure timestamp: {timestamp}")
    plt.savefig(f"{plots_dir}/loss_curves_cnn_{timestamp}.png")
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
    train_cnn(args.data_version)