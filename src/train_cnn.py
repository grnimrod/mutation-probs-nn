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
    X_train, y_train, X_val, y_val, X_test, y_test = load_splits(data_version)

    expl_files = [X_train, X_val, X_test]
    expl_files = [file.view(file.size(0), 15, 4).transpose(1, 2) for file in expl_files]
    X_train, X_val, X_test = expl_files

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
                nn.LazyConv1d(out_channels=16, kernel_size=3), nn.LazyBatchNorm1d(), nn.ReLU(),
                nn.AvgPool1d(kernel_size=2, stride=2),
                nn.LazyConv1d(out_channels=64, kernel_size=3), nn.LazyBatchNorm1d(), nn.ReLU(),
                # nn.AvgPool1d(kernel_size=2, stride=2),
                nn.LazyConv1d(out_channels=256, kernel_size=3), nn.LazyBatchNorm1d(), nn.ReLU(),
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
    lr = 0.01
    epochs = 40
    bs = 64
    print(f"Learning rate: {lr}\nEpochs: {epochs}\nBatch size: {bs}")

    train_losses, val_losses = [], []

    model = CustomCNN().to(device)
    model.layer_summary((2, 4, 15))
    opt = optim.Adam(model.parameters(), lr=lr)

    # Wrap DataLoader iterator around our custom dataset(s)
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=bs*2, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=bs*2, shuffle=False)

    loss_func = nn.CrossEntropyLoss()

    # example_out = model.predict_proba(feature)
    # print(f"Example of model output: {example_out}, {example_out.shape}")

    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            yb_indices = yb.argmax(dim=1)
            xb, yb_indices = xb.to(device), yb_indices.to(device)

            pred = model(xb)
            loss = loss_func(pred, yb_indices)
            loss.backward()
            opt.step()
            opt.zero_grad()
            running_loss += loss.item() * yb_indices.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for xb, yb in test_loader:
                yb_indices = yb.argmax(dim=1)
                xb, yb_indices = xb.to(device), yb_indices.to(device)

                pred = model(xb)
                loss = loss_func(pred, yb_indices)
                running_loss += loss.item() * yb_indices.size(0)
            
            val_loss = running_loss / len(val_loader.dataset)
            val_losses.append(val_loss)
        
        print(f"Epoch {epoch + 1}/{epochs} train loss: {train_loss}, validation loss: {val_loss}")

    plots_dir = "/faststorage/project/MutationAnalysis/Nimrod/results/figures/cnn"
    os.makedirs(plots_dir, exist_ok=True)

    plt.plot(train_losses, label="Training loss")
    plt.plot(val_losses, label="Validation loss")
    plt.legend()
    plt.title("Loss over epochs")
    timestamp = datetime.now().strftime("%m-%d_%H-%M-%S")
    plt.savefig(f"{plots_dir}/loss_curves_cnn_{timestamp}.png")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_version",
        type=str,
        choices=["fA", "fC", "sA", "sC"],
        required=True,
        help="Specify version of the data requested (full or subset, A or C as reference nucleotide)"
    )

    args = parser.parse_args()
    train_cnn(args.data_version)