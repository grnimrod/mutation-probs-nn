import os
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import argparse


def train_fc(data_version):
    """
    Load in splits, set up model architecture, train model, save loss curves figure
    """

    filepath = "/faststorage/project/MutationAnalysis/Nimrod/data/splits/"

    if data_version == "fA":
        filenames = ["X_train_A", "y_train_A", "X_val_A", "y_val_A", "X_test_A", "y_test_A"]
        X_train, y_train, X_val, y_val, X_test, y_test = [np.load(os.path.join(filepath, filename + ".npy")) for filename in filenames]

    elif data_version == "fC":
        filenames = ["X_train_C", "y_train_C", "X_val_C", "y_val_C", "X_test_C", "y_test_C"]
        X_train, y_train, X_val, y_val, X_test, y_test = [np.load(os.path.join(filepath, filename + ".npy")) for filename in filenames]

    elif data_version == "sA":
        filenames = ["X_train_subset_A", "y_train_subset_A", "X_val_subset_A", "y_val_subset_A", "X_test_subset_A", "y_test_subset_A"]
        X_train, y_train, X_val, y_val, X_test, y_test = [np.load(os.path.join(filepath, filename + ".npy")) for filename in filenames]

    elif data_version == "sC":
        filenames = ["X_train_subset_C", "y_train_subset_C", "X_val_subset_C", "y_val_subset_C", "X_test_subset_C", "y_test_subset_C"]
        X_train, y_train, X_val, y_val, X_test, y_test = [np.load(os.path.join(filepath, filename + ".npy")) for filename in filenames]

    else:
        print("Invalid file version specification")
        exit(1)

    files = [X_train, y_train, X_val, y_val, X_test, y_test]
    files = [torch.as_tensor(file, dtype=torch.float32) for file in files]
    X_train, y_train, X_val, y_val, X_test, y_test = files

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
                nn.Linear(in_features=15*4, out_features=512),
                nn.ReLU(),
                nn.Linear(in_features=512, out_features=256),
                nn.ReLU(),
                nn.Linear(in_features=256, out_features=128),
                nn.ReLU(),
                nn.Linear(in_features=128, out_features=4)
            )
        
        def forward(self, x):
            x = self.linear_relu_seq(x)
            return x
        
        def predict_proba(self, x):
            logits = self.linear_relu_seq(x)
            return F.softmax(logits, dim=-1)


    # Set model parameters
    lr = 0.01
    epochs = 40
    bs = 64
    train_losses, val_losses = [], []

    model = FullyConnectedNN()

    # Choose device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count > 1:
            model = nn.DataParallel(model)
    model.to(device)
    print(f"Using device: {device}")

    opt = optim.Adam(model.parameters(), lr=lr)

    # Wrap DataLoader iterator around our custom dataset(s)
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=bs*2, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=bs*2, shuffle=False)

    loss_func = nn.CrossEntropyLoss()

    # Calculating accuracy will not be our focus
    def accuracy(out, yb):
        preds = torch.argmax(out, dim=1) # To make it compatible with batches (where shape is [bs, nr_classes]), use dim=1
        return (preds == torch.argmax(yb, dim=1)).float().mean()


    example_out = model.predict_proba(feature)
    print(f"Example of model output: {example_out}, {example_out.shape}")

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
            for xb, yb in val_loader:
                yb_indices = yb.argmax(dim=1)
                xb, yb_indices = xb.to(device), yb_indices.to(device)

                pred = model(xb)
                loss = loss_func(pred, yb_indices)
                running_loss += loss.item() * yb_indices.size(0)
            
            val_loss = running_loss / len(val_loader.dataset)
            val_losses.append(val_loss)
        
        print(f"Epoch {epoch + 1}/{epochs} train loss: {train_loss}, validation loss: {val_loss}")

    plt.plot(train_losses, label="Training loss")
    plt.plot(val_losses, label="Validation loss")
    plt.legend()
    plt.title("Loss over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Cross Entropy Loss")
    timestamp = datetime.now().strftime("%m-%d_%H-%M-%S")
    plt.savefig(f"/faststorage/project/MutationAnalysis/Nimrod/results/figures/loss_curves_fc_{timestamp}.png")
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
    train_fc(args.data_version)