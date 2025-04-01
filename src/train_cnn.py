import os
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import argparse


def train_cnn(data_version):
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

    expl_files = [X_train, X_val, X_test]
    for file in expl_files:
        print(file.shape)

    expl_files = [file.view(file.size(0), 4, 15) for file in expl_files]
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
            self.conv1 = nn.Conv1d(in_channels=4, out_channels=32, kernel_size=3)
            self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3)
            self.fc1 = nn.Linear(in_features=32 * (15 - 2 - 2), out_features=128)
            self.fc2 = nn.Linear(in_features=128, out_features=4)
        
        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)

            x = x.view(x.size(0), -1)

            x = self.fc1(x)
            x = F.relu(x)
            x = self.fc2(x)
            return x
        
        def predict_proba(self, x):
            logits = self.forward(x)
            return F.softmax(logits, dim=-1)


    # Set model parameters
    lr = 0.01
    epochs = 10
    bs = 64
    train_losses, val_losses = [], []

    model = CustomCNN().to(device)
    opt = optim.Adam(model.parameters(), lr=lr)

    # Wrap DataLoader iterator around our custom dataset(s)
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=bs*2, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=bs*2, shuffle=False)

    loss_func = nn.CrossEntropyLoss()

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
            for xb, yb in test_loader:
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
    plt.savefig("./../loss_curves_fc.png")
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