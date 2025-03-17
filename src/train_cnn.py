import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


X_train = np.load("./../data/splits/X_train.npy")
y_train = np.load("./../data/splits/y_train.npy")
X_val = np.load("./../data/splits/X_val.npy")
y_val = np.load("./../data/splits/y_val.npy")
X_test = np.load("./../data/splits/X_test.npy")
y_test = np.load("./../data/splits/y_test.npy")

X_train = torch.as_tensor(X_train, dtype=torch.float32)
y_train = torch.as_tensor(y_train, dtype=torch.float32)
X_val = torch.as_tensor(X_val, dtype=torch.float32)
y_val = torch.as_tensor(y_val, dtype=torch.float32)
X_test = torch.as_tensor(X_test, dtype=torch.float32)
y_test = torch.as_tensor(y_test, dtype=torch.float32)

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
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=6, kernel_size=7), # Should kernel size be 60?
        self.conv2 = nn.Conv1d(in_channels=6, out_channels=, kernel_size=5),
        self.fc1 = nn.Linear(in_features=256, out_features=128),
        self.fc2 = nn.Linear(in_features=, out_features=)
    
    def forward(self, x):
        x = F.max_pool1d(F.relu(self.conv1(x)), 2)
        x = F.max_pool1d(F.relu(self.conv2(x)), 2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


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

def accuracy(out, yb):
    preds = torch.argmax(out, dim=1) # To make it compatible with batches (where shape is [bs, nr_classes]), use dim=1
    return (preds == torch.argmax(yb, dim=1)).float().mean()


example_out = model(feature)
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