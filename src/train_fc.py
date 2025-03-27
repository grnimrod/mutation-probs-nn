import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


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
plt.savefig(f"./../figures/loss_curves_fc_{timestamp}.png")
plt.close()