import os
from pathlib import Path
import tempfile
from functools import partial
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import ray
from ray import tune
from ray import train
from ray.air import session
from ray.tune import Checkpoint, get_checkpoint
from ray.tune.schedulers import ASHAScheduler
import ray.cloudpickle as pickle
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import argparse

from utils.load_splits import load_splits


class ModularModel(nn.Module):
    def __init__(self, l1, l2, use_avg_mut=False, use_bin_id_embed=False, use_bin_id_norm=False, num_bins=None, embed_dim=None):
        super().__init__()
        self.use_avg_mut = use_avg_mut
        self.use_bin_id_embed = use_bin_id_embed
        self.use_bin_id_norm = use_bin_id_norm

        if use_bin_id_embed:
            self.embedding = nn.Embedding(num_embeddings=num_bins, embedding_dim=embed_dim)
            self.flatten = nn.Flatten()
        
        self.linear_relu_seq = nn.Sequential(
            nn.LazyLinear(out_features=l1), nn.ReLU(),
            nn.LazyLinear(out_features=l2), nn.ReLU(),
            nn.LazyLinear(4)
        )
    
    def forward(self, local_context, avg_mut=None, bin_id=None, bin_id_norm=None):
        inputs = [local_context]

        if self.use_avg_mut:
            inputs.append(avg_mut)
        
        if self.use_bin_id_embed:
            assert self.embedding is not None, "Embedding layer not initialized. Did you forget to pass num_bins and embed_dim?"
            embedded = self.embedding(bin_id)
            flat = self.flatten(embedded)
            inputs.append(flat)
        
        if self.use_bin_id_norm:
            inputs.append(bin_id_norm)
        
        x = torch.cat(inputs, dim=-1)
        x = self.linear_relu_seq(x)
        return x
    
    def predict_proba(self, local_context, avg_mut=None, bin_id=None, bin_id_norm=None):
        logits = self.forward(local_context, avg_mut=avg_mut, bin_id=bin_id, bin_id_norm=bin_id_norm)
        return F.softmax(logits, dim=-1)


# Set up network architecture
class FullyConnectedNN(nn.Module):
    def __init__(self, l1=128, l2=64):
        super().__init__()
        self.linear_relu_seq = nn.Sequential(
            nn.LazyLinear(out_features=l1),
            nn.ReLU(),
            nn.Linear(in_features=l1, out_features=l2),
            nn.ReLU(),
            nn.Linear(in_features=l2, out_features=4)
        )
    
    def forward(self, x):
        x = self.linear_relu_seq(x)
        return x
    
    def predict_proba(self, x):
        logits = self.linear_relu_seq(x)
        return F.softmax(logits, dim=-1)


def train_model(config, data_version, epochs):
    # model = FullyConnectedNN(config["l1"], config["l2"])
    model = ModularModel(config["l1"], config["l2"], use_avg_mut=True)

    # X_local_train, y_train, X_local_val, y_val = load_splits(data_version, requested_splits=["train", "val"])
    X_local_train, X_avg_mut_1mb_train, y_train, X_local_val, X_avg_mut_1mb_val, y_val = load_splits(data_version, requested_splits=["train", "val"], bin_size="100kb", requested_features=["avg_mut"])

    train_dataset = TensorDataset(X_local_train, X_avg_mut_1mb_train, y_train)
    val_dataset = TensorDataset(X_local_val, X_avg_mut_1mb_val, y_val)

    # train_dataset = TensorDataset(X_local_train, y_train)
    # val_dataset = TensorDataset(X_local_val, y_val)

    with torch.no_grad():
        model(local_context=X_local_train[:2], avg_mut=X_avg_mut_1mb_train[:2])

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)
    print(f"Using device: {device}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=1e-4)

    train_losses = []
    val_losses = []

    checkpoint = get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / "data.pkl"
            with open(data_path, "rb") as fp:
                checkpoint_state = pickle.load(fp)
            start_epoch = checkpoint_state["epoch"]
            model.load_state_dict(checkpoint_state["model_state_dict"])
            optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
    else:
        start_epoch = 0


    train_loader = DataLoader(train_dataset, batch_size=int(config["batch_size"]), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=int(config["batch_size"]), shuffle=False)

    for epoch in range(start_epoch, epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        for xb_local, xb_region, yb in train_loader:
            xb_local, xb_region, yb = xb_local.to(device), xb_region.to(device), yb.to(device)

            pred = model(local_context=xb_local, avg_mut=xb_region)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * yb.size(0)

        # for xb, yb in train_loader:
        #     yb_indices = yb.argmax(dim=1)
        #     xb, yb_indices = xb.to(device), yb_indices.to(device)

        #     # Zero the parameter gradients
        #     optimizer.zero_grad()

        #     # Forward + backward + optimize
        #     pred = model(xb)
        #     loss = criterion(pred, yb_indices)
        #     loss.backward()
        #     optimizer.step()
            
        #     running_loss += loss.item() * yb_indices.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for xb_local, xb_region, yb in val_loader:
                xb_local, xb_region, yb = xb_local.to(device), xb_region.to(device), yb.to(device)
                
                pred = model(local_context=xb_local, avg_mut=xb_region)
                loss = criterion(pred, yb)
                running_loss += loss.item() * yb.size(0)
            
            val_loss = running_loss / len(val_loader.dataset)
            val_losses.append(val_loss)
        
        print(f"Epoch {epoch + 1}/{epochs} train loss: {train_loss}, validation loss: {val_loss}")

        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        }
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / "data.pkl"
            with open(data_path, "wb") as fp:
                pickle.dump(checkpoint_data, fp)
            
            checkpoint = Checkpoint.from_directory(checkpoint_dir)
            tune.report(
                {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss},
                checkpoint=checkpoint,
            )
    
    print("Finished training")


@ray.remote
def test_set_loss(model, data_version, device="cpu"):
    X_local_test, X_avg_mut_test, y_test = load_splits(
            data_version, requested_splits=["test"], bin_size="100kb", requested_features=["avg_mut"]
        )

    test_dataset = TensorDataset(X_local_test, X_avg_mut_test, y_test)

    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    criterion = nn.CrossEntropyLoss()

    test_losses = []

    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for xb_local, xb_region, yb in test_loader:
            xb_local, xb_region, yb = xb_local.to(device), xb_region.to(device), yb.to(device)
            
            pred = model(local_context=xb_local, avg_mut=xb_region)
            loss = criterion(pred, yb)
            val_running_loss += loss.item() * yb.size(0)
        
        test_loss = running_loss / len(test_loader.dataset)
        test_losses.append(test_loss)

        # for xb, yb in test_loader:
        #     yb_indices = yb.argmax(dim=1)
        #     xb, yb_indices = xb.to(device), yb_indices.to(device)

        #     pred = model(xb)
        #     loss = criterion(pred, yb_indices)
        #     running_loss += loss.item() * yb_indices.size(0)
        
        # test_loss = running_loss / len(test_loader.dataset)
        # test_losses.append(test_loss)


def main(num_samples=10, max_num_epochs=10, gpus_per_trial=1):
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

    ray.init()
    
    data_version = args.data_version
    config = {
        "l1": tune.choice([2 ** i for i in range(4, 10)]),
        "l2": tune.choice([2 ** i for i in range(4, 10)]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([4, 8, 16, 32, 64]),
    }
    scheduler = ASHAScheduler(
        metric="val_loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2,
    )
    
    result = tune.run(
        partial(train_model, data_version=data_version, epochs=max_num_epochs),
        resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        storage_path="/faststorage/project/MutationAnalysis/Nimrod/results/ray_tune/experiments"
    )

    analysis = tune.ExperimentAnalysis(result.experiment_path)

    best_trials = sorted(
        analysis.trial_dataframes.items(),
        key=lambda x: x[1]["val_loss"].min()
    )[:3]

    model_dir = f"/faststorage/project/MutationAnalysis/Nimrod/results/models/fc_ray"
    os.makedirs(model_dir, exist_ok=True)

    plots_dir = "/faststorage/project/MutationAnalysis/Nimrod/results/figures/fc"
    os.makedirs(plots_dir, exist_ok=True)

    for trial_name, df in best_trials:
        # Create loss curve figures of top 3 best performing trial
        plt.plot(df["epoch"], df["train_loss"], label="Training loss")
        plt.plot(df["epoch"], df["val_loss"], label="Validation loss")
        plt.legend()
        plt.title("Loss over epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Cross Entropy Loss")
        timestamp = datetime.now().strftime("%m-%d_%H-%M-%S")
        plt.savefig(f"{plots_dir}/loss_curves_fc_{trial_name}_{timestamp}.png")
        plt.close()

    best_trial = result.get_best_trial(
        metric="val_loss",
        mode="min",
        scope="last"
    )
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial validation loss: {best_trial.last_result['val_loss']}")

    # best_trained_model = FullyConnectedNN(best_trial.config["l1"], best_trial.config["l2"], best_trial.config["l3"])
    best_trained_model = ModularModel(best_trial.config["l1"], best_trial.config["l2"], use_avg_mut=True)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if gpus_per_trial > 1:
            model = nn.DataParallel(model)
    best_trained_model.to(device)
    print(f"Using device: {device}")

    best_checkpoint = result.get_best_checkpoint(trial=best_trial, metric="val_loss", mode="min")
    with best_checkpoint.as_directory() as checkpoint_dir:
        data_path = Path(checkpoint_dir) / "data.pkl"
        with open(data_path, "rb") as fp:
            best_checkpoint_data = pickle.load(fp)
        
        best_trained_model.load_state_dict(best_checkpoint_data["model_state_dict"])
        test_loss_future = test_set_loss.remote(best_trained_model, data_version, device)
        test_loss = ray.get(test_loss_future)
        print("Best trial test set loss: {:.4f}".format(test_loss))
        torch.save(best_trained_model.state_dict(), f"{model_dir}/best_model.pt")
    
    ray.shutdown()


if __name__ == "__main__":
    main(num_samples=10, max_num_epochs=30, gpus_per_trial=1)