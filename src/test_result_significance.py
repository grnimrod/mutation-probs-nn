import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from utils.load_splits import load_splits
from model_definitions import ModularModel


def get_loss_per_sample(data_version):
    """
    Get a np array of loss values for each prediction over the data
    predicting with the specified pre-trained model. Model choice
    limited to ones trained on local context + avg mut, 100kb.
    """

    # Load in test data
    X_local_test, X_avg_mut_test, y_test = load_splits(data_version=data_version, requested_splits=["test"], bin_size="100kb", requested_features=["avg_mut"])

    test_dataset = TensorDataset(X_local_test, X_avg_mut_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Instantiate model, load in saved weights
    model = ModularModel(use_avg_mut=True)
    model.load_state_dict(torch.load(f"/faststorage/project/MutationAnalysis/Nimrod/results/models/combined/{data_version}/avgmut/model_100kb.pth"))

    losses_per_sample = []

    criterion = nn.CrossEntropyLoss(reduction="none")

    model.eval()
    with torch.no_grad():
        for xb_local, xb_avgmut, yb in test_loader:
            pred = model(local_context=xb_local, avg_mut=xb_avgmut)
            loss = criterion(pred, yb)

            losses_per_sample.append(loss.item())
    
    return np.array(losses_per_sample)