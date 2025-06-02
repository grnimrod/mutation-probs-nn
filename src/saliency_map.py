import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utils.load_splits import load_splits
from model_definitions import ModularModel


def obtain_saliency_map(data_version):
    """
    Create average and nucleotide-specific saliency maps.
    """

    X_local_test, _ = load_splits(data_version, requested_splits=["test"])

    # Need to conduct it on the simple FC model trained only on the local context
    model = ModularModel()
    model.load_state_dict(torch.load(f"/faststorage/project/MutationAnalysis/Nimrod/results/models/fc/{data_version}/model.pth"))

    model.eval()

    avg_saliency_per_position = []
    avg_saliency_matrix = []

    for index in range(len(X_local_test)):
        X = X_local_test[index].unsqueeze(0).clone().detach()
        
        X.requires_grad_()

        scores = model(X)
        
        score_max_index = scores.argmax()
        score_max = scores[0, score_max_index]
        
        model.zero_grad()
        score_max.backward()

        saliency = X.grad.abs()

        k = X.shape[1] // 4
        saliency = saliency.view(k, 4)

        saliency_per_position = saliency.sum(dim=1).cpu().numpy() # Sums across nucleotides to acquire info on which positions are important irrespective of what the nucleotide is
        avg_saliency_per_position.append(saliency_per_position)

        saliency_matrix = saliency.cpu().numpy() # Keeps matrix, could be visualized as heatmap with x for k-mer position and y for nucleotide
        avg_saliency_matrix.append(saliency_matrix)
    
    avg_saliency_per_position = np.mean(avg_saliency_per_position, axis=0)

    avg_saliency_matrix = np.mean(avg_saliency_matrix, axis=0)
    print(f"Avg of saliency matrix: {avg_saliency_matrix}")

    plt.figure(figsize=(12, 2))
    plt.bar(x=range(len(avg_saliency_per_position)), height=avg_saliency_per_position)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 2))
    sns.heatmap(avg_saliency_matrix.T, cmap="hot", yticklabels=["A", "C", "G", "T"])
    plt.tight_layout()
    plt.show()