import torch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import colorcet as cc

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

    # for index in range(len(X_local_test)):
    for index in range(100):
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

    plt.rcParams.update({"font.family": "monospace"})

    # Saliency map as bar chart
    plt.figure(figsize=(12, 2))
    plt.bar(
        x=range(1, len(avg_saliency_per_position) + 1),
        height=avg_saliency_per_position,
        color=mpl.colormaps['cet_glasbey_dark'].colors[3]
    )
    plt.xlabel("Position", fontweight="bold")
    plt.gca().set_xticks(range(1, len(avg_saliency_per_position) + 1))
    plt.gca().set_xticklabels(range(1, len(avg_saliency_per_position) + 1))
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().grid(axis="y", linestyle="--", alpha=0.6, zorder=0)
    plt.tight_layout()
    plt.show()

    # Averaged saliency map as heatmap
    plt.figure(figsize=(len(avg_saliency_per_position), 1))
    sns.heatmap(np.expand_dims(avg_saliency_per_position, 1).T, cmap="hot")
    plt.gca().tick_params(left=False, bottom=False)
    plt.gca().set(xticklabels=[], yticklabels=[])
    # plt.xlabel("Position", fontweight="bold")
    plt.tight_layout()
    plt.show()

    # Nucleotide-specific saliency map as heatmap
    plt.figure(figsize=(12, 2))
    sns.heatmap(avg_saliency_matrix.T, cmap="hot", xticklabels=range(1, avg_saliency_matrix.shape[0] + 1), yticklabels=["A", "C", "G", "T"])
    plt.xlabel("Position", fontweight="bold")
    plt.tight_layout()
    plt.show()