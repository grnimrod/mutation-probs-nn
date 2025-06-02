import os
import json
from filelock import FileLock
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import log_loss
import argparse

from utils.load_splits import load_splits
from utils.decode_kmer import decode_kmer
from model_definitions import KmerCountsModel, ModularModel


def eval_benchmark_fc(data_version):
    """
    Evaluate performance of counts-based benchmark and fully
    connected model trained on local context. Save results to
    shared json file for further plotting.
    """

    MODEL_BASE_PATH = f"/faststorage/project/MutationAnalysis/Nimrod/results/models"

    RESULTS_PATH = "/faststorage/project/MutationAnalysis/Nimrod/results/eval_files/log_losses_benchmark_fc_models.json"
    LOCK_PATH = RESULTS_PATH + ".lock"

    losses = {}

    X_local_test, y_test = load_splits(data_version, requested_splits=["test"])

    # Evaluate benchmark model
    X_local_test_decoded = decode_kmer(X_local_test)
    y_test_decoded = decode_kmer(y_test)

    model_benchmark = KmerCountsModel.load(f"{MODEL_BASE_PATH}/counts_benchmark/{data_version}/model.pkl")
    loss_benchmark = model_benchmark.evaluate(X_local_test_decoded, y_test_decoded)

    # Evaluate fc model
    model_nn = ModularModel()
    model_nn.load_state_dict(torch.load(f"{MODEL_BASE_PATH}/fc/{data_version}/model.pth"))

    test_dataset = TensorDataset(X_local_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    test_probs = []
    test_targets = []

    model_nn.eval()
    with torch.no_grad():
        for xb, yb in test_loader:
            probs = model_nn.predict_proba(xb)

            test_probs.append(probs.cpu())
            test_targets.append(yb.cpu())

    test_probs = torch.cat(test_probs).numpy()
    test_targets = torch.cat(test_targets).numpy()

    loss_nn = log_loss(test_targets, test_probs)

    with FileLock(LOCK_PATH):
        if os.path.exists(RESULTS_PATH):
            with open(RESULTS_PATH, "r") as f:
                losses = json.load(f)
        else:
            losses = {}

        if "benchmark" not in losses:
            losses["benchmark"] = {}
        if data_version not in losses["benchmark"]:
            losses["benchmark"][data_version] = loss_benchmark
        
        if "fc" not in losses:
            losses["fc"] = {}
        if data_version not in losses["fc"]:
            losses["fc"][data_version] = loss_nn

        with open(RESULTS_PATH, "w") as f:
            json.dump(losses, f, indent=2)


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
    eval_benchmark_fc(args.data_version)