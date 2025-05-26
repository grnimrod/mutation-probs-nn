import os
import json
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import log_loss

from utils.load_splits import load_splits
from model_definitions import ModularModel


# Load in data, load in trained models
DATA_VERSION = ["3fC", "5fC", "7fC", "9fC", "11fC", "13fC", "15fC"]
BIN_SIZES = ["100kb", "500kb", "1mb"]
MODEL_VARIANTS = {
    "avgmut": {"use_avg_mut": True, "use_bin_id_embed": False, "use_bin_id_norm": False},
    "avgmut_binnorm": {"use_avg_mut": True, "use_bin_id_embed": False, "use_bin_id_norm": True},
    "avgmut_binid_binnorm": {"use_avg_mut": True, "use_bin_id_embed": True, "use_bin_id_norm": True},
    "binid": {"use_avg_mut": False, "use_bin_id_embed": True, "use_bin_id_norm": False},
    "binnorm": {"use_avg_mut": False, "use_bin_id_embed": False, "use_bin_id_norm": True}
}

MODEL_BASE_PATH = f"/faststorage/project/MutationAnalysis/Nimrod/results/models/combined"

losses = {}

for version in DATA_VERSION:
    for bin_size in BIN_SIZES:
        # Load data (shared by models that use different features)
        _, _, X_bin_id_train, _, \
        _, _, X_bin_id_val, _, \
        X_local_test, X_avg_mut_test, X_bin_id_test, y_test = load_splits(
            version, requested_splits=["train", "val", "test"], bin_size=bin_size, requested_features=["avg_mut", "bin_id"]
        )

        all_bins = np.concatenate((X_bin_id_train, X_bin_id_val, X_bin_id_test), axis=0)
        unique_bins = set(all_bins)
        n_bins = len(unique_bins)

        # Normalize integer-encoded bin IDs
        X_bin_id_normalized_test = X_bin_id_test.float() / n_bins

        for variant_name, config in MODEL_VARIANTS.items():
            print(f"Evaluating model {variant_name}")
            inputs = [X_local_test]

            if config["use_avg_mut"]:
                X_avg_mut_test_unsqueezed = X_avg_mut_test.unsqueeze(1)
                inputs.append(X_avg_mut_test_unsqueezed)
            if config["use_bin_id_embed"]:
                X_bin_id_embed_test_unsqueezed = X_bin_id_test.unsqueeze(1)
                inputs.append(X_bin_id_embed_test_unsqueezed)
            if config["use_bin_id_norm"]:
                X_bin_id_normalized_test_unsqueezed = X_bin_id_normalized_test.unsqueeze(1)
                inputs.append(X_bin_id_normalized_test_unsqueezed)
            
            test_dataset = TensorDataset(*inputs, y_test)
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

            model_kwargs = dict(config)
            if config.get("use_bin_id_embed"):
                model_kwargs["num_bins"] = n_bins
                model_kwargs["embed_dim"] = 64

            print(model_kwargs)
            
            model = ModularModel(**model_kwargs)
            model.load_state_dict(torch.load(f"{MODEL_BASE_PATH}/{version}/{variant_name}/model_{bin_size}.pth"))

            test_probs = []
            test_targets = []

            with torch.no_grad():
                for batch in test_loader:
                    *features, label = batch

                    idx = 0
                    local_context = features[idx]
                    idx += 1

                    if config["use_avg_mut"]:
                        avg_mut = features[idx]
                        idx += 1
                    else:
                        avg_mut = None
                    
                    if config["use_bin_id_embed"]:
                        bin_id = features[idx]
                        idx += 1
                    else:
                        bin_id = None
                    
                    if config["use_bin_id_norm"]:
                        bin_id_norm = features[idx]
                        idx += 1
                    else:
                        bin_id_norm = None

                    probs = model.predict_proba(local_context=local_context, avg_mut=avg_mut, bin_id=bin_id, bin_id_norm=bin_id_norm)

                    test_probs.append(probs.cpu())
                    test_targets.append(label.cpu())

            test_probs = torch.cat(test_probs).numpy()
            test_targets = torch.cat(test_targets).numpy()

            loss = log_loss(test_targets, test_probs)

            if variant_name not in losses:
                losses[variant_name] = {}
            if version not in losses[variant_name]:
                losses[variant_name][version] = {}
            losses[variant_name][version][bin_size] = loss

with open("/faststorage/project/MutationAnalysis/Nimrod/results/log_losses_combined_models_C.json", "w") as f:
    json.dump(losses, f, indent=2)