"""
GNN Cluster Model Selection Validation Engine
==============================================

Empirically benchmarks Option 1 (Zero-Shot GNN Distance), Option 2 (Soft Cluster Mixture Ensemble),
Option 3 (First 8-Trial Confidence Selection), and Oracle Control on unseen target subjects.
"""

import os
import sys
import glob
import argparse
import joblib
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from scipy.spatial.distance import cdist

# Ensure repository root is on sys.path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.append(repo_root)

from eegnet_tools.eegnet_model import EEGNet
from atcnet_tools.atcnet_model import ATCNet
from eegnet_tools.train_eegnet_models import load_subject_eeg_data
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace


def compute_tangent_features(X: np.ndarray) -> np.ndarray:
    """Compute mean Riemannian tangent space feature vector for input EEG trials."""
    cov = Covariances(estimator='lwf').fit_transform(X)
    ts = TangentSpace(metric='riemann').fit_transform(cov)
    return np.mean(ts, axis=0)


def predict_pytorch_confidence(model: nn.Module, X: np.ndarray, device_obj: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Predict labels and max softmax confidence for input EEG trials."""
    model.eval()
    model.to(device_obj)
    with torch.no_grad():
        tx = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(device_obj)
        logits = model(tx)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
        confidences = np.max(probs, axis=1)
    return preds, confidences


def main():
    parser = argparse.ArgumentParser(description="Validate GNN Cluster Model Selection Strategies.")
    parser.add_argument("--dataset", "-d", default="Dreyer2023", help="Dataset name. Default: Dreyer2023")
    parser.add_argument("--model-type", choices=["ATCNet", "EEGNet"], default="ATCNet", help="Model family to evaluate. Default: ATCNet")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", help="Execution device. Default: auto")

    args = parser.parse_args()

    device_str = "cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())) else "cpu"
    device_obj = torch.device(device_str)

    selected_channels = ["C1", "C2", "C5", "C3", "C4", "C6", "FC3", "CP3", "FC4", "CP4", "Cz"]
    out_dir = os.path.join("graph_results", "cluster_selection", args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    print("================================================================================")
    print(" GNN Cluster Model Selection Validation Engine")
    print("================================================================================")
    print(f" Dataset             : {args.dataset}")
    print(f" Evaluated Model     : {args.model_type}")
    print(f" Device             : {device_str.upper()}")
    print(f" Output Directory    : {out_dir}")
    print("================================================================================\n")

    # Load cluster assignments and centroids
    cluster_csv = os.path.join("graph_results", "clustering", args.dataset, "subject_cluster_assignments.csv")
    if not os.path.exists(cluster_csv):
        raise FileNotFoundError(f"Cluster assignments CSV '{cluster_csv}' not found. Run GNN clustering first.")

    cdf = pd.read_csv(cluster_csv)
    cluster_col = "Donor_Cluster" if "Donor_Cluster" in cdf.columns else "Cluster"
    cluster_ids = [int(c) for c in sorted(cdf[cluster_col].unique())]
    all_subjects = [int(s) for s in sorted(cdf['Subject'].unique())]

    # Load cluster models
    models_dir = os.path.join("trained_pipelines", args.model_type.lower(), args.dataset)
    cluster_models = {}

    sample_X, _ = load_subject_eeg_data(args.dataset, int(all_subjects[0]), channels=selected_channels)
    num_channels, num_samples = sample_X.shape[1], sample_X.shape[2]

    for c_id in cluster_ids:
        c_dir = os.path.join(models_dir, f"cluster_mode_{c_id}")
        pt_file = os.path.join(c_dir, f"{args.model_type}_cluster_model.pt")
        if os.path.exists(pt_file):
            if args.model_type == "ATCNet":
                net = ATCNet(n_classes=2, channels=num_channels, samples=num_samples)
            else:
                net = EEGNet(n_classes=2, channels=num_channels, samples=num_samples)
            net.load_state_dict(torch.load(pt_file, map_location="cpu", weights_only=True))
            cluster_models[c_id] = net

    print(f"Loaded {len(cluster_models)} pre-trained cluster models: Cluster IDs {list(cluster_models.keys())}")

    # Compute cluster centroid tangent features
    cluster_centroids = {}
    for c_id in cluster_ids:
        c_subs = [int(s) for s in sorted(list(cdf[cdf[cluster_col] == c_id]['Subject']))]
        c_feats = []
        for s in c_subs[:10]:
            try:
                Xs, _ = load_subject_eeg_data(args.dataset, int(s), channels=selected_channels)
                c_feats.append(compute_tangent_features(Xs[:20]))
            except Exception:
                pass
        if c_feats:
            cluster_centroids[c_id] = np.mean(c_feats, axis=0)

    results = []

    for sub in all_subjects:
        try:
            X_sub, y_sub = load_subject_eeg_data(args.dataset, int(sub), channels=selected_channels)
            if X_sub.shape[2] > num_samples:
                X_sub = X_sub[:, :, :num_samples]
            elif X_sub.shape[2] < num_samples:
                pad_w = num_samples - X_sub.shape[2]
                X_sub = np.pad(X_sub, ((0,0), (0,0), (0, pad_w)), mode='edge')

            unique_labels = sorted(list(set(y_sub)))
            label_map = {lbl: idx for idx, lbl in enumerate(unique_labels)}
            y_indices = np.array([label_map[lbl] for lbl in y_sub], dtype=np.int64)

            # Evaluate performance of ALL cluster models to find Oracle
            cluster_accuracies = {}
            cluster_confidences_8 = {}
            all_cluster_probs = []

            # Option 3 setup: First 8 trials
            X_8 = X_sub[:8]

            for c_id, model in cluster_models.items():
                preds_full, _ = predict_pytorch_confidence(model, X_sub, device_obj)
                acc = float(np.mean(preds_full == y_indices))
                cluster_accuracies[c_id] = acc

                _, conf_8 = predict_pytorch_confidence(model, X_8, device_obj)
                cluster_confidences_8[c_id] = float(np.mean(conf_8))

                # For soft ensemble
                with torch.no_grad():
                    tx = torch.tensor(X_sub, dtype=torch.float32).unsqueeze(1).to(device_obj)
                    probs = torch.softmax(model(tx), dim=1).cpu().numpy()
                    all_cluster_probs.append((c_id, probs))

            # Oracle selection
            c_oracle = max(cluster_accuracies, key=cluster_accuracies.get)
            acc_oracle = cluster_accuracies[c_oracle]

            # Option 1: Zero-Shot GNN Distance Selection
            sub_feat = compute_tangent_features(X_sub[:10])
            distances = {c_id: float(np.linalg.norm(sub_feat - centroid)) for c_id, centroid in cluster_centroids.items()}
            c_opt1 = min(distances, key=distances.get)
            acc_opt1 = cluster_accuracies[c_opt1]

            # Option 2: Soft Cluster Mixture Ensemble
            dist_arr = np.array([distances[c_id] for c_id in cluster_ids])
            weights = (1.0 / (1.0 + dist_arr))
            weights = weights / np.sum(weights)

            ensemble_probs = np.zeros_like(all_cluster_probs[0][1])
            for idx, (c_id, probs) in enumerate(all_cluster_probs):
                ensemble_probs += weights[idx] * probs
            preds_opt2 = np.argmax(ensemble_probs, axis=1)
            acc_opt2 = float(np.mean(preds_opt2 == y_indices))

            # Option 3: First 8-Trial Confidence Selection
            c_opt3 = max(cluster_confidences_8, key=cluster_confidences_8.get)
            acc_opt3 = cluster_accuracies[c_opt3]

            results.append({
                'Subject': sub,
                'Oracle_Cluster': c_oracle,
                'Oracle_Acc': acc_oracle,
                'Option1_Cluster': c_opt1,
                'Option1_Acc': acc_opt1,
                'Option1_Match': (c_opt1 == c_oracle),
                'Option2_Acc': acc_opt2,
                'Option3_Cluster': c_opt3,
                'Option3_Acc': acc_opt3,
                'Option3_Match': (c_opt3 == c_oracle)
            })

            print(f" Subject {sub:2d} | Oracle: Cluster {c_oracle} ({acc_oracle:.4f}) | Opt1: Cluster {c_opt1} ({acc_opt1:.4f}) | Opt2 Ensemble: {acc_opt2:.4f} | Opt3: Cluster {c_opt3} ({acc_opt3:.4f})")

        except Exception as e:
            print(f"  Warning: Validation failed for subject {sub}: {e}")

    df_res = pd.DataFrame(results)
    res_path = os.path.join(out_dir, "selection_validation_results.csv")
    df_res.to_csv(res_path, index=False)

    print("\n================================================================================")
    print(f" Selection Validation Complete! Saved CSV to: {res_path}")
    print("================================================================================\n")


if __name__ == "__main__":
    main()
