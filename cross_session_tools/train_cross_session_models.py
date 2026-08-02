"""
Cross-Session Foundation Model Pre-Training Engine
===================================================

Pre-trains PyTorch ATCNet, PyTorch EEGNet, CSP+LDA, and Covariance Tangent Space LR pipelines
STRICTLY using Session 0 (Day 1) data for the Yang2025 multi-day BCI dataset.
"""

import os
import sys
import copy
import glob
import argparse
import joblib
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Ensure repository root is on sys.path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.append(repo_root)

from eegnet_tools.eegnet_model import EEGNet, Conv2dWithConstraint, LinearWithConstraint
from atcnet_tools.atcnet_model import ATCNet
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from mne.decoding import CSP
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace


def load_subject_eeg_session_data(dataset_name: str, subject_id: int, channels: list[str], session_filter: str = "0"):
    """Load EEG trial data for a specific subject filtered by session ('0' = Day 1, '1' = Day 2, '2' = Day 3)."""
    import moabb.datasets as mb_ds
    from moabb.paradigms import LeftRightImagery

    if not hasattr(mb_ds, dataset_name):
        raise ValueError(f"Dataset '{dataset_name}' not found in MOABB datasets module.")

    dataset_cls = getattr(mb_ds, dataset_name)
    dataset = dataset_cls()

    if channels:
        paradigm = LeftRightImagery(channels=channels)
    else:
        paradigm = LeftRightImagery()

    X, y, metadata = paradigm.get_data(dataset=dataset, subjects=[subject_id])
    
    # Filter by session
    session_str = str(session_filter)
    mask = (metadata['session'].astype(str) == session_str)
    
    if not np.any(mask):
        # Fallback if session format is different
        mask = (metadata['session'].astype(str) == f"session_{session_str}")
        if not np.any(mask):
            mask = np.ones(len(metadata), dtype=bool)

    X_session = X[mask]
    y_session = y[mask]
    return X_session, y_session


def build_classical_csp_lda_pipeline(num_channels: int):
    """Build CSP + LDA Pipeline."""
    n_comp = min(6, num_channels)
    return Pipeline([
        ('CSP', CSP(n_components=n_comp, reg=None, log=True, norm_trace=False)),
        ('LDA', LinearDiscriminantAnalysis())
    ])


def build_classical_cov_lr_pipeline():
    """Build Covariance + Tangent Space + Logistic Regression Pipeline."""
    return Pipeline([
        ('Cov', Covariances(estimator='lwf')),
        ('TS', TangentSpace(metric='riemann')),
        ('LR', LogisticRegression(max_iter=1000, solver='lbfgs'))
    ])


def train_pytorch_model(model: nn.Module, X_train: np.ndarray, y_train: list[str], epochs: int = 80, batch_size: int = 128, device_str: str = "cuda", lr: float = 1e-3):
    """Train a PyTorch model (EEGNet or ATCNet) on input numpy data."""
    device = torch.device(device_str if (device_str == "cuda" and torch.cuda.is_available()) else "cpu")
    if device.type == "cpu" and hasattr(torch, "set_num_threads"):
        torch.set_num_threads(min(12, os.cpu_count() or 8))

    unique_labels = sorted(list(set(y_train)))
    label_map = {lbl: idx for idx, lbl in enumerate(unique_labels)}
    y_indices = np.array([label_map[lbl] for lbl in y_train], dtype=np.int64)

    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    if X_tensor.dim() == 3:
        X_tensor = X_tensor.unsqueeze(1)
    y_tensor = torch.tensor(y_indices, dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

    model.eval()
    eval_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_preds = []
    with torch.no_grad():
        for bx, _ in eval_loader:
            bx = bx.to(device)
            logits = model(bx)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.append(preds)
    all_preds = np.concatenate(all_preds, axis=0)
    acc = float(np.mean(all_preds == y_indices))

    model = model.to("cpu")
    return model, acc


def main():
    parser = argparse.ArgumentParser(description="Pre-train Cross-Session Foundation Models (Session 0 Day 1 Training).")
    parser.add_argument("--dataset", "-d", default="Yang2025", help="Dataset name. Default: Yang2025")
    parser.add_argument("--model", "-m", default="all", choices=["all", "atcnet", "eegnet", "classical"], help="Model family to pre-train. Default: all")
    parser.add_argument("--epochs", "-e", type=int, default=80, help="Pre-training epochs. Default: 80")
    parser.add_argument("--batch-size", "-b", type=int, default=128, help="Batch size. Default: 128")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", help="Execution device. Default: auto")

    args = parser.parse_args()

    device_str = "cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())) else "cpu"
    selected_channels = ["C1", "C2", "C5", "C3", "C4", "C6", "FC3", "CP3", "FC4", "CP4", "Cz"]
    output_dir = os.path.join("trained_pipelines", "cross_session", args.dataset)
    os.makedirs(output_dir, exist_ok=True)

    print("================================================================================")
    print(" Cross-Session Foundation Model Pre-Training Engine (Session 0 / Day 1)")
    print("================================================================================")
    print(f" Dataset             : {args.dataset}")
    print(f" Training Session    : Session 0 (Day 1)")
    print(f" Model Family        : {args.model.upper()}")
    print(f" Epochs              : {args.epochs}")
    print(f" Batch Size          : {args.batch_size}")
    print(f" Device             : {device_str.upper()}")
    print(f" Output Directory    : {output_dir}")
    print("================================================================================\n")

    # Load all subjects' Session 0 data
    subject_ids = list(range(1, 52))
    session0_data = {}
    print("Loading Session 0 (Day 1) EEG data across subjects...")
    for sub in subject_ids:
        try:
            X_s0, y_s0 = load_subject_eeg_session_data(args.dataset, sub, selected_channels, session_filter="0")
            session0_data[sub] = (X_s0, y_s0)
            print(f"  Subject {sub:2d} | Session 0 Trials: {len(y_s0)} | Shape: {X_s0.shape}")
        except Exception as e:
            print(f"  Warning: Could not load Session 0 for subject {sub}: {e}")

    sample_sub = list(session0_data.keys())[0]
    sample_X, _ = session0_data[sample_sub]
    num_channels, num_samples = sample_X.shape[1], sample_X.shape[2]

    records = []

    # 1. Strategy C: Single-Subject Day 1 Models
    print("\n--- Strategy C: Pre-Training Single-Subject Day 1 Models ---")
    for sub, (X_s0, y_s0) in session0_data.items():
        sub_dir = os.path.join(output_dir, f"subject_{sub}")
        os.makedirs(sub_dir, exist_ok=True)

        # ATCNet Single Model
        if args.model in ["all", "atcnet"]:
            atc = ATCNet(n_classes=2, channels=num_channels, samples=num_samples)
            atc, acc_atc = train_pytorch_model(atc, X_s0, y_s0, epochs=args.epochs, batch_size=args.batch_size, device_str=device_str)
            torch.save(atc.state_dict(), os.path.join(sub_dir, "ATCNet_model.pt"))
            records.append({'Strategy': 'Single_Subject', 'Subject': sub, 'Model': 'ATCNet', 'PreTrain_Acc': acc_atc})

        # EEGNet Single Model
        if args.model in ["all", "eegnet"]:
            eeg = EEGNet(n_classes=2, channels=num_channels, samples=num_samples)
            eeg, acc_eeg = train_pytorch_model(eeg, X_s0, y_s0, epochs=args.epochs, batch_size=args.batch_size, device_str=device_str)
            torch.save(eeg.state_dict(), os.path.join(sub_dir, "EEGNet_model.pt"))
            records.append({'Strategy': 'Single_Subject', 'Subject': sub, 'Model': 'EEGNet', 'PreTrain_Acc': acc_eeg})

        # Classical CSP+LDA & Cov+LR Single Models
        if args.model in ["all", "classical"]:
            csp = build_classical_csp_lda_pipeline(num_channels)
            csp.fit(X_s0, y_s0)
            joblib.dump(csp, os.path.join(sub_dir, "CSP_LDA_model.pkl"))

            cov = build_classical_cov_lr_pipeline()
            cov.fit(X_s0, y_s0)
            joblib.dump(cov, os.path.join(sub_dir, "Cov_Tangent_Space_LR_model.pkl"))

    # 2. Strategy A: GNN Cluster-Mode Pooled Day 1 Models
    print("\n--- Strategy A: GNN Cluster-Mode Pooled Day 1 Models ---")
    cluster_csv = os.path.join("graph_results", "clustering", args.dataset, "subject_cluster_assignments.csv")
    if os.path.exists(cluster_csv):
        cdf = pd.read_csv(cluster_csv)
        cluster_col = "Donor_Cluster" if "Donor_Cluster" in cdf.columns else "Cluster"
        cluster_ids = sorted(cdf[cluster_col].unique())

        for c_id in cluster_ids:
            c_subs = sorted(list(set(cdf[cdf[cluster_col] == c_id]['Subject'])))
            c_subs = [s for s in c_subs if s in session0_data]

            if not c_subs:
                continue

            X_pool = np.concatenate([session0_data[s][0] for s in c_subs], axis=0)
            y_pool = np.concatenate([session0_data[s][1] for s in c_subs], axis=0)

            c_dir = os.path.join(output_dir, f"cluster_mode_{c_id}")
            os.makedirs(c_dir, exist_ok=True)

            print(f"Training Cluster Mode {c_id} ({len(c_subs)} subs, {len(y_pool)} Session 0 trials)...")

            if args.model in ["all", "atcnet"]:
                atc = ATCNet(n_classes=2, channels=num_channels, samples=num_samples)
                atc, acc_atc = train_pytorch_model(atc, X_pool, y_pool, epochs=args.epochs, batch_size=args.batch_size, device_str=device_str)
                torch.save(atc.state_dict(), os.path.join(c_dir, "ATCNet_cluster_model.pt"))
                records.append({'Strategy': 'Cluster_Pooled', 'Cluster': c_id, 'Model': 'ATCNet', 'PreTrain_Acc': acc_atc})

            if args.model in ["all", "eegnet"]:
                eeg = EEGNet(n_classes=2, channels=num_channels, samples=num_samples)
                eeg, acc_eeg = train_pytorch_model(eeg, X_pool, y_pool, epochs=args.epochs, batch_size=args.batch_size, device_str=device_str)
                torch.save(eeg.state_dict(), os.path.join(c_dir, "EEGNet_cluster_model.pt"))
                records.append({'Strategy': 'Cluster_Pooled', 'Cluster': c_id, 'Model': 'EEGNet', 'PreTrain_Acc': acc_eeg})

            if args.model in ["all", "classical"]:
                csp = build_classical_csp_lda_pipeline(num_channels)
                csp.fit(X_pool, y_pool)
                joblib.dump(csp, os.path.join(c_dir, "CSP_LDA_cluster_model.pkl"))

                cov = build_classical_cov_lr_pipeline()
                cov.fit(X_pool, y_pool)
                joblib.dump(cov, os.path.join(c_dir, "Cov_Tangent_Space_LR_cluster_model.pkl"))

    # 3. Strategy B: Submodular Top-5 Pooled Day 1 Models
    print("\n--- Strategy B: Submodular Top-5 Pooled Day 1 Models ---")
    sub_donors = [2, 14, 43, 80, 82]
    sub_donors = [s for s in sub_donors if s in session0_data]

    if sub_donors:
        X_sub = np.concatenate([session0_data[s][0] for s in sub_donors], axis=0)
        y_sub = np.concatenate([session0_data[s][1] for s in sub_donors], axis=0)

        sub_dir = os.path.join(output_dir, "submodular_top5_pooled")
        os.makedirs(sub_dir, exist_ok=True)

        print(f"Training Submodular Top-5 ({len(sub_donors)} subs, {len(y_sub)} Session 0 trials)...")

        if args.model in ["all", "atcnet"]:
            atc = ATCNet(n_classes=2, channels=num_channels, samples=num_samples)
            atc, acc_atc = train_pytorch_model(atc, X_sub, y_sub, epochs=args.epochs, batch_size=args.batch_size, device_str=device_str)
            torch.save(atc.state_dict(), os.path.join(sub_dir, "ATCNet_submodular_top5_model.pt"))
            records.append({'Strategy': 'Submodular_Top5', 'Model': 'ATCNet', 'PreTrain_Acc': acc_atc})

        if args.model in ["all", "eegnet"]:
            eeg = EEGNet(n_classes=2, channels=num_channels, samples=num_samples)
            eeg, acc_eeg = train_pytorch_model(eeg, X_sub, y_sub, epochs=args.epochs, batch_size=args.batch_size, device_str=device_str)
            torch.save(eeg.state_dict(), os.path.join(sub_dir, "EEGNet_submodular_top5_model.pt"))
            records.append({'Strategy': 'Submodular_Top5', 'Model': 'EEGNet', 'PreTrain_Acc': acc_eeg})

        if args.model in ["all", "classical"]:
            csp = build_classical_csp_lda_pipeline(num_channels)
            csp.fit(X_sub, y_sub)
            joblib.dump(csp, os.path.join(sub_dir, "CSP_LDA_submodular_top5_model.pkl"))

            cov = build_classical_cov_lr_pipeline()
            cov.fit(X_sub, y_sub)
            joblib.dump(cov, os.path.join(sub_dir, "Cov_Tangent_Space_LR_submodular_top5_model.pkl"))

    df_summary = pd.DataFrame(records)
    summary_path = os.path.join(output_dir, "pretraining_accuracies.csv")
    df_summary.to_csv(summary_path, index=False)

    print("\n================================================================================")
    print(f" Cross-Session Pre-Training Complete! Saved CSV to: {summary_path}")
    print("================================================================================\n")


if __name__ == "__main__":
    main()
