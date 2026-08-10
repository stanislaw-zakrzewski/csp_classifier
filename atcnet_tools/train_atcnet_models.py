"""
ATCNet Model Pre-Training Engine
=================================

Pre-trains PyTorch ATCNet foundation models across:
1. GNN Cluster Mode Pooled Cohorts (Strategy A)
2. Submodular Top-5 Pooled Cohort (Strategy B)
3. Single Subject Datasets (Strategy C)

Includes built-in GPU acceleration, PyTorch CPU multi-threading, vector batching,
unique subject deduplication, and summary CSV upsert logic.
"""

import os
import sys
import glob
import copy
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from moabb.paradigms import MotorImagery
import moabb.datasets as mb_datasets

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from atcnet_tools.atcnet_model import ATCNet


def load_subject_eeg_data(dataset_name: str, subject_id: int, channels: list[str] = None) -> tuple[np.ndarray, list[str]]:
    """Loads EEG trials and labels for a single subject using MOABB MotorImagery paradigm."""
    if channels is None:
        channels = ["C1", "C2", "C5", "C3", "C4", "C6", "FC3", "CP3", "FC4", "CP4", "Cz"]

    dataset_cls = getattr(mb_datasets, dataset_name, None)
    if dataset_cls is None:
        raise ValueError(f"Dataset '{dataset_name}' not found in moabb.datasets.")

    dataset = dataset_cls()
    fmin, fmax = 8.0, 30.0

    paradigm = MotorImagery(
        fmin=fmin,
        fmax=fmax,
        channels=channels,
        events=["left_hand", "right_hand"],
        n_classes=2
    )

    X, labels, meta = paradigm.get_data(dataset=dataset, subjects=[subject_id])
    return X, labels


def load_pooled_cohort_data(dataset_name: str, subject_ids: list[int], channels: list[str] = None) -> tuple[np.ndarray, np.ndarray]:
    """Pools trial data across a list of donor subjects."""
    all_X, all_y = [], []

    for sub_id in sorted(list(set(subject_ids))):
        try:
            X_sub, labels_sub = load_subject_eeg_data(dataset_name, sub_id, channels=channels)
            label_map = {lbl: idx for idx, lbl in enumerate(sorted(list(set(labels_sub))))}
            y_sub = np.array([label_map[lbl] for lbl in labels_sub])

            all_X.append(X_sub)
            all_y.append(y_sub)
        except Exception as e:
            print(f"  Warning: Could not load data for subject {sub_id}: {e}")

    if not all_X:
        raise ValueError(f"No data could be loaded for pooled cohort: {subject_ids}")

    X_pooled = np.concatenate(all_X, axis=0)
    y_pooled = np.concatenate(all_y, axis=0)
    return X_pooled, y_pooled


def train_pytorch_atcnet(
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 80,
    lr: float = 1e-3,
    batch_size: int = 128,
    device: str = "auto"
) -> ATCNet:
    """Trains a PyTorch ATCNet model on pooled or single-subject EEG trial data."""
    if device == "auto":
        device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_obj = torch.device(device)

    if device_obj.type == "cpu" and hasattr(torch, "set_num_threads"):
        torch.set_num_threads(min(12, os.cpu_count() or 8))

    _, num_channels, num_samples = X_train.shape
    model = ATCNet(n_classes=2, channels=num_channels, samples=num_samples).to(device_obj)

    X_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    y_tensor = torch.tensor(y_train, dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=(device_obj.type == "cuda"))

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for ep in range(epochs):
        for bx, by in loader:
            bx, by = bx.to(device_obj), by.to(device_obj)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

    return model


def evaluate_atcnet_acc(model: ATCNet, X: np.ndarray, y: np.ndarray, batch_size: int = 256, device: str = "auto") -> float:
    """Evaluates classification accuracy of a PyTorch ATCNet model using DataLoader batching."""
    if device == "auto":
        device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_obj = torch.device(device)

    model.eval()
    model.to(device_obj)

    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    y_tensor = torch.tensor(y, dtype=torch.long)

    loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=batch_size, shuffle=False)

    correct = 0
    total = 0

    with torch.no_grad():
        for bx, by in loader:
            bx, by = bx.to(device_obj), by.to(device_obj)
            logits = model(bx)
            preds = torch.argmax(logits, dim=1)
            correct += int((preds == by).sum().item())
            total += len(by)

    return correct / max(1, total)


def upsert_summary_csv(csv_path: str, new_df: pd.DataFrame):
    """Merges new pre-training accuracy records into CSV without overwriting existing data."""
    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        combined.drop_duplicates(subset=['Pipeline', 'Subject_Or_Cohort'], keep='last', inplace=True)
        combined.to_csv(csv_path, index=False)
    else:
        new_df.to_csv(csv_path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Pre-train PyTorch ATCNet models for BCI Transfer Learning.")
    parser.add_argument("--dataset", "-d", default="Dreyer2023", help="Dataset name. Default: Dreyer2023")
    parser.add_argument("--mode", "-m", choices=["single", "cluster", "submodular", "all"], default="all", help="Training mode. Default: all")
    parser.add_argument("--epochs", "-e", type=int, default=80, help="Training epochs. Default: 80")
    parser.add_argument("--batch-size", "-b", type=int, default=128, help="Batch size. Default: 128")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", help="Execution device. Default: auto")
    parser.add_argument("--output-dir", "-o", default=None, help="Output directory for checkpoints.")

    args = parser.parse_args()

    selected_channels = ["C1", "C2", "C5", "C3", "C4", "C6", "FC3", "CP3", "FC4", "CP4", "Cz"]
    output_dir = args.output_dir or os.path.join("trained_pipelines", "atcnet", args.dataset)
    os.makedirs(output_dir, exist_ok=True)

    summary_csv = os.path.join(output_dir, "atcnet_pretraining_accuracies.csv")
    records = []

    device_str = "cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())) else "cpu"

    print("================================================================================")
    print(f" PyTorch ATCNet Foundation Pre-Training Engine")
    print("================================================================================")
    print(f" Dataset     : {args.dataset}")
    print(f" Mode        : {args.mode}")
    print(f" Epochs      : {args.epochs}")
    print(f" Batch Size  : {args.batch_size}")
    print(f" Device      : {device_str.upper()}")
    print(f" Output Dir  : {output_dir}")
    print("================================================================================\n")

    # Load subjects
    tp_dir = os.path.join("trained_pipelines", args.dataset)
    if os.path.exists(tp_dir):
        sub_dirs = [d for d in os.listdir(tp_dir) if d.startswith("subject_") and os.path.isdir(os.path.join(tp_dir, d))]
        subject_ids = sorted([int(d.replace("subject_", "")) for d in sub_dirs])
    else:
        subject_ids = list(range(1, 88))

    # Pre-load subject data dict
    subject_data = {}
    print("Pre-loading single-subject EEG data...")
    for s_id in subject_ids:
        try:
            X_sub, labels_sub = load_subject_eeg_data(args.dataset, s_id, channels=selected_channels)
            label_map = {lbl: idx for idx, lbl in enumerate(sorted(list(set(labels_sub))))}
            y_sub = np.array([label_map[lbl] for lbl in labels_sub])
            subject_data[s_id] = (X_sub, y_sub)
        except Exception as e:
            print(f"  Warning: Skipped subject {s_id}: {e}")

    # Mode 1: Cluster Mode (Strategy A)
    if args.mode in ["cluster", "all"]:
        print("\n--- Training Strategy A: GNN Cluster-Mode Pooled ATCNets ---")
        cluster_csv = os.path.join("graph_results", "clustering", args.dataset, "subject_cluster_assignments.csv")

        if os.path.exists(cluster_csv):
            cdf = pd.read_csv(cluster_csv)
            cluster_col = "Donor_Cluster" if "Donor_Cluster" in cdf.columns else "Cluster"
            cluster_ids = sorted(cdf[cluster_col].unique())

            for c_id in cluster_ids:
                cluster_subs = sorted(list(set([int(s) for s in cdf[cdf[cluster_col] == c_id]['Subject'] if int(s) in subject_data])))
                if not cluster_subs:
                    continue

                X_c = np.concatenate([subject_data[s][0] for s in cluster_subs], axis=0)
                y_c = np.concatenate([subject_data[s][1] for s in cluster_subs], axis=0)

                print(f"Training ATCNet Cluster Mode {c_id} ({len(cluster_subs)} unique subs, {len(y_c)} trials)...")
                model_c = train_pytorch_atcnet(X_c, y_c, epochs=args.epochs, batch_size=args.batch_size, device=args.device)
                acc_c = evaluate_atcnet_acc(model_c, X_c, y_c, device=args.device)

                c_dir = os.path.join(output_dir, f"cluster_mode_{c_id}")
                os.makedirs(c_dir, exist_ok=True)
                torch.save(model_c.state_dict(), os.path.join(c_dir, "ATCNet_cluster_model.pt"))

                records.append({
                    'Pipeline': 'ATCNet_Cluster_Pooled',
                    'Subject_Or_Cohort': f'Cluster_{c_id}',
                    'Accuracy': acc_c,
                    'Epochs': args.epochs,
                    'Num_Subjects': len(cluster_subs),
                    'Total_Trials': len(y_c)
                })
                print(f"  Cluster Mode {c_id} Pre-Training Acc: {acc_c * 100:.2f}%")

    # Mode 2: Submodular Top-5 (Strategy B)
    if args.mode in ["submodular", "all"]:
        print("\n--- Training Strategy B: Submodular Top-5 Pooled ATCNet ---")
        sub_top5 = [2, 14, 43, 80, 82]
        available_top5 = [s for s in sub_top5 if s in subject_data]

        if available_top5:
            X_sub5 = np.concatenate([subject_data[s][0] for s in available_top5], axis=0)
            y_sub5 = np.concatenate([subject_data[s][1] for s in available_top5], axis=0)

            print(f"Training Submodular Top-5 ATCNet ({len(available_top5)} subs, {len(y_sub5)} trials)...")
            model_sub5 = train_pytorch_atcnet(X_sub5, y_sub5, epochs=args.epochs, batch_size=args.batch_size, device=args.device)
            acc_sub5 = evaluate_atcnet_acc(model_sub5, X_sub5, y_sub5, device=args.device)

            sub_dir = os.path.join(output_dir, "submodular_top5_pooled")
            os.makedirs(sub_dir, exist_ok=True)
            torch.save(model_sub5.state_dict(), os.path.join(sub_dir, "ATCNet_submodular_top5_model.pt"))

            records.append({
                'Pipeline': 'ATCNet_Submodular_Top5_Pooled',
                'Subject_Or_Cohort': 'Submodular_Top5',
                'Accuracy': acc_sub5,
                'Epochs': args.epochs,
                'Num_Subjects': len(available_top5),
                'Total_Trials': len(y_sub5)
            })
            print(f"  Submodular Top-5 ATCNet Pre-Training Acc: {acc_sub5 * 100:.2f}%")

    # Mode 3: Single Subjects (Strategy C - 80/20 Train/Test Split)
    if args.mode in ["single", "all"]:
        from sklearn.model_selection import train_test_split
        print("\n--- Training Strategy C: Single-Subject ATCNets (80/20 Train/Test Split) ---")
        for s_id, (X_s, y_s) in subject_data.items():
            if len(X_s) >= 10:
                X_tr, X_te, y_tr, y_te = train_test_split(X_s, y_s, test_size=0.20, random_state=42, stratify=y_s)
            else:
                X_tr, X_te, y_tr, y_te = X_s, X_s, y_s, y_s

            model_s = train_pytorch_atcnet(X_tr, y_tr, epochs=args.epochs, batch_size=args.batch_size, device=args.device)
            acc_s = evaluate_atcnet_acc(model_s, X_te, y_te, device=args.device)

            s_dir = os.path.join(output_dir, f"subject_{s_id}")
            os.makedirs(s_dir, exist_ok=True)
            torch.save(model_s.state_dict(), os.path.join(s_dir, "ATCNet_model.pt"))
            np.savez(os.path.join(s_dir, "single_subject_test_split.npz"), X_test=X_te, y_test=y_te)

            records.append({
                'Pipeline': 'ATCNet_Single_Subject',
                'Subject_Or_Cohort': f'Subject_{s_id}',
                'Accuracy': acc_s,
                'Epochs': args.epochs,
                'Num_Subjects': 1,
                'Total_Trials': len(y_s),
                'Train_Trials': len(y_tr),
                'Test_Trials': len(y_te)
            })
            print(f"  Subject {s_id:2d} | Train: {len(y_tr):3d} | Test: {len(y_te):3d} | Out-of-Sample Test Acc: {acc_s * 100:.2f}%")

    if records:
        df_new = pd.DataFrame(records)
        upsert_summary_csv(summary_csv, df_new)
        print(f"\n================================================================================")
        print(f" Saved Pre-Training Summary CSV: {summary_csv}")
        print(f"================================================================================\n")


if __name__ == "__main__":
    main()
