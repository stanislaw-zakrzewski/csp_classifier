"""
EEGNet & Pooled Models Pre-Training Engine
===========================================

Pre-trains PyTorch EEGNet models and classical Scikit-Learn pipelines (CSP+LDA, Cov Tangent Space LR)
across single-subject, cluster-pooled, and submodular top-5 pooled dataset cohorts.

Supported Execution Modes:
---------------------------
- Strategy A / Baseline 3 & 5: Cluster-Mode Pooled Training (5 GNN cluster mode cohorts)
- Strategy B / Baseline 4 & 6: Submodular Top-5 Pooled Training (1 Top-5 submodular cohort)
- Strategy C: Single-Subject Training (Dedicated EEGNet for every donor subject)

Outputs:
--------
Saved to `trained_pipelines/eegnet/{dataset}/`:
- Checkpoint files (*.pt / *.pkl).
- Pre-training Accuracies Summary CSV (eegnet_pretraining_accuracies.csv).

Usage Examples:
---------------
python eegnet_tools/train_eegnet_models.py --dataset Dreyer2023 --mode all
python eegnet_tools/train_eegnet_models.py --dataset Dreyer2023 --mode cluster
python eegnet_tools/train_eegnet_models.py --dataset Dreyer2023 --mode submodular
"""

import os
# Prevent OpenBLAS/MKL thread lockup during PyRiemann matrix operations
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import glob
import argparse
import joblib
import json
import pandas as pd
import numpy as np
import moabb
from moabb.paradigms import MotorImagery
import mne
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from mne.decoding import CSP
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from eegnet_tools.eegnet_model import EEGNet
except ImportError:
    from eegnet_model import EEGNet

# Suppress verbose logs
mne.set_log_level('warning')
moabb.set_log_level('warning')


def load_subject_eeg_data(
    dataset_name: str,
    subject_id: int,
    fmin: float = 8.0,
    fmax: float = 30.0,
    channels: list[str] = None
) -> tuple[np.ndarray, np.ndarray]:
    """Load raw epoch data X and labels y for a single subject from MOABB."""
    try:
        dataset_class = getattr(moabb.datasets, dataset_name)
        dataset = dataset_class()
    except AttributeError:
        raise AttributeError(f"Dataset '{dataset_name}' not found in moabb.datasets")

    paradigm = MotorImagery(
        fmin=fmin,
        fmax=fmax,
        channels=channels,
        events=["left_hand", "right_hand"],
        n_classes=2
    )

    X, labels, meta = paradigm.get_data(dataset=dataset, subjects=[subject_id])
    
    # Map text labels to binary integers 0 and 1
    unique_labels = sorted(list(set(labels)))
    label_map = {lbl: idx for idx, lbl in enumerate(unique_labels)}
    y = np.array([label_map[lbl] for lbl in labels], dtype=np.int64)

    return X, y


def train_pytorch_eegnet(
    X_train: np.ndarray,
    y_train: np.ndarray,
    channels: int,
    samples: int,
    epochs: int = 100,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    device: str = "cpu"
) -> EEGNet:
    """Train PyTorch EEGNet model with AdamW and Cosine Annealing scheduler."""
    if device == "cpu" and hasattr(torch, "set_num_threads"):
        torch.set_num_threads(min(12, os.cpu_count() or 8))

    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    if X_tensor.ndim == 3:
        X_tensor = X_tensor.unsqueeze(1)  # Shape: (Batch, 1, Channels, Samples)
    y_tensor = torch.tensor(y_train, dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = EEGNet(n_classes=2, channels=channels, samples=samples).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    model.train()
    for ep in range(epochs):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    return model


def evaluate_eegnet_acc(model: EEGNet, X: np.ndarray, y: np.ndarray, device: str = "cpu", batch_size: int = 256) -> float:
    """Evaluate accuracy of trained EEGNet model using mini-batching to prevent memory overflow."""
    if device == "cpu" and hasattr(torch, "set_num_threads"):
        torch.set_num_threads(min(12, os.cpu_count() or 8))
    model.eval()
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_preds = []
    with torch.no_grad():
        for bx, _ in loader:
            if bx.ndim == 3:
                bx = bx.unsqueeze(1)
            bx = bx.to(device)
            logits = model(bx)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.append(preds)
    all_preds = np.concatenate(all_preds, axis=0)
    return float(np.mean(all_preds == y))


def train_classical_pooled_pipeline(pipeline_type: str, X_train: np.ndarray, y_train: np.ndarray, max_samples: int = 8000):
    """Train classical scikit-learn / PyRiemann pipeline on pooled data."""
    if len(y_train) > max_samples:
        indices = np.random.choice(len(y_train), size=max_samples, replace=False)
        X_fit, y_fit = X_train[indices], y_train[indices]
    else:
        X_fit, y_fit = X_train, y_train

    if pipeline_type == "CSP_LDA":
        pipe = make_pipeline(CSP(n_components=4), LDA())
    elif pipeline_type == "Cov_Tangent_Space_LR":
        pipe = make_pipeline(
            Covariances(estimator='oas'),
            TangentSpace(metric='logeuclid'),
            LogisticRegression(max_iter=1000)
        )
    else:
        raise ValueError(f"Unknown classical pipeline type '{pipeline_type}'")

    pipe.fit(X_fit, y_fit)
    return pipe


def main():
    parser = argparse.ArgumentParser(
        description="Pre-train PyTorch EEGNet and Classical Pooled Models (Baselines 3..6 & Strategies A..C)."
    )
    parser.add_argument(
        "--eval-all-single-subjects",
        action="store_true",
        help="Evaluate all 87 single-subject models for every target subject (very slow). Default: False"
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Execution device ('auto', 'cuda', 'cpu'). Default: auto"
    )
    parser.add_argument(
        "--dataset", "-d",
        default="Dreyer2023",
        help="Dataset name ('Dreyer2023', 'PhysionetMI', etc.). Default: Dreyer2023"
    )
    parser.add_argument(
        "--mode", "-m",
        default="all",
        choices=["all", "single", "cluster", "submodular"],
        help="Training mode: 'single' (Strategy C), 'cluster' (Strategy A & Baselines 3, 5), 'submodular' (Strategy B & Baselines 4, 6), or 'all'."
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=80,
        help="Number of PyTorch EEGNet pre-training epochs. Default: 80"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=128,
        help="PyTorch EEGNet training mini-batch size. Default: 128"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help="Custom output directory for trained model checkpoints."
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Execution device ('auto', 'cuda', 'cpu'). Default: auto"
    )

    args = parser.parse_args()

    selected_channels = ["C1", "C2", "C5", "C3", "C4", "C6", "FC3", "CP3", "FC4", "CP4", "Cz"]
    output_dir = args.output_dir or os.path.join("trained_pipelines", "eegnet", args.dataset)
    os.makedirs(output_dir, exist_ok=True)

    print("================================================================================")
    print(f" EEGNet & Pooled Models Pre-Training Engine")
    print("================================================================================")
    print(f" Dataset             : {args.dataset}")
    print(f" Mode                : {args.mode}")
    print(f" Epochs              : {args.epochs}")
    print(f" Output Directory    : {output_dir}")
    print("================================================================ launch \n")

    # Load all subject EEG data
    print(f"Loading EEG datasets for {args.dataset}...")
    
    # Read available subjects from existing trained_pipelines directory or MOABB dataset
    tp_dir = os.path.join("trained_pipelines", args.dataset)
    if os.path.exists(tp_dir):
        sub_dirs = [d for d in os.listdir(tp_dir) if d.startswith("subject_") and os.path.isdir(os.path.join(tp_dir, d))]
        subjects = sorted([int(d.replace("subject_", "")) for d in sub_dirs])
    else:
        subjects = list(range(1, 88))  # Dreyer2023 default 87 subjects

    subject_data = {}
    for s in subjects:
        try:
            X, y = load_subject_eeg_data(args.dataset, s, channels=selected_channels)
            subject_data[s] = (X, y)
        except Exception as e:
            print(f"  Warning: Failed to load subject {s}: {e}")

    if not subject_data:
        raise RuntimeError("No subject EEG datasets loaded successfully.")

    valid_subs = sorted(list(subject_data.keys()))
    print(f"Successfully loaded {len(valid_subs)} subjects for dataset '{args.dataset}'.\n")

    first_s = valid_subs[0]
    sample_X, _ = subject_data[first_s]
    _, num_channels, num_samples = sample_X.shape

    summary_rows = []

    # --- Strategy C: Single-Subject EEGNet Training ---
    if args.mode in ["all", "single"]:
        print("--- Running Strategy C: Single-Subject EEGNet Pre-Training ---")
        for s in valid_subs:
            X_s, y_s = subject_data[s]
            s_dir = os.path.join(output_dir, f"subject_{s}")
            os.makedirs(s_dir, exist_ok=True)

            model = train_pytorch_eegnet(X_s, y_s, channels=num_channels, samples=num_samples, epochs=args.epochs)
            train_acc = evaluate_eegnet_acc(model, X_s, y_s)

            ckpt_path = os.path.join(s_dir, "EEGNet_model.pt")
            torch.save(model.state_dict(), ckpt_path)

            summary_rows.append({
                'Pipeline': 'EEGNet_single_subject',
                'Subject_Or_Cohort': f"subject_{s}",
                'Sample_Size_Trials': len(y_s),
                'Pretrain_Acc': round(train_acc, 4),
                'Checkpoint_Path': ckpt_path
            })
            print(f"  Subject {s:2d} | Trials: {len(y_s):3d} | Pre-train Acc: {train_acc:.4f} -> {ckpt_path}")

    # Read GNN Cluster Assignments & Submodular Selection Sets
    cluster_csv = os.path.join("graph_results", "clustering", args.dataset, "subject_cluster_assignments.csv")
    submod_csv = os.path.join("graph_results", "wearable_selection", args.dataset, "wearable_5_classifier_sets.csv")

    # --- Strategy A & Baselines 3, 5: Cluster-Mode Pooled Training ---
    if args.mode in ["all", "cluster"] and os.path.exists(cluster_csv):
        print("\n--- Running Strategy A & Baselines 3, 5: Cluster-Mode Pooled Training ---")
        cdf = pd.read_csv(cluster_csv)
        cluster_col = 'Donor_Cluster' if 'Donor_Cluster' in cdf.columns else 'Cluster'
        for cluster_id in sorted(cdf[cluster_col].unique()):
            # Extract unique donor subjects present in subject_data
            cluster_subs = sorted(list(set([int(s) for s in cdf[cdf[cluster_col] == cluster_id]['Subject'] if int(s) in subject_data])))
            if not cluster_subs:
                continue

            X_pool = np.concatenate([subject_data[s][0] for s in cluster_subs], axis=0)
            y_pool = np.concatenate([subject_data[s][1] for s in cluster_subs], axis=0)

            print(f"  Training Cluster Mode {cluster_id} ({len(cluster_subs)} subjects, {len(y_pool)} trials)...", flush=True)

            # Strategy A: Cluster EEGNet
            eegnet_c = train_pytorch_eegnet(X_pool, y_pool, channels=num_channels, samples=num_samples, epochs=args.epochs, batch_size=args.batch_size)
            acc_eegnet = evaluate_eegnet_acc(eegnet_c, X_pool, y_pool, batch_size=args.batch_size*2)
            c_dir = os.path.join(output_dir, f"cluster_mode_{cluster_id}")
            os.makedirs(c_dir, exist_ok=True)
            ckpt_eegnet = os.path.join(c_dir, "EEGNet_cluster_model.pt")
            torch.save(eegnet_c.state_dict(), ckpt_eegnet)

            # Baseline 3: CSP + LDA Pooled
            csp_lda_c = train_classical_pooled_pipeline("CSP_LDA", X_pool, y_pool)
            acc_lda = float(csp_lda_c.score(X_pool, y_pool))
            ckpt_lda = os.path.join(c_dir, "CSP_LDA_cluster_model.pkl")
            joblib.dump(csp_lda_c, ckpt_lda)

            # Baseline 5: Cov Tangent Space LR Pooled
            cov_lr_c = train_classical_pooled_pipeline("Cov_Tangent_Space_LR", X_pool, y_pool)
            acc_cov = float(cov_lr_c.score(X_pool, y_pool))
            ckpt_cov = os.path.join(c_dir, "Cov_Tangent_Space_LR_cluster_model.pkl")
            joblib.dump(cov_lr_c, ckpt_cov)

            print(f"  Done Cluster Mode {cluster_id} | EEGNet: {acc_eegnet:.4f} | CSP+LDA: {acc_lda:.4f} | Cov+LR: {acc_cov:.4f}", flush=True)

            for p_name, acc_val, ckpt_val in [
                ("Strategy A (EEGNet Cluster Pooled)", acc_eegnet, ckpt_eegnet),
                ("Baseline 3 (CSP+LDA Cluster Pooled)", acc_lda, ckpt_lda),
                ("Baseline 5 (Cov Tangent LR Cluster Pooled)", acc_cov, ckpt_cov)
            ]:
                summary_rows.append({
                    'Pipeline': p_name,
                    'Subject_Or_Cohort': f"cluster_mode_{cluster_id}",
                    'Sample_Size_Trials': len(y_pool),
                    'Pretrain_Acc': round(acc_val, 4),
                    'Checkpoint_Path': ckpt_val
                })

            print(f"  Cluster Mode {cluster_id} | Subs: {len(cluster_subs):2d} | Trials: {len(y_pool):4d} | EEGNet: {acc_eegnet:.4f} | CSP+LDA: {acc_lda:.4f} | Cov+LR: {acc_cov:.4f}")

    # --- Strategy B & Baselines 4, 6: Submodular Top-5 Pooled Training ---
    if args.mode in ["all", "submodular"]:
        print("\n--- Running Strategy B & Baselines 4, 6: Submodular Top-5 Pooled Training ---")
        top_5_subs = []
        if os.path.exists(submod_csv):
            sdf = pd.read_csv(submod_csv)
            submod_row = sdf[sdf['Selection_Method'].str.contains("Submodular Greedy")]
            if not submod_row.empty:
                top_5_str = submod_row['Selected_Source_Classifiers'].iloc[0]
                top_5_subs = [int(s.strip()) for s in top_5_str.split(",") if s.strip().isdigit()]

        if not top_5_subs:
            top_5_subs = valid_subs[:5]

        top_5_subs = [s for s in top_5_subs if s in subject_data]
        X_sub_pool = np.concatenate([subject_data[s][0] for s in top_5_subs], axis=0)
        y_sub_pool = np.concatenate([subject_data[s][1] for s in top_5_subs], axis=0)

        sub_dir = os.path.join(output_dir, "submodular_top5_pooled")
        os.makedirs(sub_dir, exist_ok=True)

        # Strategy B: Submodular Top-5 EEGNet
        eegnet_sub = train_pytorch_eegnet(X_sub_pool, y_sub_pool, channels=num_channels, samples=num_samples, epochs=args.epochs)
        acc_eegnet_sub = evaluate_eegnet_acc(eegnet_sub, X_sub_pool, y_sub_pool)
        ckpt_eegnet_sub = os.path.join(sub_dir, "EEGNet_submodular_top5_model.pt")
        torch.save(eegnet_sub.state_dict(), ckpt_eegnet_sub)

        # Baseline 4: CSP + LDA Top-5 Pooled
        csp_lda_sub = train_classical_pooled_pipeline("CSP_LDA", X_sub_pool, y_sub_pool)
        acc_lda_sub = float(csp_lda_sub.score(X_sub_pool, y_sub_pool))
        ckpt_lda_sub = os.path.join(sub_dir, "CSP_LDA_submodular_top5_model.pkl")
        joblib.dump(csp_lda_sub, ckpt_lda_sub)

        # Baseline 6: Cov Tangent Space LR Top-5 Pooled
        cov_lr_sub = train_classical_pooled_pipeline("Cov_Tangent_Space_LR", X_sub_pool, y_sub_pool)
        acc_cov_sub = float(cov_lr_sub.score(X_sub_pool, y_sub_pool))
        ckpt_cov_sub = os.path.join(sub_dir, "Cov_Tangent_Space_LR_submodular_top5_model.pkl")
        joblib.dump(cov_lr_sub, ckpt_cov_sub)

        for p_name, acc_val, ckpt_val in [
            ("Strategy B (EEGNet Top-5 Pooled)", acc_eegnet_sub, ckpt_eegnet_sub),
            ("Baseline 4 (CSP+LDA Top-5 Pooled)", acc_lda_sub, ckpt_lda_sub),
            ("Baseline 6 (Cov Tangent LR Top-5 Pooled)", acc_cov_sub, ckpt_cov_sub)
        ]:
            summary_rows.append({
                'Pipeline': p_name,
                'Subject_Or_Cohort': "submodular_top5_pooled",
                'Sample_Size_Trials': len(y_sub_pool),
                'Pretrain_Acc': round(acc_val, 4),
                'Checkpoint_Path': ckpt_val
            })

        print(f"  Submodular Top-5 Cohort {top_5_subs} | Trials: {len(y_sub_pool):4d} | EEGNet: {acc_eegnet_sub:.4f} | CSP+LDA: {acc_lda_sub:.4f} | Cov+LR: {acc_cov_sub:.4f}")

    # Save / Upsert summary CSV
    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = os.path.join(output_dir, "eegnet_pretraining_accuracies.csv")

    if os.path.exists(summary_csv_path) and not summary_df.empty:
        try:
            existing_df = pd.read_csv(summary_csv_path)
            combined_df = pd.concat([existing_df, summary_df], ignore_index=True)
            summary_df = combined_df.drop_duplicates(subset=['Pipeline', 'Subject_Or_Cohort'], keep='last')
        except Exception as e:
            print(f"  Warning: Could not merge with existing summary CSV ({e}), writing new file.")

    summary_df.to_csv(summary_csv_path, index=False)

    print("\n================================================================================")
    print(f" Saved Pre-training Summary CSV: {summary_csv_path}")
    print("================================================================================\n")


if __name__ == "__main__":
    main()
