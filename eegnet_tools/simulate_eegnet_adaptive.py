"""
EEGNet & Pooled Models Adaptive Simulation Engine
===================================================

Simulates online trial-by-trial classification for pre-trained PyTorch EEGNet models
and classical pooled models (CSP+LDA, Cov Tangent Space LR) on target subject data streams.

Adaptation Protocols:
---------------------
- EEGNet Full-Model Fine-Tuning: Every 4 trials, fine-tunes the entire EEGNet model (lr = 1e-4).
- Classical Model Online Refitting: Every 4 trials, re-fits CSP+LDA / Tangent Space LR pipelines.
- Baseline 1 (Target Scratch EEGNet): Starts from scratch with zero prior knowledge.

Outputs:
--------
Saved to `simulation_results/eegnet/{dataset}/{subject_id}.csv`:
- Trial-by-trial predictions, correctness, and cumulative accuracy per model.

Usage Examples:
---------------
python eegnet_tools/simulate_eegnet_adaptive.py --dataset Dreyer2023
"""

import os
import glob
import argparse
import copy
import joblib
import pandas as pd
import numpy as np
import moabb
from moabb.paradigms import MotorImagery
import mne

import torch
import torch.nn as nn
import torch.optim as optim

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from eegnet_tools.eegnet_model import EEGNet
except ImportError:
    from eegnet_model import EEGNet

mne.set_log_level('warning')
moabb.set_log_level('warning')


def load_subject_eeg_data(
    dataset_name: str,
    subject_id: int,
    fmin: float = 8.0,
    fmax: float = 30.0,
    channels: list[str] = None
) -> tuple[np.ndarray, list[str]]:
    """Load raw epoch data X and text labels for a target subject."""
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
    return X, labels


def simulate_eegnet_online_adaptation(
    X: np.ndarray,
    labels: list[str],
    pretrained_models: dict[str, object],
    dataset_name: str,
    subject_id: int,
    fine_tune_lr: float = 1e-4,
    adapt_interval: int = 8,
    device: str = "auto",
    adapt_mode: str = "head_only",
    max_buffer: int = 64
) -> pd.DataFrame:
    """Simulate online adaptive classification with trial-by-trial fine-tuning."""
    if device == "auto":
        device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_obj = torch.device(device)

    if device_obj.type == "cpu" and hasattr(torch, "set_num_threads"):
        torch.set_num_threads(min(12, os.cpu_count() or 8))

    unique_labels = sorted(list(set(labels)))
    label_map = {lbl: idx for idx, lbl in enumerate(unique_labels)}
    idx_to_label = {idx: lbl for lbl, idx in label_map.items()}

    results = []

    for model_name, raw_model in pretrained_models.items():
        is_eegnet = isinstance(raw_model, EEGNet) or model_name.startswith("EEGNet")
        is_baseline_scratch = model_name.startswith("baseline_scratch")

        if is_eegnet:
            pipeline = copy.deepcopy(raw_model).to(device_obj)
            pipeline.eval()
            if is_baseline_scratch:
                # Cold-start scratch model needs full model training with lr 1e-3 and 5 mini-epochs
                for param in pipeline.parameters():
                    param.requires_grad = True
                optimizer = optim.AdamW(pipeline.parameters(), lr=1e-3, weight_decay=1e-2)
                epochs_per_step = 5
            elif adapt_mode == "head_only":
                # Freeze feature extractor layers; fine-tune linear head only
                for param in pipeline.parameters():
                    param.requires_grad = False
                for param in pipeline.classifier.parameters():
                    param.requires_grad = True
                optimizer = optim.AdamW(pipeline.classifier.parameters(), lr=1e-3, weight_decay=1e-2)
                epochs_per_step = 1
            else:
                # Full model fine-tuning with low learning rate
                for param in pipeline.parameters():
                    param.requires_grad = True
                optimizer = optim.AdamW(pipeline.parameters(), lr=fine_tune_lr, weight_decay=1e-2)
                epochs_per_step = 1

            criterion = nn.CrossEntropyLoss()
        else:
            pipeline = copy.deepcopy(raw_model)

        X_history = []
        y_history = []
        correct_predictions = 0
        total_predictions = 0

        for i, (x_trial, y_true_lbl) in enumerate(zip(X, labels)):
            y_true_idx = label_map[y_true_lbl]
            X_test = np.expand_dims(x_trial, axis=0)

            # Predict current trial
            pred_lbl = None
            try:
                if is_eegnet:
                    with torch.no_grad():
                        tx = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1).to(device_obj)
                        logits = pipeline(tx)
                        pred_idx = int(torch.argmax(logits, dim=1).cpu().item())
                        pred_lbl = idx_to_label[pred_idx]
                else:
                    pred_lbl = pipeline.predict(X_test)[0]
            except Exception:
                pred_lbl = None

            is_correct = (str(pred_lbl).lower() == str(y_true_lbl).lower())
            if is_correct:
                correct_predictions += 1
            total_predictions += 1

            results.append({
                'Classifier': model_name,
                'Trial': i,
                'True_Label': y_true_lbl,
                'Predicted_Label': pred_lbl,
                'Is_Correct': is_correct,
                'Cumulative_Accuracy': correct_predictions / total_predictions
            })

            X_history.append(x_trial)
            y_history.append(y_true_idx if is_eegnet else y_true_lbl)

            # Online adaptation / fine-tuning at configured trial interval using sliding max_buffer
            if len(X_history) >= adapt_interval and len(X_history) % adapt_interval == 0:
                buf_X = X_history[-max_buffer:]
                buf_y = y_history[-max_buffer:]
                if len(set(buf_y)) == len(unique_labels):
                    try:
                        X_arr = np.array(buf_X)
                        if is_eegnet:
                            pipeline.train()
                            tx_hist = torch.tensor(X_arr, dtype=torch.float32).unsqueeze(1).to(device_obj)
                            ty_hist = torch.tensor(buf_y, dtype=torch.long).to(device_obj)
                            for _ in range(epochs_per_step):
                                optimizer.zero_grad()
                                logits = pipeline(tx_hist)
                                loss = criterion(logits, ty_hist)
                                loss.backward()
                                optimizer.step()
                            pipeline.eval()
                        else:
                            pipeline.fit(X_arr, np.array(buf_y))
                    except Exception as e:
                        pass

    df_results = pd.DataFrame(results)
    out_dir = os.path.join("simulation_results", "eegnet", dataset_name)
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{subject_id}.csv")
    df_results.to_csv(out_file, index=False)
    return df_results


def main():
    parser = argparse.ArgumentParser(
        description="Simulate online adaptive classification for PyTorch EEGNet and Pooled Models."
    )
    parser.add_argument(
        "--dataset", "-d",
        default="Dreyer2023",
        help="Dataset name ('Dreyer2023', 'PhysionetMI', etc.). Default: Dreyer2023"
    )
    parser.add_argument(
        "--models-dir", "-m",
        default=None,
        help="Directory containing pre-trained model checkpoints."
    )
    parser.add_argument(
        "--adapt-interval", "-i",
        type=int,
        default=8,
        help="Online adaptation fine-tuning trial interval (e.g. 4, 8, 12). Default: 8"
    )
    parser.add_argument(
        "--adapt-mode",
        choices=["head_only", "full_model"],
        default="head_only",
        help="Adaptation fine-tuning mode ('head_only' vs 'full_model'). Default: head_only"
    )
    parser.add_argument(
        "--max-buffer",
        type=int,
        default=64,
        help="Maximum sliding window history buffer size for adaptation steps. Default: 64"
    )
    parser.add_argument(
        "--strict-loso",
        action="store_true",
        help="Strict Leave-One-Subject-Out (LOSO): Skip cluster models that included target subject during pre-training. Default: False"
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Execution device ('auto', 'cuda', 'cpu'). Default: auto"
    )
    parser.add_argument(
        "--eval-all-single-subjects",
        action="store_true",
        help="Evaluate all 87 single-subject models for every target subject (very slow). Default: False"
    )

    args = parser.parse_args()
    selected_channels = ["C1", "C2", "C5", "C3", "C4", "C6", "FC3", "CP3", "FC4", "CP4", "Cz"]
    models_dir = args.models_dir or os.path.join("trained_pipelines", "eegnet", args.dataset)

    if not os.path.exists(models_dir):
        raise FileNotFoundError(f"Model checkpoints directory '{models_dir}' does not exist.")

    print("================================================================================")
    print(f" EEGNet & Pooled Models Adaptive Online Simulation Engine")
    print("================================================================================")
    print(f" Dataset             : {args.dataset}")
    print(f" Checkpoints Directory: {models_dir}")
    print("================================================================================\n")

    # Load sample subject to inspect num_samples dynamically
    sample_X, _ = load_subject_eeg_data(args.dataset, 1, channels=selected_channels)
    _, num_channels, num_samples = sample_X.shape

    # Load pre-trained models
    pretrained_models = {}

    # 1. Baseline 1: Scratch EEGNet
    pretrained_models["baseline_scratch_EEGNet"] = EEGNet(n_classes=2, channels=num_channels, samples=num_samples)

    # 2. Single Subject EEGNets (Strategy C)
    sub_dirs = glob.glob(os.path.join(models_dir, "subject_*"))
    for s_dir in sub_dirs:
        sub_id = os.path.basename(s_dir).replace("subject_", "")
        pt_file = os.path.join(s_dir, "EEGNet_model.pt")
        if os.path.exists(pt_file):
            net = EEGNet(n_classes=2, channels=num_channels, samples=num_samples)
            net.load_state_dict(torch.load(pt_file, map_location="cpu", weights_only=True))
            pretrained_models[f"subject_{sub_id}_EEGNet_pipeline"] = net

    # 3. Cluster Mode Models (Strategy A & Baselines 3, 5)
    cluster_dirs = glob.glob(os.path.join(models_dir, "cluster_mode_*"))
    for c_dir in cluster_dirs:
        c_id = os.path.basename(c_dir).replace("cluster_mode_", "")
        net_file = os.path.join(c_dir, "EEGNet_cluster_model.pt")
        lda_file = os.path.join(c_dir, "CSP_LDA_cluster_model.pkl")
        cov_file = os.path.join(c_dir, "Cov_Tangent_Space_LR_cluster_model.pkl")

        if os.path.exists(net_file):
            net = EEGNet(n_classes=2, channels=num_channels, samples=num_samples)
            net.load_state_dict(torch.load(net_file, map_location="cpu", weights_only=True))
            pretrained_models[f"cluster_{c_id}_EEGNet_pipeline"] = net

        if os.path.exists(lda_file):
            pretrained_models[f"cluster_{c_id}_CSP_LDA_pipeline"] = joblib.load(lda_file)

        if os.path.exists(cov_file):
            pretrained_models[f"cluster_{c_id}_Cov_Tangent_Space_LR_pipeline"] = joblib.load(cov_file)

    # 4. Submodular Top-5 Models (Strategy B & Baselines 4, 6)
    sub_top5_dir = os.path.join(models_dir, "submodular_top5_pooled")
    if os.path.exists(sub_top5_dir):
        net_file = os.path.join(sub_top5_dir, "EEGNet_submodular_top5_model.pt")
        lda_file = os.path.join(sub_top5_dir, "CSP_LDA_submodular_top5_model.pkl")
        cov_file = os.path.join(sub_top5_dir, "Cov_Tangent_Space_LR_submodular_top5_model.pkl")

        if os.path.exists(net_file):
            net = EEGNet(n_classes=2, channels=num_channels, samples=num_samples)
            net.load_state_dict(torch.load(net_file, map_location="cpu", weights_only=True))
            pretrained_models["submodular_top5_EEGNet_pipeline"] = net

        if os.path.exists(lda_file):
            pretrained_models["submodular_top5_CSP_LDA_pipeline"] = joblib.load(lda_file)

        if os.path.exists(cov_file):
            pretrained_models["submodular_top5_Cov_Tangent_Space_LR_pipeline"] = joblib.load(cov_file)

    print(f"Loaded {len(pretrained_models)} pre-trained models for online simulation.")

    # Determine subjects to simulate
    tp_dir = os.path.join("trained_pipelines", args.dataset)
    if os.path.exists(tp_dir):
        sub_dirs = [d for d in os.listdir(tp_dir) if d.startswith("subject_") and os.path.isdir(os.path.join(tp_dir, d))]
        subjects = sorted([int(d.replace("subject_", "")) for d in sub_dirs])
    else:
        subjects = list(range(1, 88))

    # Load cluster assignments for strict LOSO filtering
    cluster_map = {}
    cluster_csv = os.path.join("graph_results", "clustering", args.dataset, "subject_cluster_assignments.csv")
    if os.path.exists(cluster_csv):
        cdf = pd.read_csv(cluster_csv)
        cluster_col = "Donor_Cluster" if "Donor_Cluster" in cdf.columns else "Cluster"
        for c_id in cdf[cluster_col].unique():
            cluster_map[int(c_id)] = set([int(s) for s in cdf[cdf[cluster_col] == c_id]['Subject']])

    for target_subject in subjects:
        try:
            X, labels = load_subject_eeg_data(args.dataset, target_subject, channels=selected_channels)

            target_models = {}
            for m_name, m_obj in pretrained_models.items():
                if not m_name.startswith("subject_"):
                    if args.strict_loso and m_name.startswith("cluster_"):
                        try:
                            c_id = int(m_name.split("_")[1])
                            if target_subject in cluster_map.get(c_id, set()):
                                continue # Skip cluster model that contained target subject in pre-training
                        except Exception:
                            pass
                    target_models[m_name] = m_obj

            # Add matching single subject model (Strategy C - Matched Subject)
            matching_key = f"subject_{target_subject}_EEGNet_pipeline"
            if matching_key in pretrained_models:
                target_models[matching_key] = pretrained_models[matching_key]

            # Add 5 random mismatched single subject donor models (Strategy C - Mismatched Donors)
            mismatched_keys = [k for k in pretrained_models if k.startswith("subject_") and k != matching_key]
            if mismatched_keys:
                rng = np.random.RandomState(target_subject)
                chosen_mismatched = rng.choice(mismatched_keys, size=min(5, len(mismatched_keys)), replace=False)
                for idx, m_key in enumerate(chosen_mismatched):
                    target_models[f"subject_mismatched_{idx+1}_EEGNet_pipeline"] = pretrained_models[m_key]

            print(f"Simulating target subject {target_subject:2d} ({len(labels)} trials, {len(target_models)} models, adapt mode: {args.adapt_mode}, max buffer: {args.max_buffer})...", flush=True)
            simulate_eegnet_online_adaptation(X, labels, target_models, args.dataset, target_subject, adapt_interval=args.adapt_interval, device=args.device, adapt_mode=args.adapt_mode, max_buffer=args.max_buffer)
        except Exception as e:
            print(f"  Warning: Simulation failed for target subject {target_subject}: {e}")

    print("\n================================================================================")
    print(f" Simulation Complete! Saved CSVs to simulation_results/eegnet/{args.dataset}/")
    print("================================================================================\n")


if __name__ == "__main__":
    main()
