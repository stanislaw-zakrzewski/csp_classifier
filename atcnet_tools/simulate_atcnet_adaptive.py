"""
ATCNet Adaptive Online Simulation Engine
=========================================

Simulates online adaptive classification with trial-by-trial evaluation and 
intermittent full-model fine-tuning for PyTorch ATCNet models.

Supports GPU acceleration, CPU multi-threading, Zero-Shot Trial 0 prediction logging,
Scratch ATCNet cold-start training, and Mismatched Single-Donor cross-subject transfer benchmarking.
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import glob
import copy
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from moabb.paradigms import MotorImagery
import moabb.datasets as mb_datasets

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


def simulate_atcnet_online_adaptation(
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
    """Simulate online adaptive classification for ATCNet with trial-by-trial fine-tuning."""
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
        is_atcnet = isinstance(raw_model, ATCNet) or model_name.startswith("ATCNet") or "ATCNet" in model_name
        is_baseline_scratch = model_name.startswith("baseline_scratch")

        if is_atcnet:
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

            # Predict current trial (Zero-Shot on Trial 0)
            pred_lbl = None
            try:
                if is_atcnet:
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
            y_history.append(y_true_idx if is_atcnet else y_true_lbl)

            # Online adaptation / fine-tuning at configured trial interval using sliding max_buffer
            if len(X_history) >= adapt_interval and len(X_history) % adapt_interval == 0:
                buf_X = X_history[-max_buffer:]
                buf_y = y_history[-max_buffer:]
                if len(set(buf_y)) == len(unique_labels):
                    try:
                        X_arr = np.array(buf_X)
                        if is_atcnet:
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
                    except Exception:
                        pass

    df_results = pd.DataFrame(results)
    out_dir = os.path.join("simulation_results", "atcnet", dataset_name)
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{subject_id}.csv")
    df_results.to_csv(out_file, index=False)
    return df_results


def main():
    parser = argparse.ArgumentParser(description="Simulate online adaptive classification for PyTorch ATCNet models.")
    parser.add_argument("--dataset", "-d", default="Dreyer2023", help="Dataset name. Default: Dreyer2023")
    parser.add_argument("--models-dir", "-m", default=None, help="Directory containing pre-trained ATCNet checkpoints.")
    parser.add_argument("--adapt-interval", "-i", type=int, default=8, help="Online adaptation trial interval. Default: 8")
    parser.add_argument("--adapt-mode", choices=["head_only", "full_model"], default="head_only", help="Adaptation fine-tuning mode ('head_only' vs 'full_model'). Default: head_only")
    parser.add_argument("--max-buffer", type=int, default=64, help="Maximum sliding window history buffer size. Default: 64")
    parser.add_argument("--strict-loso", action="store_true", help="Strict Leave-One-Subject-Out (LOSO): Skip cluster models that included target subject during pre-training. Default: False")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", help="Execution device. Default: auto")

    args = parser.parse_args()
    selected_channels = ["C1", "C2", "C5", "C3", "C4", "C6", "FC3", "CP3", "FC4", "CP4", "Cz"]
    models_dir = args.models_dir or os.path.join("trained_pipelines", "atcnet", args.dataset)

    if not os.path.exists(models_dir):
        raise FileNotFoundError(f"Model checkpoints directory '{models_dir}' does not exist.")

    device_str = "cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())) else "cpu"

    print("================================================================================")
    print(f" ATCNet Adaptive Online Simulation Engine")
    print("================================================================================")
    print(f" Dataset             : {args.dataset}")
    print(f" Checkpoints Directory: {models_dir}")
    print(f" Adapt Interval      : Every {args.adapt_interval} trials")
    print(f" Strict LOSO Filter  : {args.strict_loso}")
    print(f" Device             : {device_str.upper()}")
    print("================================================================Threshold\n")

    # Load sample subject to inspect num_samples dynamically
    sample_X, _ = load_subject_eeg_data(args.dataset, 1, channels=selected_channels)
    _, num_channels, num_samples = sample_X.shape

    # Load pre-trained models
    pretrained_models = {}

    # 1. Baseline 1: Scratch ATCNet
    pretrained_models["baseline_scratch_ATCNet"] = ATCNet(n_classes=2, channels=num_channels, samples=num_samples)

    # 2. Single Subject ATCNets
    sub_dirs = glob.glob(os.path.join(models_dir, "subject_*"))
    for s_dir in sub_dirs:
        sub_id = os.path.basename(s_dir).replace("subject_", "")
        pt_file = os.path.join(s_dir, "ATCNet_model.pt")
        if os.path.exists(pt_file):
            net = ATCNet(n_classes=2, channels=num_channels, samples=num_samples)
            net.load_state_dict(torch.load(pt_file, map_location="cpu", weights_only=True))
            pretrained_models[f"subject_{sub_id}_ATCNet_pipeline"] = net

    # 3. Cluster Mode ATCNets (Strategy A)
    cluster_dirs = glob.glob(os.path.join(models_dir, "cluster_mode_*"))
    for c_dir in cluster_dirs:
        c_id = os.path.basename(c_dir).replace("cluster_mode_", "")
        net_file = os.path.join(c_dir, "ATCNet_cluster_model.pt")
        if os.path.exists(net_file):
            net = ATCNet(n_classes=2, channels=num_channels, samples=num_samples)
            net.load_state_dict(torch.load(net_file, map_location="cpu", weights_only=True))
            pretrained_models[f"cluster_{c_id}_ATCNet_pipeline"] = net

    # 4. Submodular Top-5 ATCNet (Strategy B)
    sub_top5_dir = os.path.join(models_dir, "submodular_top5_pooled")
    if os.path.exists(sub_top5_dir):
        net_file = os.path.join(sub_top5_dir, "ATCNet_submodular_top5_model.pt")
        if os.path.exists(net_file):
            net = ATCNet(n_classes=2, channels=num_channels, samples=num_samples)
            net.load_state_dict(torch.load(net_file, map_location="cpu", weights_only=True))
            pretrained_models["submodular_top5_ATCNet_pipeline"] = net

    print(f"Loaded {len(pretrained_models)} pre-trained ATCNet models for online simulation.")

    # Load cluster assignments for strict LOSO filtering
    cluster_map = {}
    cluster_csv = os.path.join("graph_results", "clustering", args.dataset, "subject_cluster_assignments.csv")
    if os.path.exists(cluster_csv):
        cdf = pd.read_csv(cluster_csv)
        cluster_col = "Donor_Cluster" if "Donor_Cluster" in cdf.columns else "Cluster"
        for c_id in cdf[cluster_col].unique():
            cluster_map[int(c_id)] = set([int(s) for s in cdf[cdf[cluster_col] == c_id]['Subject']])

    # Determine target subjects
    tp_dir = os.path.join("trained_pipelines", args.dataset)
    if os.path.exists(tp_dir):
        sub_dirs = [d for d in os.listdir(tp_dir) if d.startswith("subject_") and os.path.isdir(os.path.join(tp_dir, d))]
        subjects = sorted([int(d.replace("subject_", "")) for d in sub_dirs])
    else:
        subjects = list(range(1, 88))

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
            matching_key = f"subject_{target_subject}_ATCNet_pipeline"
            if matching_key in pretrained_models:
                target_models[matching_key] = pretrained_models[matching_key]

            # Add 5 random mismatched single subject donor models (Strategy C - Mismatched Donors)
            mismatched_keys = [k for k in pretrained_models if k.startswith("subject_") and k != matching_key]
            if mismatched_keys:
                rng = np.random.RandomState(target_subject)
                chosen_mismatched = rng.choice(mismatched_keys, size=min(5, len(mismatched_keys)), replace=False)
                for idx, m_key in enumerate(chosen_mismatched):
                    target_models[f"subject_mismatched_{idx+1}_ATCNet_pipeline"] = pretrained_models[m_key]

            print(f"Simulating target subject {target_subject:2d} ({len(labels)} trials, {len(target_models)} models, adapt mode: {args.adapt_mode}, max buffer: {args.max_buffer})...", flush=True)
            simulate_atcnet_online_adaptation(X, labels, target_models, args.dataset, target_subject, adapt_interval=args.adapt_interval, device=args.device, adapt_mode=args.adapt_mode, max_buffer=args.max_buffer)
        except Exception as e:
            print(f"  Warning: Simulation failed for target subject {target_subject}: {e}")

    print("\n================================================================================")
    print(f" ATCNet Simulation Complete! Saved CSVs to simulation_results/atcnet/{args.dataset}/")
    print("================================================================================\n")


if __name__ == "__main__":
    main()
