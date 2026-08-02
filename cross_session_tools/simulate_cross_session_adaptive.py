"""
Cross-Session Out-of-Session Adaptive Simulation Engine
=========================================================

Evaluates Session 0 (Day 1) pre-trained ATCNet, EEGNet, CSP+LDA, and Cov+LR models
on Session 1 (Day 2) and Session 2 (Day 3) data to quantify cross-session performance and multi-day time decay.
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

# Ensure repository root is on sys.path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.append(repo_root)

from eegnet_tools.eegnet_model import EEGNet
from atcnet_tools.atcnet_model import ATCNet
from cross_session_tools.train_cross_session_models import load_subject_eeg_session_data


def simulate_session_trials(
    X: np.ndarray,
    labels: list[str],
    pretrained_models: dict[str, object],
    dataset_name: str,
    subject_id: int,
    session_tag: str,
    adapt_interval: int = 8,
    device: str = "auto",
    adapt_mode: str = "head_only",
    max_buffer: int = 64,
    fine_tune_lr: float = 1e-4
) -> list[dict]:
    """Simulate online trial-by-trial adaptive classification for a single session."""
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
        is_eegnet = isinstance(raw_model, EEGNet) or ("EEGNet" in model_name and not model_name.startswith("CSP_LDA"))
        is_atcnet = isinstance(raw_model, ATCNet) or ("ATCNet" in model_name)
        is_pytorch = (is_eegnet or is_atcnet)
        is_baseline_scratch = model_name.startswith("baseline_scratch")

        if is_pytorch:
            pipeline = copy.deepcopy(raw_model).to(device_obj)
            pipeline.eval()
            if is_baseline_scratch:
                for param in pipeline.parameters():
                    param.requires_grad = True
                optimizer = optim.AdamW(pipeline.parameters(), lr=1e-3, weight_decay=1e-2)
                epochs_per_step = 5
            elif adapt_mode == "head_only":
                for param in pipeline.parameters():
                    param.requires_grad = False
                for param in pipeline.classifier.parameters():
                    param.requires_grad = True
                optimizer = optim.AdamW(pipeline.classifier.parameters(), lr=1e-3, weight_decay=1e-2)
                epochs_per_step = 1
            else:
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
                if is_pytorch:
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
                'Session_Tag': session_tag,
                'Subject': subject_id,
                'Classifier': model_name,
                'Trial': i,
                'True_Label': y_true_lbl,
                'Predicted_Label': pred_lbl,
                'Is_Correct': is_correct,
                'Cumulative_Accuracy': correct_predictions / total_predictions
            })

            X_history.append(x_trial)
            y_history.append(y_true_idx if is_pytorch else y_true_lbl)

            # Online adaptation / fine-tuning at configured trial interval using sliding max_buffer
            if len(X_history) >= adapt_interval and len(X_history) % adapt_interval == 0:
                buf_X = X_history[-max_buffer:]
                buf_y = y_history[-max_buffer:]
                if len(set(buf_y)) == len(unique_labels):
                    try:
                        X_arr = np.array(buf_X)
                        if is_pytorch:
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

    return results


def main():
    parser = argparse.ArgumentParser(description="Simulate Cross-Session Out-of-Session Adaptive Classification.")
    parser.add_argument("--dataset", "-d", default="Yang2025", help="Dataset name. Default: Yang2025")
    parser.add_argument("--model", "-m", default="all", help="Model family ('all', 'atcnet', 'eegnet', 'classical'). Default: all")
    parser.add_argument("--models-dir", default=None, help="Directory containing Session 0 pre-trained checkpoints.")
    parser.add_argument("--adapt-interval", "-i", type=int, default=8, help="Online adaptation trial interval. Default: 8")
    parser.add_argument("--adapt-mode", choices=["head_only", "full_model"], default="head_only", help="Adaptation mode. Default: head_only")
    parser.add_argument("--max-buffer", type=int, default=64, help="Maximum sliding window buffer size. Default: 64")
    parser.add_argument("--strict-loso", action="store_true", help="Strict Leave-One-Subject-Out (LOSO) for cluster models. Default: False")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", help="Execution device. Default: auto")

    args = parser.parse_args()
    selected_channels = ["C1", "C2", "C5", "C3", "C4", "C6", "FC3", "CP3", "FC4", "CP4", "Cz"]

    if args.models_dir and os.path.isdir(args.models_dir):
        models_dir = args.models_dir
    else:
        models_dir = os.path.join("trained_pipelines", "cross_session", args.dataset)

    if not os.path.exists(models_dir):
        raise FileNotFoundError(f"Cross-session checkpoints directory '{models_dir}' not found. Run train_cross_session_models.py first.")

    device_str = "cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())) else "cpu"

    print("================================================================================")
    print(" Cross-Session Out-of-Session Adaptive Simulation Engine")
    print("================================================================================")
    print(f" Dataset             : {args.dataset}")
    print(f" Checkpoints Dir     : {models_dir}")
    print(f" Test Sessions       : Session 1 (Day 2) & Session 2 (Day 3)")
    print(f" Adapt Mode          : {args.adapt_mode}")
    print(f" Max Buffer          : {args.max_buffer}")
    print(f" Device             : {device_str.upper()}")
    print("================================================================================\n")

    # Load sample subject to inspect num_samples dynamically
    sample_X, _ = load_subject_eeg_session_data(args.dataset, 1, selected_channels, session_filter="1")
    num_channels, num_samples = sample_X.shape[1], sample_X.shape[2]

    # Load all Session 0 pre-trained models
    pretrained_models = {}

    # 1. Scratch Baselines
    pretrained_models["baseline_scratch_ATCNet"] = ATCNet(n_classes=2, channels=num_channels, samples=num_samples)
    pretrained_models["baseline_scratch_EEGNet"] = EEGNet(n_classes=2, channels=num_channels, samples=num_samples)

    # 2. Single Subject Models
    sub_dirs = glob.glob(os.path.join(models_dir, "subject_*"))
    for s_dir in sub_dirs:
        sub_id = os.path.basename(s_dir).replace("subject_", "")
        atc_file = os.path.join(s_dir, "ATCNet_model.pt")
        eeg_file = os.path.join(s_dir, "EEGNet_model.pt")
        csp_file = os.path.join(s_dir, "CSP_LDA_model.pkl")
        cov_file = os.path.join(s_dir, "Cov_Tangent_Space_LR_model.pkl")

        if os.path.exists(atc_file):
            net = ATCNet(n_classes=2, channels=num_channels, samples=num_samples)
            net.load_state_dict(torch.load(atc_file, map_location="cpu", weights_only=True))
            pretrained_models[f"subject_{sub_id}_ATCNet_pipeline"] = net

        if os.path.exists(eeg_file):
            net = EEGNet(n_classes=2, channels=num_channels, samples=num_samples)
            net.load_state_dict(torch.load(eeg_file, map_location="cpu", weights_only=True))
            pretrained_models[f"subject_{sub_id}_EEGNet_pipeline"] = net

        if os.path.exists(csp_file):
            pretrained_models[f"subject_{sub_id}_CSP_LDA_pipeline"] = joblib.load(csp_file)

        if os.path.exists(cov_file):
            pretrained_models[f"subject_{sub_id}_Cov_Tangent_Space_LR_pipeline"] = joblib.load(cov_file)

    # 3. Cluster Mode Models
    cluster_dirs = glob.glob(os.path.join(models_dir, "cluster_mode_*"))
    for c_dir in cluster_dirs:
        c_id = os.path.basename(c_dir).replace("cluster_mode_", "")
        atc_file = os.path.join(c_dir, "ATCNet_cluster_model.pt")
        eeg_file = os.path.join(c_dir, "EEGNet_cluster_model.pt")
        csp_file = os.path.join(c_dir, "CSP_LDA_cluster_model.pkl")
        cov_file = os.path.join(c_dir, "Cov_Tangent_Space_LR_cluster_model.pkl")

        if os.path.exists(atc_file):
            net = ATCNet(n_classes=2, channels=num_channels, samples=num_samples)
            net.load_state_dict(torch.load(atc_file, map_location="cpu", weights_only=True))
            pretrained_models[f"cluster_{c_id}_ATCNet_pipeline"] = net

        if os.path.exists(eeg_file):
            net = EEGNet(n_classes=2, channels=num_channels, samples=num_samples)
            net.load_state_dict(torch.load(eeg_file, map_location="cpu", weights_only=True))
            pretrained_models[f"cluster_{c_id}_EEGNet_pipeline"] = net

        if os.path.exists(csp_file):
            pretrained_models[f"cluster_{c_id}_CSP_LDA_pipeline"] = joblib.load(csp_file)

        if os.path.exists(cov_file):
            pretrained_models[f"cluster_{c_id}_Cov_Tangent_Space_LR_pipeline"] = joblib.load(cov_file)

    # 4. Submodular Top-5 Models
    sub_top5_dir = os.path.join(models_dir, "submodular_top5_pooled")
    if os.path.exists(sub_top5_dir):
        atc_file = os.path.join(sub_top5_dir, "ATCNet_submodular_top5_model.pt")
        eeg_file = os.path.join(sub_top5_dir, "EEGNet_submodular_top5_model.pt")
        csp_file = os.path.join(sub_top5_dir, "CSP_LDA_submodular_top5_model.pkl")
        cov_file = os.path.join(sub_top5_dir, "Cov_Tangent_Space_LR_submodular_top5_model.pkl")

        if os.path.exists(atc_file):
            net = ATCNet(n_classes=2, channels=num_channels, samples=num_samples)
            net.load_state_dict(torch.load(atc_file, map_location="cpu", weights_only=True))
            pretrained_models["submodular_top5_ATCNet_pipeline"] = net

        if os.path.exists(eeg_file):
            net = EEGNet(n_classes=2, channels=num_channels, samples=num_samples)
            net.load_state_dict(torch.load(eeg_file, map_location="cpu", weights_only=True))
            pretrained_models["submodular_top5_EEGNet_pipeline"] = net

        if os.path.exists(csp_file):
            pretrained_models["submodular_top5_CSP_LDA_pipeline"] = joblib.load(csp_file)

        if os.path.exists(cov_file):
            pretrained_models["submodular_top5_Cov_Tangent_Space_LR_pipeline"] = joblib.load(cov_file)

    print(f"Loaded {len(pretrained_models)} Session 0 pre-trained models for simulation.")

    # Load cluster assignments for strict LOSO filtering
    cluster_map = {}
    cluster_csv = os.path.join("graph_results", "clustering", args.dataset, "subject_cluster_assignments.csv")
    if os.path.exists(cluster_csv):
        cdf = pd.read_csv(cluster_csv)
        cluster_col = "Donor_Cluster" if "Donor_Cluster" in cdf.columns else "Cluster"
        for c_id in cdf[cluster_col].unique():
            cluster_map[int(c_id)] = set([int(s) for s in cdf[cdf[cluster_col] == c_id]['Subject']])

    subject_ids = list(range(1, 52))

    for target_subject in subject_ids:
        all_sub_results = []
        target_models = {}

        for m_name, m_obj in pretrained_models.items():
            if not m_name.startswith("subject_"):
                if args.strict_loso and m_name.startswith("cluster_"):
                    try:
                        c_id = int(m_name.split("_")[1])
                        if target_subject in cluster_map.get(c_id, set()):
                            continue
                    except Exception:
                        pass
                target_models[m_name] = m_obj

        # Add matching single subject Day 1 model (Strategy C - Matched Subject)
        for p_type in ["ATCNet", "EEGNet", "CSP_LDA", "Cov_Tangent_Space_LR"]:
            m_key = f"subject_{target_subject}_{p_type}_pipeline"
            if m_key in pretrained_models:
                target_models[m_key] = pretrained_models[m_key]

        # Add 5 random mismatched single subject donor models
        for p_type in ["ATCNet", "EEGNet"]:
            mismatched_keys = [k for k in pretrained_models if k.startswith("subject_") and k.endswith(f"_{p_type}_pipeline") and not k.startswith(f"subject_{target_subject}_")]
            if mismatched_keys:
                rng = np.random.RandomState(target_subject)
                chosen = rng.choice(mismatched_keys, size=min(5, len(mismatched_keys)), replace=False)
                for idx, m_key in enumerate(chosen):
                    target_models[f"subject_mismatched_{idx+1}_{p_type}_pipeline"] = pretrained_models[m_key]

        # 1. Simulate Session 1 (Day 2)
        try:
            X_s1, y_s1 = load_subject_eeg_session_data(args.dataset, target_subject, selected_channels, session_filter="1")
            res_s1 = simulate_session_trials(X_s1, y_s1, target_models, args.dataset, target_subject, "Day2", adapt_interval=args.adapt_interval, device=args.device, adapt_mode=args.adapt_mode, max_buffer=args.max_buffer)
            all_sub_results.extend(res_s1)
        except Exception as e:
            print(f"  Warning: Session 1 (Day 2) simulation failed for subject {target_subject}: {e}")

        # 2. Simulate Session 2 (Day 3)
        try:
            X_s2, y_s2 = load_subject_eeg_session_data(args.dataset, target_subject, selected_channels, session_filter="2")
            res_s2 = simulate_session_trials(X_s2, y_s2, target_models, args.dataset, target_subject, "Day3", adapt_interval=args.adapt_interval, device=args.device, adapt_mode=args.adapt_mode, max_buffer=args.max_buffer)
            all_sub_results.extend(res_s2)
        except Exception as e:
            print(f"  Warning: Session 2 (Day 3) simulation failed for subject {target_subject}: {e}")

        if all_sub_results:
            df_sub = pd.DataFrame(all_sub_results)
            out_dir = os.path.join("simulation_results", "cross_session", args.dataset)
            os.makedirs(out_dir, exist_ok=True)
            df_sub.to_csv(os.path.join(out_dir, f"{target_subject}.csv"), index=False)
            print(f"Saved Subject {target_subject:2d} cross-session results ({len(df_sub)} rows) to: {out_dir}")

    print("\n================================================================================")
    print(f" Cross-Session Simulation Complete! Saved to simulation_results/cross_session/{args.dataset}/")
    print("================================================================================\n")


if __name__ == "__main__":
    main()
