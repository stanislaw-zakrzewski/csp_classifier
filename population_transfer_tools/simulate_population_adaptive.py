"""
Population Transfer Adaptive Simulation Engine
================================================

Runs online trial-by-trial adaptive simulation EXCLUSIVELY on the 5 held-out test subjects
using Population Models trained on N - 5 subjects (Upper Theoretical Bound of Cross-Subject Transfer).
"""

import os
import sys
import json
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
from eegnet_tools.train_eegnet_models import load_subject_eeg_data


def simulate_population_trials(
    X: np.ndarray,
    labels: list[str],
    pretrained_models: dict[str, object],
    dataset_name: str,
    subject_id: int,
    adapt_interval: int = 8,
    device: str = "auto",
    adapt_mode: str = "head_only",
    max_buffer: int = 64,
    fine_tune_lr: float = 1e-4
) -> pd.DataFrame:
    """Simulate online adaptive classification for a held-out test subject."""
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

    df_results = pd.DataFrame(results)
    out_dir = os.path.join("simulation_results", "population_transfer", dataset_name)
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{subject_id}.csv")
    df_results.to_csv(out_file, index=False)
    return df_results


def main():
    parser = argparse.ArgumentParser(description="Simulate Population Transfer Adaptive Classification on Held-Out Test Subjects.")
    parser.add_argument("--dataset", "-d", default="Dreyer2023", help="Dataset name. Default: Dreyer2023")
    parser.add_argument("--models-dir", "-m", default=None, help="Directory containing Population pre-trained checkpoints.")
    parser.add_argument("--adapt-interval", "-i", type=int, default=8, help="Online adaptation trial interval. Default: 8")
    parser.add_argument("--adapt-mode", choices=["head_only", "full_model"], default="head_only", help="Adaptation mode. Default: head_only")
    parser.add_argument("--max-buffer", type=int, default=64, help="Maximum sliding window buffer size. Default: 64")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", help="Execution device. Default: auto")

    args = parser.parse_args()
    selected_channels = ["C1", "C2", "C5", "C3", "C4", "C6", "FC3", "CP3", "FC4", "CP4", "Cz"]
    models_dir = args.models_dir or os.path.join("trained_pipelines", "population_transfer", args.dataset)

    split_file = os.path.join(models_dir, "held_out_subjects.json")
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"Held-out split file '{split_file}' not found. Run train_population_models.py first.")

    with open(split_file, "r") as f:
        split_info = json.load(f)

    held_out_subjects = split_info['held_out_subjects']
    device_str = "cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())) else "cpu"

    print("================================================================================")
    print(" Population Transfer Upper-Limit Adaptive Simulation Engine")
    print("================================================================================")
    print(f" Dataset             : {args.dataset}")
    print(f" Checkpoints Dir     : {models_dir}")
    print(f" Held-Out Test Subs  : {held_out_subjects} ({len(held_out_subjects)} subs)")
    print(f" Population Donors   : {split_info['training_count']} donor subjects")
    print(f" Adapt Mode          : {args.adapt_mode}")
    print(f" Max Buffer          : {args.max_buffer}")
    print(f" Device             : {device_str.upper()}")
    print("================================================================================\n")

    # Retrieve exact training dimensions from split_info
    sample_X, _ = load_subject_eeg_data(args.dataset, held_out_subjects[0], channels=selected_channels)
    num_channels = split_info.get('num_channels', sample_X.shape[1])
    num_samples = split_info.get('num_samples', sample_X.shape[2])

    # Load Population Models
    pretrained_models = {}

    # Scratch Baselines
    pretrained_models["baseline_scratch_ATCNet"] = ATCNet(n_classes=2, channels=num_channels, samples=num_samples)
    pretrained_models["baseline_scratch_EEGNet"] = EEGNet(n_classes=2, channels=num_channels, samples=num_samples)

    # Population Models
    atc_pop = os.path.join(models_dir, "ATCNet_population_model.pt")
    eeg_pop = os.path.join(models_dir, "EEGNet_population_model.pt")
    csp_pop = os.path.join(models_dir, "CSP_LDA_population_model.pkl")
    cov_pop = os.path.join(models_dir, "Cov_Tangent_Space_LR_population_model.pkl")

    if os.path.exists(atc_pop):
        net = ATCNet(n_classes=2, channels=num_channels, samples=num_samples)
        net.load_state_dict(torch.load(atc_pop, map_location="cpu", weights_only=True))
        pretrained_models["population_ATCNet_pipeline"] = net

    if os.path.exists(eeg_pop):
        net = EEGNet(n_classes=2, channels=num_channels, samples=num_samples)
        net.load_state_dict(torch.load(eeg_pop, map_location="cpu", weights_only=True))
        pretrained_models["population_EEGNet_pipeline"] = net

    if os.path.exists(csp_pop):
        pretrained_models["population_CSP_LDA_pipeline"] = joblib.load(csp_pop)

    if os.path.exists(cov_pop):
        pretrained_models["population_Cov_Tangent_Space_LR_pipeline"] = joblib.load(cov_pop)

    print(f"Loaded {len(pretrained_models)} population models for held-out simulation.")

    for target_subject in held_out_subjects:
        try:
            X_target, y_target = load_subject_eeg_data(args.dataset, target_subject, channels=selected_channels)
            if X_target.shape[2] > num_samples:
                X_target = X_target[:, :, :num_samples]
            elif X_target.shape[2] < num_samples:
                pad_w = num_samples - X_target.shape[2]
                X_target = np.pad(X_target, ((0,0), (0,0), (0, pad_w)), mode='edge')

            print(f"Simulating held-out test subject {target_subject:2d} ({len(y_target)} trials, {len(pretrained_models)} models)...", flush=True)
            simulate_population_trials(
                X_target, y_target, pretrained_models, args.dataset, target_subject,
                adapt_interval=args.adapt_interval, device=args.device, adapt_mode=args.adapt_mode, max_buffer=args.max_buffer
            )
        except Exception as e:
            print(f"  Warning: Simulation failed for test subject {target_subject}: {e}")

    print("\n================================================================================")
    print(f" Held-Out Simulation Complete! Saved CSVs to: simulation_results/population_transfer/{args.dataset}/")
    print("================================================================================\n")


if __name__ == "__main__":
    main()
