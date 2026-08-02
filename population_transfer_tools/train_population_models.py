"""
Population Upper-Limit Pre-Training Engine
===========================================

Pre-trains PyTorch ATCNet, PyTorch EEGNet, CSP+LDA, and Covariance Tangent Space LR models
on ALL available subjects EXCEPT 5 randomly selected held-out test subjects (N - 5 donor pool).
"""

import os
import sys
import json
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

from eegnet_tools.eegnet_model import EEGNet
from atcnet_tools.atcnet_model import ATCNet
from eegnet_tools.train_eegnet_models import load_subject_eeg_data
from cross_session_tools.train_cross_session_models import (
    build_classical_csp_lda_pipeline,
    build_classical_cov_lr_pipeline,
    train_pytorch_model
)


def main():
    parser = argparse.ArgumentParser(description="Pre-train Population Upper-Limit Models (N - 5 Donors).")
    parser.add_argument("--dataset", "-d", default="Dreyer2023", help="Dataset name. Default: Dreyer2023")
    parser.add_argument("--held-out-count", type=int, default=5, help="Number of held-out test subjects. Default: 5")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for held-out selection. Default: 42")
    parser.add_argument("--epochs", "-e", type=int, default=120, help="Pre-training epochs. Default: 120")
    parser.add_argument("--batch-size", "-b", type=int, default=128, help="Batch size. Default: 128")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", help="Execution device. Default: auto")

    args = parser.parse_args()

    device_str = "cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())) else "cpu"
    selected_channels = ["C1", "C2", "C5", "C3", "C4", "C6", "FC3", "CP3", "FC4", "CP4", "Cz"]
    output_dir = os.path.join("trained_pipelines", "population_transfer", args.dataset)
    os.makedirs(output_dir, exist_ok=True)

    # Determine total subject count dynamically
    import moabb.datasets as mb_ds
    dataset_cls = getattr(mb_ds, args.dataset)
    ds_obj = dataset_cls()
    all_subject_ids = sorted(ds_obj.subject_list)

    # Randomly select 5 held-out subjects
    rng = np.random.RandomState(args.seed)
    held_out_subjects = sorted([int(s) for s in rng.choice(all_subject_ids, size=min(args.held_out_count, len(all_subject_ids)), replace=False)])
    training_subjects = sorted([int(s) for s in all_subject_ids if s not in held_out_subjects])

    # Save held-out subjects JSON
    split_info = {
        'dataset': args.dataset,
        'seed': args.seed,
        'total_subjects': len(all_subject_ids),
        'held_out_count': len(held_out_subjects),
        'held_out_subjects': held_out_subjects,
        'training_count': len(training_subjects),
        'training_subjects': training_subjects
    }
    with open(os.path.join(output_dir, "held_out_subjects.json"), "w") as f:
        json.dump(split_info, f, indent=4)

    print("================================================================================")
    print(" Population Transfer Upper-Limit Pre-Training Engine")
    print("================================================================================")
    print(f" Dataset             : {args.dataset}")
    print(f" Total Subjects      : {len(all_subject_ids)}")
    print(f" Held-Out Test Subs  : {held_out_subjects} ({len(held_out_subjects)} subs)")
    print(f" Population Donors   : {len(training_subjects)} donor subjects")
    print(f" Pre-Training Epochs : {args.epochs}")
    print(f" Batch Size          : {args.batch_size}")
    print(f" Device             : {device_str.upper()}")
    print(f" Output Directory    : {output_dir}")
    print("================================================================================\n")

    # Load and pool data across all N - 5 donor subjects
    print(f"Loading EEG data across {len(training_subjects)} population donor subjects...")
    donor_X_list = []
    donor_y_list = []

    for sub in training_subjects:
        try:
            X_sub, y_sub = load_subject_eeg_data(args.dataset, sub, channels=selected_channels)
            donor_X_list.append(X_sub)
            donor_y_list.append(y_sub)
        except Exception as e:
            print(f"  Warning: Could not load data for donor subject {sub}: {e}")

    # Uniformly align time samples across donor arrays
    min_samples = min([x.shape[2] for x in donor_X_list])
    donor_X_list = [x[:, :, :min_samples] for x in donor_X_list]

    X_pop = np.concatenate(donor_X_list, axis=0)
    y_pop = np.concatenate(donor_y_list, axis=0)
    num_channels, num_samples = X_pop.shape[1], X_pop.shape[2]

    # Save complete split_info with num_samples
    split_info['num_channels'] = int(num_channels)
    split_info['num_samples'] = int(num_samples)
    with open(os.path.join(output_dir, "held_out_subjects.json"), "w") as f:
        json.dump(split_info, f, indent=4)

    print(f"\nPopulation Donor Pool Loaded: {len(y_pop)} trials | Input Shape: {X_pop.shape}")

    records = []

    # 1. Pre-Train Population ATCNet
    print("\n--- Pre-Training Population ATCNet (120 Epochs) ---")
    atc = ATCNet(n_classes=2, channels=num_channels, samples=num_samples)
    atc, acc_atc = train_pytorch_model(atc, X_pop, y_pop, epochs=args.epochs, batch_size=args.batch_size, device_str=device_str)
    torch.save(atc.state_dict(), os.path.join(output_dir, "ATCNet_population_model.pt"))
    records.append({'Model': 'ATCNet', 'Donor_Count': len(training_subjects), 'Trial_Count': len(y_pop), 'PreTrain_Acc': acc_atc})

    # 2. Pre-Train Population EEGNet
    print("--- Pre-Training Population EEGNet (120 Epochs) ---")
    eeg = EEGNet(n_classes=2, channels=num_channels, samples=num_samples)
    eeg, acc_eeg = train_pytorch_model(eeg, X_pop, y_pop, epochs=args.epochs, batch_size=args.batch_size, device_str=device_str)
    torch.save(eeg.state_dict(), os.path.join(output_dir, "EEGNet_population_model.pt"))
    records.append({'Model': 'EEGNet', 'Donor_Count': len(training_subjects), 'Trial_Count': len(y_pop), 'PreTrain_Acc': acc_eeg})

    # 3. Pre-Train Population CSP+LDA
    print("--- Fitting Population CSP + LDA ---")
    csp = build_classical_csp_lda_pipeline(num_channels)
    csp.fit(X_pop, y_pop)
    joblib.dump(csp, os.path.join(output_dir, "CSP_LDA_population_model.pkl"))

    # 4. Pre-Train Population Covariance Tangent Space LR
    print("--- Fitting Population Covariance Tangent Space LR ---")
    cov = build_classical_cov_lr_pipeline()
    cov.fit(X_pop, y_pop)
    joblib.dump(cov, os.path.join(output_dir, "Cov_Tangent_Space_LR_population_model.pkl"))

    df_summary = pd.DataFrame(records)
    summary_path = os.path.join(output_dir, "pretraining_accuracies.csv")
    df_summary.to_csv(summary_path, index=False)

    print("\n================================================================================")
    print(f" Population Pre-Training Complete! Saved Summary to: {summary_path}")
    print("================================================================================\n")


if __name__ == "__main__":
    main()
