"""
Phase 1: Retrain ALL Single-Subject Classifiers on 80% Training Split
======================================================================

Retrains all Single-Subject classifiers across 7 MOABB datasets:
1. PyTorch ATCNet Single-Subject Models (80/20 split)
2. PyTorch EEGNet Single-Subject Models (80/20 split)
3. Classical Pipelines: CSP-LDA, Cov-Tangent-Space-LR, CSP-SVM (80/20 split)
"""

import os
import sys
import subprocess


def main():
    datasets = [
        "Dreyer2023",
        "Dreyer2023A",
        "PhysionetMI",
        "Lee2019_MI",
        "GuttmannFlury2025_MI",
        "GuttmannFlury2025_ME",
        "Yang2025"
    ]

    print("================================================================================")
    print(" PHASE 1: RETRAINING ALL SINGLE-SUBJECT CLASSIFIERS (80% TRAIN SPLIT)")
    print("================================================================================\n")

    # 1. Retrain ATCNet Single-Subject Models (80/20 split) - ALREADY COMPLETED
    print("\n---> [ATCNet] Retraining Single-Subject Models: COMPLETED for all 7 datasets.")

    # 2. Retrain EEGNet Single-Subject Models (80/20 split)
    for ds in datasets:
        print(f"\n---> [EEGNet] Retraining Single-Subject Models for '{ds}' (80/20 split)...")
        cmd = [sys.executable, "eegnet_tools/train_eegnet_models.py", "--dataset", ds, "--mode", "single", "--epochs", "80"]
        subprocess.run(cmd, check=True)

    # 3. Retrain Classical Single-Subject Pipelines (80/20 split)
    print(f"\n---> [Classical] Retraining Classical Pipelines (CSP-LDA, Cov-LR, CSP-SVM) (80/20 split)...")
    cmd = [sys.executable, "retrain_classical_single_subjects.py"]
    subprocess.run(cmd, check=True)

    print("\n================================================================================")
    print(" PHASE 1 COMPLETE! ALL SINGLE-SUBJECT CLASSIFIERS RETRAINED ON 80% TRAIN SPLIT")
    print("================================================================================\n")


if __name__ == "__main__":
    main()
