"""
Master Pipeline Runner for Experiment 10: Longitudinal Multi-Session & DRY/WET Analysis
=======================================================================================

Runs end-to-end:
1. Detects available sessions in `longitudal_dataset/`.
2. Loops through training split combinations k in 1..N-1.
3. Evaluates all 6 models across modality pairs (DRY->DRY, WET->WET, WET->DRY).
4. Generates static & adaptive simulation metrics.
5. Produces plots in `graph_results/exp10_longitudinal/` and `EXPERIMENT_10_LONGITUDINAL_REPORT.md`.
"""

import os
import sys

# Append project root directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from longitudinal_tools.longitudinal_dataset_loader import (
    get_available_sessions,
    get_k_session_split
)
from longitudinal_tools.train_longitudinal_models import train_all_longitudinal_models
from longitudinal_tools.simulate_longitudinal_adaptive import run_full_session_evaluations
from longitudinal_tools.evaluate_longitudinal_benchmarks import run_evaluation_and_plotting


def main():
    print("================================================================================")
    print(" EXPERIMENT 10: LONGITUDINAL MULTI-SESSION & DRY/WET ELECTRODE EVALUATION")
    print("================================================================================\n")

    data_dir = "longitudal_dataset"
    sessions = get_available_sessions(data_dir)
    n_sessions = len(sessions)

    print(f"---> Found {n_sessions} session(s) in '{data_dir}': {sessions}")
    if n_sessions < 2:
        print("[ERROR] Experiment 10 requires at least 2 sessions in 'longitudal_dataset/'. Exiting.")
        sys.exit(1)

    # Modality combinations to evaluate
    modality_pairs = [
        ("DRY", "DRY"),
        ("WET", "WET"),
        ("WET", "DRY")  # Cross-modal transfer
    ]

    for k in range(1, n_sessions):
        print(f"\n=================== EVALUATING k = {k} TRAINING SESSIONS ===================")
        for train_mod, test_mod in modality_pairs:
            print(f"\n---> Training Modality: {train_mod} | Test Modality: {test_mod} (k={k})")

            # 1. Load data split
            X_tr, y_tr, test_dict = get_k_session_split(
                k=k,
                train_modality=train_mod,
                test_modality=test_mod,
                data_dir=data_dir
            )
            print(f"     Train samples: {X_tr.shape[0]}, Test sessions: {list(test_dict.keys())}")

            # 2. Train all 6 approaches
            print("     Training/Fine-tuning all 6 approaches...")
            models_dict = train_all_longitudinal_models(X_tr, y_tr, k_sessions=k)

            # 3. Simulate static & adaptive on test sessions
            print("     Running Static & Adaptive Online simulations on test sessions...")
            run_full_session_evaluations(
                models_dict=models_dict,
                test_sessions_dict=test_dict,
                k_sessions=k,
                train_modality=train_mod,
                test_modality=test_mod
            )

    print("\n================================================================================")
    print(" GENERATING MASTER BENCHMARK PLOTS & REPORT FOR EXPERIMENT 10")
    print("================================================================================\n")
    run_evaluation_and_plotting()

    print("\n================================================================================")
    print(" EXPERIMENT 10 COMPLETED SUCCESSFULLY!")
    print(" Results saved to 'simulation_results/longitudinal/'")
    print(" Plots saved to 'graph_results/exp10_longitudinal/'")
    print(" Report saved to 'EXPERIMENT_10_LONGITUDINAL_REPORT.md'")
    print("================================================================================\n")


if __name__ == "__main__":
    main()
