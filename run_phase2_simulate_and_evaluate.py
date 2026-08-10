"""
Phase 2: Online Adaptive Simulation & Out-of-Sample Evaluation
================================================================

Simulates online adaptive classification strictly on the 20% held-out test split for Strategy C,
evaluates benchmark metrics, and generates updated Grand Mean plots.
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
    print(" PHASE 2: ONLINE ADAPTIVE SIMULATION & OUT-OF-SAMPLE EVALUATION (20% TEST SPLIT)")
    print("================================================================================\n")

    for ds in datasets:
        print(f"\n---> [{ds}] Running ATCNet Adaptive Simulation (20% test split)...")
        subprocess.run([sys.executable, "atcnet_tools/simulate_atcnet_adaptive.py", "--dataset", ds], check=True)

        print(f"---> [{ds}] Running EEGNet Adaptive Simulation (20% test split)...")
        subprocess.run([sys.executable, "eegnet_tools/simulate_eegnet_adaptive.py", "--dataset", ds], check=True)

        print(f"---> [{ds}] Evaluating ATCNet Benchmarks...")
        subprocess.run([sys.executable, "atcnet_tools/evaluate_atcnet_benchmarks.py", "--dataset", ds], check=True)

        print(f"---> [{ds}] Evaluating EEGNet Benchmarks...")
        subprocess.run([sys.executable, "eegnet_tools/evaluate_eegnet_benchmarks.py", "--dataset", ds], check=True)

    print("\n---> Regenerating Grand Mean Benchmark Trajectory Plots across all 7 datasets...")
    subprocess.run([sys.executable, "generate_grand_mean_benchmarks.py"], check=True)

    print("\n================================================================================")
    print(" PHASE 2 COMPLETE! ALL OUT-OF-SAMPLE SIMULATIONS & GRAND MEAN PLOTS GENERATED")
    print("================================================================================\n")


if __name__ == "__main__":
    main()
