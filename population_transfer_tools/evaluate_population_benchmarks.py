"""
Population Transfer Master Benchmark Evaluation Engine
========================================================

Aggregates simulation results for held-out test subjects and compares:
1. Population ATCNet (N - 5 Donors Upper Limit)
2. GNN Cluster Pooled ATCNet (~50 Donors)
3. Submodular Top-5 Pooled ATCNet (5 Donors)
4. Population EEGNet, CSP+LDA, and Cov+LR
5. Scratch Cold-Start Baselines
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def categorize_population_classifier(name: str) -> str:
    """Categorize classifier string into clean strategy names."""
    if "population_ATCNet" in name:
        return "Strategy A (Population ATCNet - N-5 Donors Upper Limit)"
    if "population_EEGNet" in name:
        return "Strategy A (Population EEGNet - N-5 Donors Upper Limit)"
    if "population_CSP_LDA" in name:
        return "Baseline 3 (Population CSP+LDA - N-5 Donors)"
    if "population_Cov_Tangent" in name:
        return "Baseline 5 (Population Cov Tangent LR - N-5 Donors)"

    if "baseline_scratch_ATCNet" in name:
        return "Baseline 1 (Scratch ATCNet Cold-Start)"
    if "baseline_scratch_EEGNet" in name:
        return "Baseline 1 (Scratch EEGNet Cold-Start)"

    return name


def main():
    parser = argparse.ArgumentParser(description="Evaluate Population Transfer Upper-Limit Master Benchmarks.")
    parser.add_argument("--dataset", "-d", default="Dreyer2023", help="Dataset name. Default: Dreyer2023")
    parser.add_argument("--bin-size", "-b", type=int, default=24, help="Bin size for trajectory plotting. Default: 24")

    args = parser.parse_args()

    sim_dir = os.path.join("simulation_results", "population_transfer", args.dataset)
    csv_files = glob.glob(os.path.join(sim_dir, "*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No population simulation CSV files found in '{sim_dir}'. Run simulate_population_adaptive.py first.")

    out_dir = os.path.join("graph_results", "population_transfer_benchmark", args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    print("================================================================================")
    print(" Population Transfer Upper-Limit Benchmark Evaluation Engine")
    print("================================================================================")
    print(f" Dataset             : {args.dataset}")
    print(f" Test Subjects       : {len(csv_files)} held-out test subjects")
    print(f" Output Directory    : {out_dir}")
    print("================================================================================\n")

    dfs = [pd.read_csv(f) for f in csv_files]
    full_df = pd.concat(dfs, ignore_index=True)
    full_df['Category'] = full_df['Classifier'].apply(categorize_population_classifier)

    # Compute overall category performance summary
    summary_rows = []
    for cat, grp in full_df.groupby('Category'):
        mean_acc = grp['Is_Correct'].mean()
        std_acc = grp.groupby('Subject')['Is_Correct'].mean().std()
        max_trial = grp['Trial'].max()
        final_acc = grp[grp['Trial'] >= (max_trial - 20)]['Is_Correct'].mean()
        summary_rows.append({
            'Category': cat,
            'Mean_Accuracy': mean_acc,
            'Std_Accuracy': std_acc,
            'Final_Accuracy': final_acc
        })

    summary_df = pd.DataFrame(summary_rows).sort_values(by='Mean_Accuracy', ascending=False)

    # Load GNN Cluster summary if available for side-by-side comparison
    gnn_summary_file = os.path.join("graph_results", "atcnet_benchmark", args.dataset, "atcnet_benchmark_summary.csv")
    if os.path.exists(gnn_summary_file):
        gnn_df = pd.read_csv(gnn_summary_file)

    # Save summary CSV
    summary_csv = os.path.join(out_dir, "population_benchmark_summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    # Plot binned trial trajectories
    full_df['Bin'] = full_df['Trial'] // args.bin_size
    bin_summary = full_df.groupby(['Category', 'Bin'])['Is_Correct'].mean().reset_index()

    plt.figure(figsize=(12, 7))
    styles = {
        'Strategy A (Population ATCNet - N-5 Donors Upper Limit)': ('#1f77b4', '-'),
        'Strategy A (Population EEGNet - N-5 Donors Upper Limit)': ('#2ca02c', '--'),
        'Baseline 3 (Population CSP+LDA - N-5 Donors)': ('#9467bd', ':'),
        'Baseline 5 (Population Cov Tangent LR - N-5 Donors)': ('#8c564b', ':'),
        'Baseline 1 (Scratch ATCNet Cold-Start)': ('#d62728', '-'),
        'Baseline 1 (Scratch EEGNet Cold-Start)': ('#ff9896', '--')
    }

    for cat in sorted(bin_summary['Category'].unique()):
        sub = bin_summary[bin_summary['Category'] == cat]
        color, ls = styles.get(cat, (None, '-'))
        plt.plot(sub['Bin'] * args.bin_size, sub['Is_Correct'], label=cat, color=color, linestyle=ls, linewidth=2.5)

    plt.title(f"Population Transfer Upper-Limit Trajectories ({args.dataset} Held-Out Test Subs)", fontsize=14, fontweight='bold')
    plt.xlabel(f"Trial Number ({args.bin_size}-Trial Bins)", fontsize=12)
    plt.ylabel("Classification Accuracy", fontsize=12)
    plt.ylim(0.40, 1.02)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.tight_layout()

    plot_png = os.path.join(out_dir, "population_upper_limit_trajectories.png")
    plt.savefig(plot_png, dpi=300)
    plt.close()

    # Write Master Markdown Report
    report_path = os.path.join(out_dir, "population_upper_limit_benchmark_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# {args.dataset} Population Transfer Upper-Limit Master Benchmark Report\n\n")
        f.write(f"- **Dataset**: `{args.dataset}`\n")
        f.write(f"- **Held-Out Test Subjects**: `{len(csv_files)} subjects`\n")
        f.write(f"- **Pre-Training Donor Pool**: `N - 5 Subjects`\n\n")
        f.write(f"## 1. Upper-Limit Performance Summary Table\n\n")
        f.write("| Category | Mean Accuracy | Final Accuracy |\n| --- | --- | --- |\n")
        for _, row in summary_df.iterrows():
            f.write(f"| {row['Category']} | {row['Mean_Accuracy']:.4f} | {row['Final_Accuracy']:.4f} |\n")

        f.write(f"\n\n## 2. Generated Visual Artifacts\n")
        f.write(f"- **Summary CSV**: [`population_benchmark_summary.csv`](file:///{os.path.abspath(summary_csv)})\n")
        f.write(f"- **Trajectory Plot**: [`population_upper_limit_trajectories.png`](file:///{os.path.abspath(plot_png)})\n")

    print("================================================================================")
    print(f" Saved Summary CSV          : {summary_csv}")
    print(f" Saved Trajectory Plot      : {plot_png}")
    print(f" Saved Master Markdown Report: {report_path}")
    print("================================================================================\n")


if __name__ == "__main__":
    main()
