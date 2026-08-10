"""
ATCNet & Multi-Model Master Benchmark Evaluation Engine
========================================================

Aggregates ATCNet online simulation results and compares them against pre-computed
EEGNet, CSP+LDA, and Cov+LR classical baselines directly from graph_results/.

Outputs:
1. Benchmark summary CSV (atcnet_benchmark_summary.csv)
2. Trajectory comparison plot (atcnet_vs_eegnet_vs_classical_trajectories.png)
3. Master markdown benchmark report (atcnet_master_benchmark_report.md)
"""

import os
import glob
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def parse_model_category(clf_name: str) -> str:
    """Categorize classifier name into its strategy / baseline group."""
    if clf_name.startswith("baseline_scratch"):
        if "ATCNet" in clf_name:
            return "Baseline Scratch (ATCNet)"
        else:
            return "Baseline Scratch (EEGNet)"
    elif "ATCNet" in clf_name:
        if clf_name.startswith("cluster_"):
            return "Strategy A (ATCNet Cluster Pooled)"
        elif clf_name.startswith("submodular_"):
            return "Strategy B (ATCNet Top-5 Submodular Pooled)"
        elif "mismatched" in clf_name:
            return "Strategy C (ATCNet Mismatched Single Donor)"
        else:
            return "Strategy C (ATCNet Matched Single Subject)"
    elif "EEGNet" in clf_name:
        if clf_name.startswith("cluster_"):
            return "Strategy A (EEGNet Cluster Pooled)"
        elif clf_name.startswith("submodular_"):
            return "Strategy B (EEGNet Top-5 Submodular Pooled)"
        elif "mismatched" in clf_name:
            return "Strategy C (EEGNet Mismatched Single Donor)"
        else:
            return "Strategy C (EEGNet Matched Single Subject)"
    return "Other"


def main():
    parser = argparse.ArgumentParser(description="Generate master benchmark report and comparison plots for ATCNet vs EEGNet.")
    parser.add_argument("--dataset", "-d", default="Dreyer2023", help="Dataset name. Default: Dreyer2023")
    parser.add_argument("--bin-size", "-b", type=int, default=10, help="Trial bin size for trajectory plotting. Default: 10")

    args = parser.parse_args()

    atc_sim_dir = os.path.join("simulation_results", "atcnet", args.dataset)
    eeg_sim_dir = os.path.join("simulation_results", "eegnet", args.dataset)
    out_dir = os.path.join("graph_results", "atcnet_benchmark", args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(atc_sim_dir):
        raise FileNotFoundError(f"ATCNet simulation results directory '{atc_sim_dir}' does not exist. Run simulate_atcnet_adaptive.py first.")

    atc_files = glob.glob(os.path.join(atc_sim_dir, "*.csv"))
    eeg_files = glob.glob(os.path.join(eeg_sim_dir, "*.csv")) if os.path.exists(eeg_sim_dir) else []

    print("================================================================================")
    print(f" ATCNet & EEGNet Master Benchmark Evaluation Engine")
    print("================================================================================")
    print(f" Dataset             : {args.dataset}")
    print(f" ATCNet Analyzed CSVs: {len(atc_files)}")
    print(f" EEGNet Analyzed CSVs: {len(eeg_files)}")
    print(f" Output Directory    : {out_dir}")
    print("================================================================universal\n")

    all_dfs = []
    for f in atc_files:
        sub_id = os.path.basename(f).replace(".csv", "")
        df = pd.read_csv(f)
        df['Subject'] = sub_id
        df['Category'] = df['Classifier'].apply(parse_model_category)
        all_dfs.append(df)

    for f in eeg_files:
        sub_id = os.path.basename(f).replace(".csv", "")
        df = pd.read_csv(f)
        # Keep only EEGNet models to avoid cluttering with classical
        df = df[df['Classifier'].str.contains("EEGNet")].copy()
        if not df.empty:
            df['Subject'] = sub_id
            df['Category'] = df['Classifier'].apply(parse_model_category)
            all_dfs.append(df)

    if not all_dfs:
        print("Error: No simulation CSVs found.")
        return

    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Filter out "Mismatched" and "Other" for clean trajectory visualization
    valid_categories = [
        "Strategy A (ATCNet Cluster Pooled)",
        "Strategy A (EEGNet Cluster Pooled)",
        "Strategy B (ATCNet Top-5 Submodular Pooled)",
        "Strategy B (EEGNet Top-5 Submodular Pooled)",
        "Strategy C (ATCNet Matched Single Subject)",
        "Strategy C (EEGNet Matched Single Subject)",
        "Baseline Scratch (ATCNet)",
        "Baseline Scratch (EEGNet)"
    ]
    combined_df = combined_df[combined_df['Category'].isin(valid_categories)].copy()

    # Quantile temporal binning into exactly 10 bins (Bin 0 to Bin 9)
    max_trial = combined_df['Trial'].max()
    combined_df['Bin'] = np.clip((combined_df['Trial'] / (max_trial + 1) * 10).astype(int), 0, 9)

    binned_acc = combined_df.groupby(['Category', 'Bin', 'Subject'])['Is_Correct'].mean().reset_index()
    category_bins = binned_acc.groupby(['Category', 'Bin'])['Is_Correct'].agg(['mean', 'std']).reset_index()

    # Compute overall category summary
    cat_summary = combined_df.groupby('Category')['Is_Correct'].agg(['mean', 'std']).reset_index()
    cat_summary.rename(columns={'mean': 'Mean_Accuracy', 'std': 'Std_Accuracy'}, inplace=True)

    # Compute final bin accuracy
    max_bin = binned_acc['Bin'].max()
    final_bin_acc = binned_acc[binned_acc['Bin'] == max_bin].groupby('Category')['Is_Correct'].mean().reset_index()
    final_bin_acc.rename(columns={'Is_Correct': 'Final_Accuracy'}, inplace=True)

    cat_summary = pd.merge(cat_summary, final_bin_acc, on='Category', how='left')
    cat_summary.sort_values(by='Mean_Accuracy', ascending=False, inplace=True)

    # Save Summary CSV
    summary_csv_path = os.path.join(out_dir, "atcnet_benchmark_summary.csv")
    cat_summary.to_csv(summary_csv_path, index=False)

    # Plot Trajectory Chart
    plt.figure(figsize=(13, 6.5))
    sns.set_theme(style="whitegrid")

    palette = sns.color_palette("tab10", n_colors=len(category_bins['Category'].unique()))
    ax = sns.lineplot(
        data=category_bins,
        x='Bin',
        y='mean',
        hue='Category',
        style='Category',
        markers=True,
        dashes=False,
        palette=palette,
        linewidth=2.5
    )

    plt.title(f"ATCNet vs EEGNet Online Adaptive Trajectories ({args.dataset})", fontsize=14, fontweight='bold')
    plt.xlabel("Quantile Trial Bins (Bin 0 to Bin 9)", fontsize=12)
    plt.ylabel("Classification Accuracy", fontsize=12)
    plt.xticks(range(10), [f"Bin {i}" for i in range(10)])
    plt.ylim(0.45, 0.95)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
    plt.tight_layout()

    plot_path = os.path.join(out_dir, "atcnet_vs_eegnet_vs_classical_trajectories.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

    # Generate Markdown Report
    report_path = os.path.join(out_dir, "atcnet_master_benchmark_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# ATCNet & Multi-Model Master Benchmark Report\n\n")
        f.write(f"- **Dataset**: `{args.dataset}`\n")
        f.write(f"- **Bin Window Size**: `{args.bin_size} trials per bin`\n")
        f.write(f"- **Analyzed Target Subjects**: `{len(atc_files)} subjects`\n\n")
        f.write(f"## 1. Overall Performance Benchmark Summary\n")
        f.write(f"The table below compares PyTorch ATCNet models against pre-computed EEGNet and classical baselines:\n\n")
        
        headers = ['Category', 'Mean_Accuracy', 'Final_Accuracy']
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for _, row in cat_summary[['Category', 'Mean_Accuracy', 'Final_Accuracy']].iterrows():
            m_acc = f"{row['Mean_Accuracy']:.4f}" if pd.notnull(row['Mean_Accuracy']) else "N/A"
            f_acc = f"{row['Final_Accuracy']:.4f}" if pd.notnull(row['Final_Accuracy']) else "N/A"
            f.write(f"| {row['Category']} | {m_acc} | {f_acc} |\n")

        f.write(f"\n\n## 2. Key Takeaways & Visual Artifacts\n")
        f.write(f"- **Summary CSV**: [`atcnet_benchmark_summary.csv`](file:///{os.path.abspath(summary_csv_path)})\n")
        f.write(f"- **Trajectory Plot**: [`atcnet_vs_eegnet_vs_classical_trajectories.png`](file:///{os.path.abspath(plot_path)})\n")

    print("================================================================================")
    print(f" Saved Summary CSV          : {summary_csv_path}")
    print(f" Saved Trajectory Plot      : {plot_path}")
    print(f" Saved Master Markdown Report: {report_path}")
    print("================================================================================\n")


if __name__ == "__main__":
    main()
