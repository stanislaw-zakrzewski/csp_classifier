"""
Cross-Session Master Benchmark Evaluation Engine
=================================================

Aggregates simulation results from simulation_results/cross_session/Yang2025/ into:
1. Day 2 (Session 1) Performance Table
2. Day 3 (Session 2) Performance Table
3. Day 2 vs Day 3 Time-Decay Breakdown (Delta = Day3 - Day2)
4. Trajectory plots and master markdown benchmark report.
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def categorize_cross_session_classifier(name: str) -> str:
    """Categorize classifier string into clean experimental strategy names."""
    # Classical baselines first
    if "CSP_LDA" in name:
        if "cluster" in name:
            return "Baseline 3 (CSP+LDA Cluster Pooled Day 1 -> Day 2/3)"
        elif "submodular" in name:
            return "Baseline 4 (CSP+LDA Top-5 Submodular Day 1 -> Day 2/3)"
        elif "subject_" in name:
            return "Strategy C (CSP+LDA Matched Single Day 1 -> Day 2/3)"

    if "Cov_Tangent_Space_LR" in name:
        if "cluster" in name:
            return "Baseline 5 (Cov Tangent LR Cluster Pooled Day 1 -> Day 2/3)"
        elif "submodular" in name:
            return "Baseline 6 (Cov Tangent LR Top-5 Submodular Day 1 -> Day 2/3)"
        elif "subject_" in name:
            return "Strategy C (Cov Tangent LR Matched Single Day 1 -> Day 2/3)"

    if "baseline_scratch_ATCNet" in name:
        return "Baseline 1 (Scratch ATCNet Day 2/3)"

    if "baseline_scratch_EEGNet" in name:
        return "Baseline 1 (Scratch EEGNet Day 2/3)"

    if "submodular_top5_ATCNet" in name:
        return "Strategy B (ATCNet Top-5 Submodular Day 1 -> Day 2/3)"

    if "submodular_top5_EEGNet" in name:
        return "Strategy B (EEGNet Top-5 Submodular Day 1 -> Day 2/3)"

    if "cluster_" in name:
        if "ATCNet" in name:
            return "Strategy A (ATCNet Cluster Pooled Day 1 -> Day 2/3)"
        elif "EEGNet" in name:
            return "Strategy A (EEGNet Cluster Pooled Day 1 -> Day 2/3)"

    if "mismatched" in name:
        if "ATCNet" in name:
            return "Strategy C (ATCNet Mismatched Single Donor Day 1 -> Day 2/3)"
        elif "EEGNet" in name:
            return "Strategy C (EEGNet Mismatched Single Donor Day 1 -> Day 2/3)"

    if name.startswith("subject_"):
        if "ATCNet" in name:
            return "Strategy C (ATCNet Matched Single Day 1 -> Day 2/3)"
        elif "EEGNet" in name:
            return "Strategy C (EEGNet Matched Single Day 1 -> Day 2/3)"

    return name


def plot_session_trajectories(df: pd.DataFrame, title_suffix: str, out_png_path: str, bin_size: int = 20):
    """Plot binned trial trajectories across categories."""
    df_copy = df.copy()
    df_copy['Bin'] = df_copy['Trial'] // bin_size

    # Compute average accuracy per bin per category
    bin_summary = df_copy.groupby(['Category', 'Bin'])['Is_Correct'].mean().reset_index()

    plt.figure(figsize=(12, 7))
    categories = bin_summary['Category'].unique()

    styles = {
        'Strategy C (ATCNet Matched Single Day 1 -> Day 2/3)': ('#1f77b4', '-'),
        'Strategy C (EEGNet Matched Single Day 1 -> Day 2/3)': ('#aec7e8', '--'),
        'Strategy A (ATCNet Cluster Pooled Day 1 -> Day 2/3)': ('#2ca02c', '-'),
        'Strategy A (EEGNet Cluster Pooled Day 1 -> Day 2/3)': ('#98df8a', '--'),
        'Strategy B (ATCNet Top-5 Submodular Day 1 -> Day 2/3)': ('#ff7f0e', '-'),
        'Strategy B (EEGNet Top-5 Submodular Day 1 -> Day 2/3)': ('#ffbb78', '--'),
        'Baseline 1 (Scratch ATCNet Day 2/3)': ('#d62728', '-'),
        'Baseline 1 (Scratch EEGNet Day 2/3)': ('#ff9896', '--'),
        'Baseline 3 (CSP+LDA Cluster Pooled Day 1 -> Day 2/3)': ('#9467bd', ':'),
        'Baseline 5 (Cov Tangent LR Cluster Pooled Day 1 -> Day 2/3)': ('#8c564b', ':')
    }

    for cat in sorted(categories):
        sub = bin_summary[bin_summary['Category'] == cat]
        color, ls = styles.get(cat, (None, '-'))
        plt.plot(sub['Bin'] * bin_size, sub['Is_Correct'], label=cat, color=color, linestyle=ls, linewidth=2)

    plt.title(f"Yang2025 Multi-Day Cross-Session Trajectories: {title_suffix}", fontsize=14, fontweight='bold')
    plt.xlabel(f"Trial Number ({bin_size}-Trial Bins)", fontsize=12)
    plt.ylabel("Classification Accuracy", fontsize=12)
    plt.ylim(0.35, 1.02)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.tight_layout()
    plt.savefig(out_png_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate Cross-Session Master Benchmarks for Yang2025.")
    parser.add_argument("--dataset", "-d", default="Yang2025", help="Dataset name. Default: Yang2025")
    parser.add_argument("--model", "-m", default="all", help="Model family filter ('all', 'atcnet', 'eegnet', etc.). Default: all")
    parser.add_argument("--bin-size", "-b", type=int, default=20, help="Bin size for trajectory plotting. Default: 20")

    args = parser.parse_args()

    sim_dir = os.path.join("simulation_results", "cross_session", args.dataset)
    csv_files = glob.glob(os.path.join(sim_dir, "*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No simulation CSV files found in '{sim_dir}'. Run simulate_cross_session_adaptive.py first.")

    out_dir = os.path.join("graph_results", "cross_session_benchmark", args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    print("================================================================================")
    print(" Cross-Session Master Benchmark Evaluation Engine")
    print("================================================================================")
    print(f" Dataset             : {args.dataset}")
    print(f" Analyzed Subjects   : {len(csv_files)} subjects")
    print(f" Output Directory    : {out_dir}")
    print("================================================================================\n")

    dfs = [pd.read_csv(f) for f in csv_files]
    full_df = pd.concat(dfs, ignore_index=True)

    # Categorize model names
    full_df['Category'] = full_df['Classifier'].apply(categorize_cross_session_classifier)

    # Split data by Session Tag (Day 2 vs Day 3)
    df_day2 = full_df[full_df['Session_Tag'] == 'Day2']
    df_day3 = full_df[full_df['Session_Tag'] == 'Day3']

    def summarize_dataset(df_sub: pd.DataFrame):
        summary_rows = []
        for cat, grp in df_sub.groupby('Category'):
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
        return pd.DataFrame(summary_rows).sort_values(by='Mean_Accuracy', ascending=False)

    summary_day2 = summarize_dataset(df_day2)
    summary_day3 = summarize_dataset(df_day3)

    # Merge Day 2 and Day 3 summaries to compute time-decay Delta (Day3 - Day2)
    decay_df = pd.merge(summary_day2[['Category', 'Mean_Accuracy']], summary_day3[['Category', 'Mean_Accuracy']], on='Category', suffixes=('_Day2', '_Day3'))
    decay_df['Time_Decay_Delta'] = decay_df['Mean_Accuracy_Day3'] - decay_df['Mean_Accuracy_Day2']
    decay_df = decay_df.sort_values(by='Mean_Accuracy_Day2', ascending=False)

    # Save CSVs
    day2_csv = os.path.join(out_dir, "cross_session_day2_summary.csv")
    day3_csv = os.path.join(out_dir, "cross_session_day3_summary.csv")
    decay_csv = os.path.join(out_dir, "cross_session_decay_summary.csv")

    summary_day2.to_csv(day2_csv, index=False)
    summary_day3.to_csv(day3_csv, index=False)
    decay_df.to_csv(decay_csv, index=False)

    # Plot trajectories
    plot_day2_png = os.path.join(out_dir, "cross_session_day2_trajectories.png")
    plot_day3_png = os.path.join(out_dir, "cross_session_day3_trajectories.png")

    plot_session_trajectories(df_day2, "Day 2 (Session 1 Out-of-Session Test)", plot_day2_png, bin_size=args.bin_size)
    plot_session_trajectories(df_day3, "Day 3 (Session 2 Out-of-Session Test)", plot_day3_png, bin_size=args.bin_size)

    # Write Master Markdown Report
    report_path = os.path.join(out_dir, "cross_session_master_benchmark_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Yang2025 Multi-Day Cross-Session Master Benchmark Report\n\n")
        f.write(f"- **Dataset**: `{args.dataset}`\n")
        f.write(f"- **Training Session**: `Session 0 (Day 1)` (200 trials per subject)\n")
        f.write(f"- **Test Sessions**: `Session 1 (Day 2)` & `Session 2 (Day 3)` (200 trials each)\n")
        f.write(f"- **Analyzed Target Subjects**: `{len(csv_files)} subjects`\n\n")

        f.write("## 1. Day 2 Performance (Session 1 Out-of-Session Test)\n")
        f.write("| Category | Mean Accuracy | Final Accuracy |\n| --- | --- | --- |\n")
        for _, row in summary_day2.iterrows():
            f.write(f"| {row['Category']} | {row['Mean_Accuracy']:.4f} | {row['Final_Accuracy']:.4f} |\n")

        f.write("\n\n## 2. Day 3 Performance (Session 2 Out-of-Session Test)\n")
        f.write("| Category | Mean Accuracy | Final Accuracy |\n| --- | --- | --- |\n")
        for _, row in summary_day3.iterrows():
            f.write(f"| {row['Category']} | {row['Mean_Accuracy']:.4f} | {row['Final_Accuracy']:.4f} |\n")

        f.write("\n\n## 3. Multi-Day Time-Decay Analysis (Day 2 -> Day 3 Delta)\n")
        f.write("| Category | Day 2 Mean Acc | Day 3 Mean Acc | Time-Decay Delta (Day3 - Day2) |\n| --- | --- | --- | --- |\n")
        for _, row in decay_df.iterrows():
            delta_str = f"{row['Time_Decay_Delta']:+.4f}"
            f.write(f"| {row['Category']} | {row['Mean_Accuracy_Day2']:.4f} | {row['Mean_Accuracy_Day3']:.4f} | {delta_str} |\n")

        f.write("\n\n## 4. Visual Artifacts & Reports\n")
        f.write(f"- **Day 2 Trajectories Plot**: [`cross_session_day2_trajectories.png`](file:///{os.path.abspath(plot_day2_png)})\n")
        f.write(f"- **Day 3 Trajectories Plot**: [`cross_session_day3_trajectories.png`](file:///{os.path.abspath(plot_day3_png)})\n")
        f.write(f"- **Day 2 Summary CSV**: [`cross_session_day2_summary.csv`](file:///{os.path.abspath(day2_csv)})\n")
        f.write(f"- **Day 3 Summary CSV**: [`cross_session_day3_summary.csv`](file:///{os.path.abspath(day3_csv)})\n")
        f.write(f"- **Decay Delta CSV**: [`cross_session_decay_summary.csv`](file:///{os.path.abspath(decay_csv)})\n")

    print("================================================================================")
    print(f" Saved Day 2 Summary CSV     : {day2_csv}")
    print(f" Saved Day 3 Summary CSV     : {day3_csv}")
    print(f" Saved Decay Delta CSV       : {decay_csv}")
    print(f" Saved Day 2 Trajectory Plot : {plot_day2_png}")
    print(f" Saved Day 3 Trajectory Plot : {plot_day3_png}")
    print(f" Saved Master Markdown Report: {report_path}")
    print("================================================================================\n")


if __name__ == "__main__":
    main()
