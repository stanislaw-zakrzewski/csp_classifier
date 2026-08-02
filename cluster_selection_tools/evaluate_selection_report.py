"""
Cluster Selection Master Recommendation Engine
================================================

Aggregates selection validation CSVs across datasets, evaluates the decision rule,
and generates the final recommendation report and strategy comparison chart.
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Evaluate Cluster Selection Strategies and Issue Master Recommendation.")
    args = parser.parse_args()

    csv_files = glob.glob(os.path.join("graph_results", "cluster_selection", "*", "selection_validation_results.csv"))

    if not csv_files:
        raise FileNotFoundError("No cluster selection validation CSV files found. Run validate_cluster_selection.py first.")

    out_dir = os.path.join("graph_results", "cluster_selection_recommendation")
    os.makedirs(out_dir, exist_ok=True)

    print("================================================================================")
    print(" Cluster Selection Master Recommendation Engine")
    print("================================================================================")
    print(f" Found CSVs          : {len(csv_files)} datasets")
    print(f" Output Directory    : {out_dir}")
    print("================================================================================\n")

    all_dfs = []
    for c_path in csv_files:
        ds_name = os.path.basename(os.path.dirname(c_path))
        df = pd.read_csv(c_path)
        df['Dataset'] = ds_name
        all_dfs.append(df)

    full_df = pd.concat(all_dfs, ignore_index=True)

    # Compute summary metrics per dataset
    dataset_summaries = []
    for ds_name, grp in full_df.groupby('Dataset'):
        dataset_summaries.append({
            'Dataset': ds_name,
            'Subjects': len(grp),
            'Oracle_Acc': grp['Oracle_Acc'].mean(),
            'Option1_Acc': grp['Option1_Acc'].mean(),
            'Option1_MatchRate': grp['Option1_Match'].mean(),
            'Option2_Acc': grp['Option2_Acc'].mean(),
            'Option3_Acc': grp['Option3_Acc'].mean(),
            'Option3_MatchRate': grp['Option3_Match'].mean()
        })

    sum_df = pd.DataFrame(dataset_summaries).sort_values(by='Oracle_Acc', ascending=False)

    # Grand Means
    grand_oracle = full_df['Oracle_Acc'].mean()
    grand_opt1 = full_df['Option1_Acc'].mean()
    grand_opt1_match = full_df['Option1_Match'].mean()
    grand_opt2 = full_df['Option2_Acc'].mean()
    grand_opt3 = full_df['Option3_Acc'].mean()
    grand_opt3_match = full_df['Option3_Match'].mean()

    # Determine Recommendation Rule
    if grand_opt1 >= grand_opt3 or grand_opt2 >= grand_opt3:
        if grand_opt2 >= grand_opt1:
            recommended_option = "Option 2 (Soft Cluster Mixture Ensemble)"
            rec_code = "OPTION_2"
        else:
            recommended_option = "Option 1 (Zero-Shot GNN Distance Selection)"
            rec_code = "OPTION_1"
    else:
        recommended_option = "Option 3 (First 8-Trial Confidence Selection)"
        rec_code = "OPTION_3"

    # Save summary CSV
    summary_csv = os.path.join(out_dir, "selection_strategy_summary.csv")
    sum_df.to_csv(summary_csv, index=False)

    # Generate Comparison Plot
    plt.figure(figsize=(10, 6))
    strategies = ['Oracle Upper Bound', 'Option 1 (GNN Distance)', 'Option 2 (Soft Ensemble)', 'Option 3 (8-Trial Confidence)']
    accuracies = [grand_oracle, grand_opt1, grand_opt2, grand_opt3]
    colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728']

    bars = plt.bar(strategies, accuracies, color=colors, width=0.55, edgecolor='black', linewidth=1.2)
    plt.title("Cluster Selection Strategy Grand Mean Comparison (Across Datasets)", fontsize=14, fontweight='bold')
    plt.ylabel("Classification Accuracy", fontsize=12)
    plt.ylim(0.40, 1.0)
    plt.grid(True, linestyle='--', alpha=0.5, axis='y')

    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2.0, bar.get_height() + 0.01, f"{acc:.4f}", ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plot_png = os.path.join(out_dir, "cluster_selection_strategy_comparison.png")
    plt.savefig(plot_png, dpi=300)
    plt.close()

    # Write Master Recommendation Report
    report_path = os.path.join(out_dir, "cluster_selection_master_recommendation.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Master Recommendation Report: GNN Cluster Model Selection Strategy\n\n")
        f.write(f"## 🏆 Formal Recommendation: `{recommended_option}`\n\n")

        f.write("### 1. Empirical Performance Matrix (Grand Mean Across Datasets)\n\n")
        f.write("| Selection Strategy | Grand Mean Accuracy | Oracle Match Rate | Zero-Shot Calibration Requirement |\n| --- | --- | --- | --- |\n")
        f.write(f"| **Oracle Upper Bound Control** | **{grand_oracle:.4f}** | 100.0% | Full Test Session |\n")
        f.write(f"| **Option 1 (Zero-Shot GNN Distance)** | **{grand_opt1:.4f}** | {grand_opt1_match*100:.1f}% | 0 Trials (Resting Baseline) |\n")
        f.write(f"| **Option 2 (Soft Cluster Mixture)** | **{grand_opt2:.4f}** | N/A (Soft Ensemble) | 0 Trials (Resting Baseline) |\n")
        f.write(f"| **Option 3 (8-Trial Confidence)** | **{grand_opt3:.4f}** | {grand_opt3_match*100:.1f}% | 8 Online Calibration Trials |\n\n")

        f.write("### 2. Per-Dataset Selection Summary Table\n\n")
        f.write("| Dataset | Subjects | Oracle Acc | Option 1 Acc | Option 2 Acc | Option 3 Acc |\n| --- | --- | --- | --- | --- | --- |\n")
        for _, row in sum_df.iterrows():
            f.write(f"| `{row['Dataset']}` | {row['Subjects']} | {row['Oracle_Acc']:.4f} | {row['Option1_Acc']:.4f} | {row['Option2_Acc']:.4f} | {row['Option3_Acc']:.4f} |\n")

        f.write("\n\n### 3. Decision Rationale & Engineering Guidance\n\n")
        if rec_code in ["OPTION_1", "OPTION_2"]:
            f.write(f"Empirical validation confirms that **{recommended_option}** achieves superior or equal performance to Option 3 **without requiring any calibration trials**.\n")
        else:
            f.write(f"Empirical validation demonstrates that GNN feature matching alone (Option 1/2) is insufficient to guarantee optimal cluster selection across all subject variations. **Option 3 (First 8-Trial Confidence Selection)** outperforms zero-shot feature matching and is **formally recommended** for deployment.\n")

        f.write(f"\n- **Strategy Comparison Plot**: [`cluster_selection_strategy_comparison.png`](file:///{os.path.abspath(plot_png)})\n")
        f.write(f"- **Summary CSV**: [`selection_strategy_summary.csv`](file:///{os.path.abspath(summary_csv)})\n")

    print("================================================================================")
    print(f" Formal Recommendation       : {recommended_option}")
    print(f" Saved Summary CSV          : {summary_csv}")
    print(f" Saved Strategy Plot        : {plot_png}")
    print(f" Saved Master Markdown Report: {report_path}")
    print("================================================================================\n")


if __name__ == "__main__":
    main()
