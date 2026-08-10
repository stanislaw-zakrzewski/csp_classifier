"""
Grand Mean Population Upper Bound Benchmark Plot Generator for Exp 6
======================================================================

Aggregates population transfer simulation CSVs across all 7 MOABB datasets (35 held-out test subjects),
computes 10 Quantile Temporal Bins (Bin 0 to Bin 9), and plots the Grand Mean comparison showing Negative Transfer.

Saved to:
1. `graph_results/population_transfer/grand_mean_population_upper_bound.png`
2. `graph_results/population_transfer_benchmark/grand_mean_population_upper_bound.png`
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def parse_pop_classifier(clf_name: str) -> str:
    if "population_ATCNet" in clf_name or "ATCNet_population" in clf_name:
        return "Population ATCNet (N-5 Raw Donors - Negative Transfer Plateau)"
    elif "population_EEGNet" in clf_name or "EEGNet_population" in clf_name:
        return "Population EEGNet (N-5 Raw Donors)"
    elif "population_CSP_LDA" in clf_name:
        return "Population CSP-LDA (N-5 Donors)"
    elif "population_Cov" in clf_name:
        return "Population Cov-LR (N-5 Donors)"
    elif "scratch" in clf_name:
        return "Baseline Cold-Start Scratch (No Pre-training)"
    return clf_name


def main():
    datasets = ["Dreyer2023", "Dreyer2023A", "PhysionetMI", "Lee2019_MI", "GuttmannFlury2025_MI", "GuttmannFlury2025_ME", "Yang2025"]

    print("================================================================================")
    print(" Grand Mean Population Transfer Engine (Exp 6 - Negative Transfer Analysis)")
    print("================================================================================\n")

    # 1. Load Population Simulation CSVs across all datasets
    pop_dfs = []
    for ds in datasets:
        pop_dir = os.path.join("simulation_results", "population_transfer", ds)
        if os.path.exists(pop_dir):
            for f in glob.glob(os.path.join(pop_dir, "*.csv")):
                sub_id = f"{ds}_{os.path.basename(f).replace('.csv', '')}"
                df = pd.read_csv(f)
                df['Dataset'] = ds
                df['Subject'] = sub_id
                df['Category'] = df['Classifier'].apply(parse_pop_classifier)
                pop_dfs.append(df)

    if not pop_dfs:
        print("Error: No population simulation CSVs found.")
        return

    combined_pop_df = pd.concat(pop_dfs, ignore_index=True)

    # Compute Quantile Temporal Bins (Bin 0 to Bin 9) for each held-out subject
    pop_binned_list = []
    for (ds, sub_id, cat), group in combined_pop_df.groupby(['Dataset', 'Subject', 'Category']):
        group = group.copy()
        max_tr = group['Trial'].max()
        group['Bin'] = np.clip((group['Trial'] / (max_tr + 1) * 10).astype(int), 0, 9)
        binned_sub = group.groupby('Bin')['Is_Correct'].mean().reset_index()
        binned_sub['Dataset'] = ds
        binned_sub['Subject'] = sub_id
        binned_sub['Category'] = cat
        pop_binned_list.append(binned_sub)

    pop_binned_df = pd.concat(pop_binned_list, ignore_index=True)
    pop_summary = pop_binned_df.groupby(['Category', 'Bin'])['Is_Correct'].mean().reset_index()

    # 2. Also load GNN Cluster Strategy A & Strategy B trajectories from Exp 3 Grand Mean data for comparison
    atc_dfs = []
    for ds in datasets:
        atc_dir = os.path.join("simulation_results", "atcnet", ds)
        if os.path.exists(atc_dir):
            for f in glob.glob(os.path.join(atc_dir, "*.csv")):
                sub_id = f"{ds}_{os.path.basename(f).replace('.csv', '')}"
                df = pd.read_csv(f)
                df = df[df['Classifier'].str.contains("cluster_0|submodular_top5")].copy()
                if not df.empty:
                    df['Dataset'] = ds
                    df['Subject'] = sub_id
                    df['Category'] = df['Classifier'].apply(
                        lambda c: "GNN Cluster Strategy A (ATCNet Pooled ~50 Donors)" if "cluster" in c else "Submodular Strategy B (ATCNet Top-5 Donors)"
                    )
                    atc_dfs.append(df)

    if atc_dfs:
        combined_atc_df = pd.concat(atc_dfs, ignore_index=True)
        atc_binned_list = []
        for (ds, sub_id, cat), group in combined_atc_df.groupby(['Dataset', 'Subject', 'Category']):
            group = group.copy()
            max_tr = group['Trial'].max()
            group['Bin'] = np.clip((group['Trial'] / (max_tr + 1) * 10).astype(int), 0, 9)
            binned_sub = group.groupby('Bin')['Is_Correct'].mean().reset_index()
            binned_sub['Dataset'] = ds
            binned_sub['Subject'] = sub_id
            binned_sub['Category'] = cat
            atc_binned_list.append(binned_sub)

        atc_binned_df = pd.concat(atc_binned_list, ignore_index=True)
        atc_summary = atc_binned_df.groupby(['Category', 'Bin'])['Is_Correct'].mean().reset_index()
        final_plot_df = pd.concat([pop_summary, atc_summary], ignore_index=True)
    else:
        final_plot_df = pop_summary

    # Filter out minor baselines if any to keep plot clean
    target_categories = [
        "GNN Cluster Strategy A (ATCNet Pooled ~50 Donors)",
        "Submodular Strategy B (ATCNet Top-5 Donors)",
        "Population ATCNet (N-5 Raw Donors - Negative Transfer Plateau)",
        "Population EEGNet (N-5 Raw Donors)",
        "Baseline Cold-Start Scratch (No Pre-training)"
    ]
    final_plot_df = final_plot_df[final_plot_df['Category'].isin(target_categories)].copy()

    # Plot
    plt.figure(figsize=(12, 6.5))
    sns.set_theme(style="whitegrid")

    colors = {
        "GNN Cluster Strategy A (ATCNet Pooled ~50 Donors)": "#1f77b4",
        "Submodular Strategy B (ATCNet Top-5 Donors)": "#2ca02c",
        "Population ATCNet (N-5 Raw Donors - Negative Transfer Plateau)": "#ff7f0e",
        "Population EEGNet (N-5 Raw Donors)": "#d62728",
        "Baseline Cold-Start Scratch (No Pre-training)": "#7f7f7f"
    }

    sns.lineplot(
        data=final_plot_df,
        x='Bin',
        y='Is_Correct',
        hue='Category',
        style='Category',
        markers=True,
        dashes=False,
        palette=colors,
        linewidth=2.8
    )

    plt.title("Exp 6: Grand Mean Population Upper Bound vs GNN Clustering Across 7 Datasets", fontsize=13, fontweight='bold')
    plt.xlabel("Quantile Trial Bins (Bin 0 to Bin 9)", fontsize=11)
    plt.ylabel("Grand Mean Classification Accuracy", fontsize=11)
    plt.xticks(range(10), [f"Bin {i}" for i in range(10)])
    plt.ylim(0.48, 0.82)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
    plt.tight_layout()

    # Save to both paths
    out_dir1 = os.path.join("graph_results", "population_transfer")
    os.makedirs(out_dir1, exist_ok=True)
    plot_path1 = os.path.join(out_dir1, "grand_mean_population_upper_bound.png")
    plt.savefig(plot_path1, dpi=300)

    out_dir2 = os.path.join("graph_results", "population_transfer_benchmark")
    os.makedirs(out_dir2, exist_ok=True)
    plot_path2 = os.path.join(out_dir2, "grand_mean_population_upper_bound.png")
    plt.savefig(plot_path2, dpi=300)

    plt.close()

    print(f"[EXP 6 GRAND MEAN] Saved Grand Mean Population Transfer Plot at:\n  -> {plot_path1}\n  -> {plot_path2}")


if __name__ == "__main__":
    main()
