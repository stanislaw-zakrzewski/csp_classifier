"""
Grand Mean Benchmark Trajectory Plot Generator for Exp 2 & Exp 3
===================================================================

Generates Grand Mean Online Adaptive Trajectory comparison plots across ALL 7 MOABB datasets (423 subjects total).

Saved Outputs:
1. `graph_results/eegnet_benchmark/grand_mean_eegnet_trajectories.png` (Exp 2: EEGNet Grand Mean Trajectories)
2. `graph_results/atcnet_benchmark/grand_mean_atcnet_vs_eegnet_trajectories.png` (Exp 3: ATCNet vs EEGNet Grand Mean Trajectories)
"""

import os
import glob
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
    datasets = ["Dreyer2023", "Dreyer2023A", "PhysionetMI", "Lee2019_MI", "GuttmannFlury2025_MI", "GuttmannFlury2025_ME", "Yang2025"]

    print("================================================================================")
    print(f" Grand Mean Benchmark Engine (423 Subjects across 7 Datasets)")
    print("================================================================================\n")

    all_dfs = []
    for ds in datasets:
        # Load ATCNet simulation CSVs
        atc_dir = os.path.join("simulation_results", "atcnet", ds)
        if os.path.exists(atc_dir):
            for f in glob.glob(os.path.join(atc_dir, "*.csv")):
                sub_id = f"{ds}_{os.path.basename(f).replace('.csv', '')}"
                df = pd.read_csv(f)
                df['Dataset'] = ds
                df['Subject'] = sub_id
                df['Category'] = df['Classifier'].apply(parse_model_category)
                all_dfs.append(df)

        # Load EEGNet simulation CSVs
        eeg_dir = os.path.join("simulation_results", "eegnet", ds)
        if os.path.exists(eeg_dir):
            for f in glob.glob(os.path.join(eeg_dir, "*.csv")):
                sub_id = f"{ds}_{os.path.basename(f).replace('.csv', '')}"
                df = pd.read_csv(f)
                df = df[df['Classifier'].str.contains("EEGNet")].copy()
                if not df.empty:
                    df['Dataset'] = ds
                    df['Subject'] = sub_id
                    df['Category'] = df['Classifier'].apply(parse_model_category)
                    all_dfs.append(df)

    if not all_dfs:
        print("Error: No simulation CSVs found across datasets.")
        return

    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Filter for valid categories
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

    # Compute quantile temporal bins (Bin 0 to Bin 9) for each dataset & subject
    binned_list = []
    for (ds, sub_id, cat), group in combined_df.groupby(['Dataset', 'Subject', 'Category']):
        group = group.copy()
        max_tr = group['Trial'].max()
        group['Bin'] = np.clip((group['Trial'] / (max_tr + 1) * 10).astype(int), 0, 9)
        binned_sub = group.groupby('Bin')['Is_Correct'].mean().reset_index()
        binned_sub['Dataset'] = ds
        binned_sub['Subject'] = sub_id
        binned_sub['Category'] = cat
        binned_list.append(binned_sub)

    grand_binned_df = pd.concat(binned_list, ignore_index=True)
    grand_summary = grand_binned_df.groupby(['Category', 'Bin'])['Is_Correct'].mean().reset_index()

    sns.set_theme(style="whitegrid")

    # --------------------------------------------------------------------------
    # 1. EXP 2 GRAND MEAN PLOT: EEGNet Strategies (Grand Mean)
    # --------------------------------------------------------------------------
    eeg_summary = grand_summary[grand_summary['Category'].str.contains("EEGNet")].copy()
    eeg_out_dir = os.path.join("graph_results", "eegnet_benchmark")
    os.makedirs(eeg_out_dir, exist_ok=True)
    eeg_plot_path = os.path.join(eeg_out_dir, "grand_mean_eegnet_trajectories.png")

    plt.figure(figsize=(12, 6.5))
    eeg_palette = sns.color_palette("Set2", n_colors=len(eeg_summary['Category'].unique()))
    ax = sns.lineplot(
        data=eeg_summary,
        x='Bin',
        y='Is_Correct',
        hue='Category',
        style='Category',
        markers=True,
        dashes=False,
        palette=eeg_palette,
        linewidth=2.8
    )

    plt.title("Exp 2: Grand Mean EEGNet Online Adaptive Trajectories Across 7 Datasets (423 Subjects)", fontsize=13, fontweight='bold')
    plt.xlabel("Quantile Trial Bins (Bin 0 to Bin 9)", fontsize=11)
    plt.ylabel("Grand Mean Classification Accuracy", fontsize=11)
    plt.xticks(range(10), [f"Bin {i}" for i in range(10)])
    plt.ylim(0.48, 0.82)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
    plt.tight_layout()
    plt.savefig(eeg_plot_path, dpi=300)
    plt.close()
    print(f"[EXP 2 GRAND MEAN] Saved: {eeg_plot_path}")

    # --------------------------------------------------------------------------
    # 2. EXP 3 GRAND MEAN PLOT: ATCNet vs EEGNet Strategies (Grand Mean)
    # --------------------------------------------------------------------------
    atc_out_dir = os.path.join("graph_results", "atcnet_benchmark")
    os.makedirs(atc_out_dir, exist_ok=True)
    atc_plot_path = os.path.join(atc_out_dir, "grand_mean_atcnet_vs_eegnet_trajectories.png")

    plt.figure(figsize=(13, 6.5))
    atc_palette = sns.color_palette("tab10", n_colors=len(grand_summary['Category'].unique()))
    ax = sns.lineplot(
        data=grand_summary,
        x='Bin',
        y='Is_Correct',
        hue='Category',
        style='Category',
        markers=True,
        dashes=False,
        palette=atc_palette,
        linewidth=2.8
    )

    plt.title("Exp 3: Grand Mean ATCNet vs EEGNet Trajectories Across 7 Datasets (423 Subjects)", fontsize=13, fontweight='bold')
    plt.xlabel("Quantile Trial Bins (Bin 0 to Bin 9)", fontsize=11)
    plt.ylabel("Grand Mean Classification Accuracy", fontsize=11)
    plt.xticks(range(10), [f"Bin {i}" for i in range(10)])
    plt.ylim(0.48, 0.82)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
    plt.tight_layout()
    plt.savefig(atc_plot_path, dpi=300)
    plt.close()
    print(f"[EXP 3 GRAND MEAN] Saved: {atc_plot_path}")


if __name__ == "__main__":
    main()
