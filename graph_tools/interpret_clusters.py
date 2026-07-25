"""
Cluster Results Interpretation & Model Recommendation Engine
============================================================

This script analyzes and interprets the output of GNN subject compatibility clustering
(`graph_results/clustering/{dataset}/` or `graph_results/clustering_binned/{dataset}/`),
producing actionable insights, cluster-to-cluster transfer matrices, statistical significance reports,
and practical donor model selection recommendation rules.

Key Outputs:
------------
1. Cluster Transfer Matrix (K x K): Mean transfer accuracy from Donor Cluster k to Recipient Cluster m.
2. Optimal Donor Cluster Recommendation Rules: For each target recipient cluster, identifies the best donor cluster and expected accuracy gain over baseline.
3. Cluster Proficiency & Purity Profiles: Maps cluster members to baseline performance tiers (High >75%, Mid 60-75%, Low <60%).
4. Visual Heatmap & Bar Plot Artifacts:
   - `{dataset}_cluster_transfer_matrix.png`: K x K Donor-to-Recipient cluster transfer heatmap.
   - `{dataset}_cluster_proficiency_breakdown.png`: Cluster baseline proficiency distribution.
5. Master Interpretation Report: `cluster_interpretation_report.md` & `cluster_recommendations.csv`.

Usage Examples:
---------------
1. Interpret standard clustering results for Dreyer2023:
    python graph_tools/interpret_clusters.py --dataset Dreyer2023

2. Interpret 10-bin temporal binned clustering results:
    python graph_tools/interpret_clusters.py --dataset Dreyer2023 --binned
"""

import os
import glob
import argparse
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind


def find_clustering_dir(input_dir: str, dataset: str, is_binned: bool = False) -> str:
    """Locate clustering output directory for standard or binned mode."""
    subfolder = "clustering_binned" if is_binned else "clustering"
    candidates = [
        os.path.join(input_dir, subfolder, dataset),
        os.path.join(input_dir, dataset),
        os.path.join("graph_results", subfolder, dataset),
    ]
    for cand in candidates:
        if os.path.exists(cand) and os.path.exists(os.path.join(cand, "subject_cluster_assignments.csv")):
            return cand
    raise FileNotFoundError(
        f"Clustering output directory containing 'subject_cluster_assignments.csv' was not found for dataset '{dataset}' (is_binned={is_binned}).\n"
        f"Please run 'python graph_tools/cluster_subjects.py --dataset {dataset} {'--binned' if is_binned else ''}' first."
    )


def find_graph_files(dataset: str, is_binned: bool = False) -> list[str]:
    """Find matching GraphML graph files for dataset."""
    subfolder = "graphs_binned" if is_binned else "graphs"
    candidates = [
        os.path.join("graph_results", subfolder, dataset),
        os.path.join("graph_results", dataset),
        os.path.join("simulation_results", subfolder, dataset),
    ]
    for cand in candidates:
        if os.path.exists(cand):
            files = glob.glob(os.path.join(cand, "*.graphml"))
            files = [f for f in files if not f.endswith("_above_baseline.graphml")]
            if is_binned:
                files = [f for f in files if "binned" in f.lower()]
            else:
                files = [f for f in files if "binned" not in f.lower() and "_bin_" not in f]
            if files:
                return files
    return []


def analyze_pipeline_clusters(
    pipe_df: pd.DataFrame,
    G: nx.DiGraph,
    pipe_name: str
) -> dict:
    """Perform detailed cluster interpretation for a single pipeline configuration."""
    subjects = pipe_df['Subject'].astype(str).tolist()
    N = len(subjects)

    W_df = nx.to_pandas_adjacency(G, nodelist=subjects, weight='weight')
    W = W_df.values

    baselines = pipe_df['Baseline_Accuracy'].values
    donor_clusters = pipe_df['Donor_Cluster'].values
    recip_clusters = pipe_df['Recipient_Cluster'].values

    num_donor_k = len(np.unique(donor_clusters))
    num_recip_k = len(np.unique(recip_clusters))

    cluster_transfer_matrix = np.zeros((num_donor_k, num_recip_k))
    cluster_transfer_counts = np.zeros((num_donor_k, num_recip_k))

    sub_to_idx = {s: i for i, s in enumerate(subjects)}

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            don_k = donor_clusters[i]
            rec_m = recip_clusters[j]
            cluster_transfer_matrix[don_k, rec_m] += W[i, j]
            cluster_transfer_counts[don_k, rec_m] += 1

    with np.errstate(divide='ignore', invalid='ignore'):
        cluster_transfer_avg = np.where(cluster_transfer_counts > 0, cluster_transfer_matrix / cluster_transfer_counts, 0.0)

    recommendations = []
    for rec_m in range(num_recip_k):
        rec_subjects = pipe_df[pipe_df['Recipient_Cluster'] == rec_m]['Subject'].tolist()
        rec_baselines = pipe_df[pipe_df['Recipient_Cluster'] == rec_m]['Baseline_Accuracy'].values
        rec_mean_baseline = float(np.mean(rec_baselines)) if len(rec_baselines) > 0 else 0.0

        donor_scores = cluster_transfer_avg[:, rec_m]
        best_donor_k = int(np.argmax(donor_scores))
        best_donor_acc = float(donor_scores[best_donor_k])

        donor_subjects = pipe_df[pipe_df['Donor_Cluster'] == best_donor_k]['Subject'].tolist()
        gain_over_baseline = best_donor_acc - rec_mean_baseline

        recommendations.append({
            'Pipeline_Config': pipe_name,
            'Recipient_Cluster': rec_m,
            'Recipient_Count': len(rec_subjects),
            'Recipient_Subjects': ", ".join(map(str, rec_subjects[:8])) + ("..." if len(rec_subjects) > 8 else ""),
            'Recipient_Mean_Baseline': round(rec_mean_baseline, 4),
            'Recommended_Donor_Cluster': best_donor_k,
            'Recommended_Donor_Count': len(donor_subjects),
            'Top_Donor_Subjects': ", ".join(map(str, donor_subjects[:8])) + ("..." if len(donor_subjects) > 8 else ""),
            'Expected_Transfer_Accuracy': round(best_donor_acc, 4),
            'Expected_Gain_Over_Baseline': round(gain_over_baseline, 4)
        })

    donor_profiles = []
    for don_k in range(num_donor_k):
        members = pipe_df[pipe_df['Donor_Cluster'] == don_k]
        m_base = members['Baseline_Accuracy'].values

        high_c = np.sum(m_base >= 0.75)
        mid_c = np.sum((m_base >= 0.60) & (m_base < 0.75))
        low_c = np.sum(m_base < 0.60)

        star_donors = []
        for _, row in members.iterrows():
            idx = sub_to_idx[str(row['Subject'])]
            out_mean = float(np.mean([W[idx, j] for j in range(N) if j != idx]))
            star_donors.append((row['Subject'], out_mean))
        star_donors.sort(key=lambda x: x[1], reverse=True)
        top_stars = [f"S{s}({acc:.2f})" for s, acc in star_donors[:3]]

        donor_profiles.append({
            'Pipeline_Config': pipe_name,
            'Donor_Cluster': don_k,
            'Member_Count': len(members),
            'Mean_Baseline': round(float(np.mean(m_base)), 4) if len(m_base) > 0 else 0.0,
            'High_Proficiency_75+': int(high_c),
            'Mid_Proficiency_60_75': int(mid_c),
            'Low_Proficiency_lt60': int(low_c),
            'Star_Donors': ", ".join(top_stars)
        })

    return {
        'recommendations': pd.DataFrame(recommendations),
        'donor_profiles': pd.DataFrame(donor_profiles),
        'cluster_transfer_avg': cluster_transfer_avg
    }


def generate_interpretation_plots(
    clustering_dir: str,
    pipe_name: str,
    cluster_transfer_avg: np.ndarray,
    donor_profiles_df: pd.DataFrame
):
    """Generate visual interpretation heatmaps and proficiency breakdown bar plots."""
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cluster_transfer_avg,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        ax=ax,
        cbar_kws={'label': 'Mean Transfer Accuracy'}
    )
    ax.set_title(f"Cluster Transfer Matrix: {pipe_name}\n(Donor Cluster -> Recipient Cluster)", fontsize=12, fontweight='bold')
    ax.set_xlabel("Recipient Cluster ID", fontsize=11)
    ax.set_ylabel("Donor Cluster ID", fontsize=11)
    plt.tight_layout()

    matrix_plot_path = os.path.join(clustering_dir, f"{pipe_name}_cluster_transfer_matrix.png")
    plt.savefig(matrix_plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    prof_data = donor_profiles_df[donor_profiles_df['Pipeline_Config'] == pipe_name].copy()
    if not prof_data.empty:
        plot_df = prof_data.melt(
            id_vars=['Donor_Cluster'],
            value_vars=['High_Proficiency_75+', 'Mid_Proficiency_60_75', 'Low_Proficiency_lt60'],
            var_name='Proficiency_Tier',
            value_name='Subject_Count'
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(
            data=plot_df,
            x='Donor_Cluster',
            y='Subject_Count',
            hue='Proficiency_Tier',
            palette=['#2ecc71', '#f39c12', '#e74c3c'],
            ax=ax
        )
        ax.set_title(f"Subject Proficiency Tier Breakdown by Donor Cluster ({pipe_name})", fontsize=12, fontweight='bold')
        ax.set_xlabel("Donor Cluster ID", fontsize=11)
        ax.set_ylabel("Number of Subjects", fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.4, axis='y')
        plt.tight_layout()

        bar_plot_path = os.path.join(clustering_dir, f"{pipe_name}_proficiency_breakdown.png")
        plt.savefig(bar_plot_path, dpi=150, bbox_inches='tight')
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Interpret GNN subject compatibility clustering results and generate donor recommendation rules."
    )
    parser.add_argument(
        "--input", "-i",
        default="graph_results",
        help="Input directory containing clustering folder. Default: graph_results"
    )
    parser.add_argument(
        "--dataset", "-d",
        default="Dreyer2023",
        help="Dataset name ('Dreyer2023', 'PhysionetMI', etc.). Default: Dreyer2023"
    )
    parser.add_argument(
        "--binned",
        action="store_true",
        help="Interpret 10-bin temporal binned clustering outputs from clustering_binned."
    )

    args = parser.parse_args()

    clustering_dir = find_clustering_dir(args.input, args.dataset, is_binned=args.binned)
    assignments_path = os.path.join(clustering_dir, "subject_cluster_assignments.csv")
    summary_path = os.path.join(clustering_dir, "clustering_summary.csv")

    assignments_df = pd.read_csv(assignments_path)
    summary_df = pd.read_csv(summary_path) if os.path.exists(summary_path) else None

    graph_files = find_graph_files(args.dataset, is_binned=args.binned)

    print("================================================================================")
    print(f" GNN Subject Compatibility Cluster Interpretation Engine (Binned={args.binned})")
    print("================================================================================")
    print(f" Dataset             : {args.dataset}")
    print(f" Clustering Directory: {clustering_dir}")
    print(f" Matched Graph Files : {len(graph_files)}")
    print("================================================================================\n")

    all_recommendations = []
    all_donor_profiles = []

    for g_file in graph_files:
        pipe_name = os.path.splitext(os.path.basename(g_file))[0].replace("_binned", "")
        pipe_df = assignments_df[assignments_df['Pipeline_Config'] == pipe_name].copy()

        if pipe_df.empty:
            continue

        G = nx.read_graphml(g_file)
        results = analyze_pipeline_clusters(pipe_df, G, pipe_name)

        all_recommendations.append(results['recommendations'])
        all_donor_profiles.append(results['donor_profiles'])

        generate_interpretation_plots(
            clustering_dir=clustering_dir,
            pipe_name=pipe_name,
            cluster_transfer_avg=results['cluster_transfer_avg'],
            donor_profiles_df=results['donor_profiles']
        )

        print(f"--- Pipeline Configuration: '{pipe_name}' ---")
        for _, rec in results['recommendations'].iterrows():
            print(f"  Recipient Cluster {rec['Recipient_Cluster']} ({rec['Recipient_Count']} subjects, avg baseline {rec['Recipient_Mean_Baseline']:.2f}):")
            print(f"    -> Recommend Donor Cluster {rec['Recommended_Donor_Cluster']} (expected transfer {rec['Expected_Transfer_Accuracy']:.2f}, gain: +{rec['Expected_Gain_Over_Baseline']:.4f})")
        print()

    combined_recommendations = pd.concat(all_recommendations, ignore_index=True) if all_recommendations else pd.DataFrame()
    combined_profiles = pd.concat(all_donor_profiles, ignore_index=True) if all_donor_profiles else pd.DataFrame()

    rec_csv_path = os.path.join(clustering_dir, "cluster_recommendations.csv")
    combined_recommendations.to_csv(rec_csv_path, index=False)

    prof_csv_path = os.path.join(clustering_dir, "cluster_proficiency_profiles.csv")
    combined_profiles.to_csv(prof_csv_path, index=False)

    def df_to_markdown(df_to_format):
        try:
            return df_to_format.to_markdown(index=False)
        except Exception:
            cols = list(df_to_format.columns)
            header = "| " + " | ".join(cols) + " |"
            divider = "| " + " | ".join(["---"] * len(cols)) + " |"
            rows = []
            for _, row in df_to_format.iterrows():
                rows.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
            return "\n".join([header, divider] + rows)

    report_name = "binned_cluster_interpretation_report.md" if args.binned else "cluster_interpretation_report.md"
    md_lines = [
        f"# Subject Compatibility Cluster Interpretation & Model Recommendation Report (Binned={args.binned})",
        f"",
        f"- **Dataset**: `{args.dataset}`",
        f"- **Temporal 10-Bin Mode**: `{args.binned}`",
        f"- **Analyzed Pipelines**: `{len(graph_files)} configurations`",
        f"",
        f"## 1. Optimal Donor Model Selection Recommendation Rules",
        f"The table below defines actionable donor cluster recommendation rules for target subjects based on their assigned Recipient Cluster:",
        f"",
        df_to_markdown(combined_recommendations[['Pipeline_Config', 'Recipient_Cluster', 'Recipient_Count', 'Recipient_Mean_Baseline', 'Recommended_Donor_Cluster', 'Expected_Transfer_Accuracy', 'Expected_Gain_Over_Baseline', 'Top_Donor_Subjects']]),
        f"",
        f"## 2. Donor Cluster Member & Proficiency Breakdown",
        f"",
        df_to_markdown(combined_profiles[['Pipeline_Config', 'Donor_Cluster', 'Member_Count', 'Mean_Baseline', 'High_Proficiency_75+', 'Mid_Proficiency_60_75', 'Low_Proficiency_lt60', 'Star_Donors']]),
        f"",
        f"## 3. Generated Visual Artifacts",
        f"The following visual plots have been created in `{clustering_dir}`:",
        f"- **Cluster Transfer Matrix Heatmaps (`*_cluster_transfer_matrix.png`)**: Visually displays transfer performance between all Donor Cluster -> Recipient Cluster pairs.",
        f"- **Proficiency Tier Distributions (`*_proficiency_breakdown.png`)**: Stacked bar charts displaying high, medium, and low proficiency subject distributions across clusters."
    ]

    report_path = os.path.join(clustering_dir, report_name)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(md_lines))

    print("================================================================================")
    print(f" Saved Cluster Recommendation Rules CSV : {rec_csv_path}")
    print(f" Saved Cluster Proficiency CSV         : {prof_csv_path}")
    print(f" Saved Interpretation Markdown Report   : {report_path}")
    print("================================================================================\n")


if __name__ == "__main__":
    main()
