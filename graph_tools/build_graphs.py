"""
Build Standard Performance Graphs
=================================

This script processes raw trial-by-trial simulation CSV files for cross-subject EEG classification,
extracts final cumulative cross-subject accuracies for edges, and attaches target subject baseline model
accuracies directly to nodes.

Outputs:
--------
1. GraphML files for each pipeline configuration (3 pipelines x 2 modes: static and adaptive).
2. Accuracy Heatmap PNG plots for each graph.
3. Summary CSV tables per dataset.

Usage:
------
    python graph_tools/build_graphs.py --dataset Dreyer2023
    python graph_tools/build_graphs.py --dataset PhysionetMI
    python graph_tools/build_graphs.py --dataset GutmannFlury2025_MI
"""

import os
import glob
import re
import argparse
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


def parse_classifier_name(clf_name: str):
    """
    Extract source subject and pipeline type from classifier name.
    Example: 'subject_2_Cov_Tangent_Space_LR_pipeline' -> ('2', 'Cov_Tangent_Space_LR_pipeline_adaptive')
    Example: 'subject_2_CSP_LDA_pipeline_static' -> ('2', 'CSP_LDA_pipeline_static')
    """
    match = re.match(r'subject_(\d+)_(.+)', clf_name)
    if not match:
        return None, None
    src_sub, clf_type = match.groups()
    if not clf_type.endswith('_static') and not clf_type.endswith('_adaptive'):
        clf_type += '_adaptive'
    return src_sub, clf_type


def parse_baseline_classifier_name(clf_name: str):
    """
    Map baseline classifier name to matching graph pipeline keys (both static and adaptive).
    Example: 'baseline_CSP_LDA' -> ['CSP_LDA_pipeline_adaptive', 'CSP_LDA_pipeline_static']
    """
    if "Cov_Tangent" in clf_name:
        return ['Cov_Tangent_Space_LR_pipeline_adaptive', 'Cov_Tangent_Space_LR_pipeline_static']
    elif "CSP_LDA" in clf_name:
        return ['CSP_LDA_pipeline_adaptive', 'CSP_LDA_pipeline_static']
    elif "CSP_SVM" in clf_name:
        return ['CSP_SVM_pipeline_adaptive', 'CSP_SVM_pipeline_static']
    return []


def build_graphs(
    data_dir: str,
    output_dir: str,
    save_heatmaps: bool = True
):
    """
    Process simulation CSV files, build standard directed performance graphs, attach node baseline attributes,
    and save GraphML files and heatmaps.
    """
    os.makedirs(output_dir, exist_ok=True)

    csv_files = [
        f for f in os.listdir(data_dir)
        if f.endswith('.csv') and f != 'averaged_accuracies.csv' and not f.startswith('baseline') and not f.startswith('classifier_')
    ]

    if not csv_files:
        raise FileNotFoundError(f"No simulation CSV files found in '{data_dir}'.")

    subjects = sorted(list(set(f.split('.')[0] for f in csv_files)), key=lambda x: int(x) if x.isdigit() else x)

    # Initialize standard graphs dictionary
    default_pipeline_names = [
        'Cov_Tangent_Space_LR_pipeline_adaptive',
        'CSP_LDA_pipeline_adaptive',
        'CSP_SVM_pipeline_adaptive',
        'Cov_Tangent_Space_LR_pipeline_static',
        'CSP_LDA_pipeline_static',
        'CSP_SVM_pipeline_static'
    ]

    graphs = {name: nx.DiGraph() for name in default_pipeline_names}

    for g in graphs.values():
        g.add_nodes_from(subjects)

    print(f"Found {len(csv_files)} CSV files in '{data_dir}'. Parsing simulation data...")

    for file in tqdm(csv_files):
        target_subject = file.split('.')[0]
        filepath = os.path.join(data_dir, file)

        df = pd.read_csv(filepath)

        # Extract final cumulative accuracy per classifier
        final_accs = df.groupby('Classifier').tail(1)

        for _, row in final_accs.iterrows():
            clf_name = str(row['Classifier'])
            acc = float(row['Cumulative_Accuracy'])

            # Baseline classifier handling
            if clf_name.startswith('baseline'):
                target_types = parse_baseline_classifier_name(clf_name)
                for clf_type in target_types:
                    if clf_type in graphs and target_subject in graphs[clf_type]:
                        graphs[clf_type].nodes[target_subject]['baseline_accuracy'] = round(acc, 5)
                        graphs[clf_type].nodes[target_subject]['baseline_classifier'] = clf_name
                continue

            # Cross-subject classifier handling
            src_subject, clf_type = parse_classifier_name(clf_name)

            if src_subject:
                if clf_type not in graphs:
                    graphs[clf_type] = nx.DiGraph()
                    graphs[clf_type].add_nodes_from(subjects)

                graphs[clf_type].add_edge(src_subject, target_subject, weight=round(acc, 5))

    summary_rows = []

    # Save GraphML files & heatmaps
    print("\nSaving GraphML files and visualization heatmaps...")
    for name, g in graphs.items():
        if g.number_of_edges() == 0:
            continue

        print(f"Graph '{name}': {g.number_of_nodes()} nodes, {g.number_of_edges()} edges.")

        graphml_path = os.path.join(output_dir, f"{name}.graphml")
        nx.write_graphml(g, graphml_path)

        edge_weights = [d['weight'] for _, _, d in g.edges(data=True) if 'weight' in d]
        avg_acc = float(np.mean(edge_weights)) if edge_weights else 0.0
        min_acc = float(np.min(edge_weights)) if edge_weights else 0.0
        max_acc = float(np.max(edge_weights)) if edge_weights else 0.0

        summary_rows.append({
            'Pipeline_Graph': name,
            'Node_Count': g.number_of_nodes(),
            'Edge_Count': g.number_of_edges(),
            'Mean_Accuracy': round(avg_acc, 4),
            'Min_Accuracy': round(min_acc, 4),
            'Max_Accuracy': round(max_acc, 4)
        })

        if save_heatmaps and g.number_of_edges() > 0:
            adj_matrix = nx.to_pandas_adjacency(g, nodelist=subjects, weight='weight')

            # Populate diagonal (Source s == Target s) with target subject's baseline classifier accuracy
            for s in subjects:
                if s in g.nodes and 'baseline_accuracy' in g.nodes[s]:
                    adj_matrix.loc[s, s] = float(g.nodes[s]['baseline_accuracy'])

            fig, ax = plt.subplots(figsize=(11, 9))
            sns.heatmap(adj_matrix, cmap="viridis", vmin=0, vmax=1, ax=ax)
            ax.set_title(f"Accuracy Heatmap: {name}\n(Source: Row, Target: Col)", fontsize=12, fontweight='bold')
            ax.set_xlabel("Target Subject")
            ax.set_ylabel("Source Subject")

            tick_step = max(1, len(subjects) // 10)
            ax.set_xticks(ticks=range(0, len(subjects), tick_step))
            ax.set_xticklabels([subjects[i] for i in range(0, len(subjects), tick_step)])
            ax.set_yticks(ticks=range(0, len(subjects), tick_step))
            ax.set_yticklabels([subjects[i] for i in range(0, len(subjects), tick_step)])

            plt.tight_layout()
            heatmap_path = os.path.join(output_dir, f"{name}_heatmap.png")
            plt.savefig(heatmap_path, dpi=150)
            plt.close()

    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = os.path.join(output_dir, "graphs_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Saved graph summary CSV to: {summary_csv_path}")
    print(f"Graph construction complete! Outputs saved to: {output_dir}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Construct standard performance graphs from simulation CSVs."
    )
    parser.add_argument(
        "--dataset", "-d",
        default="Dreyer2023",
        help="Dataset name to process ('Dreyer2023', 'PhysionetMI', etc.). Default: Dreyer2023"
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Custom input directory containing simulation CSVs. Default: simulation_results/{dataset}"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help="Output directory for graphml and heatmaps. Default: graph_results/graphs/{dataset}"
    )
    parser.add_argument(
        "--no-heatmaps",
        action="store_true",
        help="Do not generate heatmap PNG plots."
    )

    args = parser.parse_args()

    data_dir = args.data_dir or os.path.join("simulation_results", args.dataset)
    output_dir = args.output_dir or os.path.join("graph_results", "graphs", args.dataset)

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Input simulation directory '{data_dir}' does not exist.")

    build_graphs(
        data_dir=data_dir,
        output_dir=output_dir,
        save_heatmaps=not args.no_heatmaps
    )


if __name__ == "__main__":
    main()
