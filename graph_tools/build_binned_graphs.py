"""
Build Temporal Trial-Binned Performance Graphs
================================================

This script processes raw trial-by-trial simulation CSV files for cross-subject EEG classification,
splits trial runs into N temporal bins (default: 10 equal trial windows), and constructs directed
graphs enriched with temporal accuracy evolution attributes on edges AND baseline classifier performance
metrics stored directly on nodes.

Outputs:
--------
1. Vector-Attributed GraphML files containing 10-bin interval & cumulative accuracies and adaptation slopes for edges,
   and matching within-subject baseline performance trajectory attributes on nodes.
2. Per-bin Snapshot GraphML files representing state of the graph at each trial checkpoint.
3. Visualization heatmaps showing Zero-Shot (Bin 1), Mid-term (Bin 5), Final (Bin 10), and Adaptation Slope.
4. Summary CSV tables per pipeline.

Usage:
------
    python graph_tools/build_binned_graphs.py --dataset Dreyer2023
    python graph_tools/build_binned_graphs.py --dataset PhysionetMI --num-bins 10
"""

import os
import glob
import re
import json
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


def compute_trial_bin_stats(df_clf_trials: pd.DataFrame, num_bins: int = 10) -> dict:
    """
    Given trial-by-trial DataFrame for a single classifier run, compute 10-bin interval & cumulative stats.
    """
    df_sorted = df_clf_trials.sort_values('Trial').reset_index(drop=True)
    total_trials = len(df_sorted)
    
    if total_trials == 0:
        return None

    bin_size = max(1, total_trials // num_bins)
    
    interval_accs = []
    cumulative_accs = []
    
    for b in range(num_bins):
        start_idx = b * bin_size
        end_idx = (b + 1) * bin_size if b < num_bins - 1 else total_trials
        
        bin_slice = df_sorted.iloc[start_idx:end_idx]
        
        if len(bin_slice) > 0:
            if 'Is_Correct' in bin_slice.columns:
                inter_acc = float(bin_slice['Is_Correct'].mean())
            else:
                inter_acc = float(bin_slice['Cumulative_Accuracy'].iloc[-1])
            cum_acc = float(df_sorted.iloc[:end_idx]['Cumulative_Accuracy'].iloc[-1])
        else:
            inter_acc = 0.0
            cum_acc = 0.0
            
        interval_accs.append(round(inter_acc, 5))
        cumulative_accs.append(round(cum_acc, 5))

    # Calculate linear adaptation slope across bins
    x = np.arange(num_bins)
    slope = float(np.polyfit(x, interval_accs, 1)[0])

    return {
        'interval_accs': interval_accs,
        'cumulative_accs': cumulative_accs,
        'final_cum_acc': cumulative_accs[-1],
        'initial_interval_acc': interval_accs[0],
        'final_interval_acc': interval_accs[-1],
        'adaptation_slope': round(slope, 6),
        'total_trials': total_trials
    }


def build_binned_graphs(
    data_dir: str,
    output_dir: str,
    num_bins: int = 10,
    save_snapshots: bool = True,
    save_heatmaps: bool = True
):
    """
    Main function to construct temporal trial-binned performance graphs with baseline node attributes.
    """
    os.makedirs(output_dir, exist_ok=True)
    snapshots_dir = os.path.join(output_dir, "snapshots")
    plots_dir = os.path.join(output_dir, "plots")
    
    if save_snapshots:
        os.makedirs(snapshots_dir, exist_ok=True)
    if save_heatmaps:
        os.makedirs(plots_dir, exist_ok=True)

    csv_files = [
        f for f in os.listdir(data_dir)
        if f.endswith('.csv') and not f.startswith('averaged_') and not f.startswith('baseline') and not f.startswith('classifier_')
    ]

    if not csv_files:
        raise FileNotFoundError(f"No valid simulation CSV files found in '{data_dir}'.")

    # Extract subject IDs
    subjects = sorted(list(set(f.split('.')[0] for f in csv_files)), key=lambda x: int(x) if x.isdigit() else x)

    pipeline_names = [
        'Cov_Tangent_Space_LR_pipeline_adaptive',
        'CSP_LDA_pipeline_adaptive',
        'CSP_SVM_pipeline_adaptive',
        'Cov_Tangent_Space_LR_pipeline_static',
        'CSP_LDA_pipeline_static',
        'CSP_SVM_pipeline_static'
    ]

    # Initialize Graphs & Snapshot Graphs
    full_graphs = {p: nx.DiGraph() for p in pipeline_names}
    snapshot_graphs = {
        p: [nx.DiGraph() for _ in range(num_bins)] for p in pipeline_names
    }

    # Add nodes to all graphs
    for g in full_graphs.values():
        g.add_nodes_from(subjects)
    for p in pipeline_names:
        for snap_g in snapshot_graphs[p]:
            snap_g.add_nodes_from(subjects)

    print(f"Found {len(csv_files)} subject simulation CSV files in '{data_dir}'.")
    print(f"Processing temporal trial-binned graph generation (Bins = {num_bins})...")

    summary_rows = []

    for file in tqdm(csv_files):
        target_subject = file.split('.')[0]
        filepath = os.path.join(data_dir, file)
        
        df = pd.read_csv(filepath)
        
        # Group by Classifier to process each run separately
        for clf_name, clf_df in df.groupby('Classifier'):
            clf_name = str(clf_name)
            
            # Handle baseline classifiers by attaching trajectory attributes to Target Subject Node
            if clf_name.startswith('baseline'):
                stats = compute_trial_bin_stats(clf_df, num_bins=num_bins)
                if stats is None:
                    continue
                    
                target_types = parse_baseline_classifier_name(clf_name)
                for clf_type in target_types:
                    if clf_type in full_graphs and target_subject in full_graphs[clf_type]:
                        node_attrs = {
                            'baseline_accuracy': stats['final_cum_acc'],
                            'baseline_weight': stats['final_cum_acc'],
                            'baseline_initial_zero_shot_acc': stats['initial_interval_acc'],
                            'baseline_final_interval_acc': stats['final_interval_acc'],
                            'baseline_adaptation_slope': stats['adaptation_slope'],
                            'baseline_total_trials': stats['total_trials'],
                            'baseline_classifier': clf_name,
                            'baseline_bin_interval_accs_json': json.dumps(stats['interval_accs']),
                            'baseline_bin_cumulative_accs_json': json.dumps(stats['cumulative_accs'])
                        }
                        for b_idx, (inter_a, cum_a) in enumerate(zip(stats['interval_accs'], stats['cumulative_accs']), 1):
                            node_attrs[f'baseline_bin_interval_acc_{b_idx}'] = inter_a
                            node_attrs[f'baseline_bin_cumulative_acc_{b_idx}'] = cum_a

                        full_graphs[clf_type].nodes[target_subject].update(node_attrs)

                        if save_snapshots:
                            for b_idx in range(num_bins):
                                snapshot_graphs[clf_type][b_idx].nodes[target_subject].update({
                                    'baseline_weight': stats['interval_accs'][b_idx],
                                    'baseline_cumulative_weight': stats['cumulative_accs'][b_idx]
                                })
                continue

            src_subject, clf_type = parse_classifier_name(clf_name)

            if src_subject and clf_type in full_graphs:
                stats = compute_trial_bin_stats(clf_df, num_bins=num_bins)
                if stats is None:
                    continue

                # Build edge attributes dictionary for full GraphML
                edge_attrs = {
                    'weight': stats['final_cum_acc'],
                    'initial_zero_shot_acc': stats['initial_interval_acc'],
                    'final_interval_acc': stats['final_interval_acc'],
                    'adaptation_slope': stats['adaptation_slope'],
                    'total_trials': stats['total_trials'],
                    'bin_interval_accs_json': json.dumps(stats['interval_accs']),
                    'bin_cumulative_accs_json': json.dumps(stats['cumulative_accs'])
                }

                # Add individual bin attributes for easy scalar querying in NetworkX / PyG
                for b_idx, (inter_a, cum_a) in enumerate(zip(stats['interval_accs'], stats['cumulative_accs']), 1):
                    edge_attrs[f'bin_interval_acc_{b_idx}'] = inter_a
                    edge_attrs[f'bin_cumulative_acc_{b_idx}'] = cum_a

                full_graphs[clf_type].add_edge(src_subject, target_subject, **edge_attrs)

                # Add edges to snapshot graphs
                if save_snapshots:
                    for b_idx in range(num_bins):
                        snapshot_graphs[clf_type][b_idx].add_edge(
                            src_subject,
                            target_subject,
                            weight=stats['interval_accs'][b_idx],
                            cumulative_weight=stats['cumulative_accs'][b_idx]
                        )

                # Append to summary rows list
                row_dict = {
                    'Pipeline': clf_type,
                    'Source_Subject': src_subject,
                    'Target_Subject': target_subject,
                    'Final_Cumulative_Acc': stats['final_cum_acc'],
                    'Zero_Shot_Acc_Bin1': stats['initial_interval_acc'],
                    'Final_Interval_Acc_Bin10': stats['final_interval_acc'],
                    'Adaptation_Slope': stats['adaptation_slope']
                }
                for b_idx, inter_a in enumerate(stats['interval_accs'], 1):
                    row_dict[f'Interval_Acc_Bin_{b_idx}'] = inter_a
                summary_rows.append(row_dict)

    # Save GraphML files, snapshots, and heatmaps
    print("\nSaving GraphML files and heatmaps...")
    for name, g in full_graphs.items():
        print(f"Graph '{name}': {g.number_of_nodes()} nodes, {g.number_of_edges()} edges.")
        
        # Save full vector-attributed GraphML
        graphml_path = os.path.join(output_dir, f"{name}_binned.graphml")
        nx.write_graphml(g, graphml_path)

        # Save snapshot GraphMLs
        if save_snapshots:
            for b_idx in range(num_bins):
                snap_path = os.path.join(snapshots_dir, f"{name}_bin_{b_idx + 1:02d}.graphml")
                nx.write_graphml(snapshot_graphs[name][b_idx], snap_path)

        # Save heatmaps
        if save_heatmaps and g.number_of_edges() > 0:
            adj_zero_shot = nx.to_pandas_adjacency(g, nodelist=subjects, weight='bin_interval_acc_1')
            adj_final = nx.to_pandas_adjacency(g, nodelist=subjects, weight='weight')
            adj_slope = nx.to_pandas_adjacency(g, nodelist=subjects, weight='adaptation_slope')

            # Populate diagonal (Source s == Target s) with baseline classifier accuracies
            for s in subjects:
                if s in g.nodes:
                    node_data = g.nodes[s]
                    if 'baseline_bin_interval_acc_1' in node_data:
                        adj_zero_shot.loc[s, s] = float(node_data['baseline_bin_interval_acc_1'])
                    if 'baseline_accuracy' in node_data:
                        adj_final.loc[s, s] = float(node_data['baseline_accuracy'])

            fig, axes = plt.subplots(1, 3, figsize=(24, 7))
            
            sns.heatmap(adj_zero_shot, cmap="viridis", vmin=0, vmax=1, ax=axes[0])
            axes[0].set_title(f"{name}\nZero-Shot Accuracy (Bin 1)")
            axes[0].set_xlabel("Target Subject")
            axes[0].set_ylabel("Source Subject")

            sns.heatmap(adj_final, cmap="viridis", vmin=0, vmax=1, ax=axes[1])
            axes[1].set_title(f"{name}\nFinal Accuracy (Bin 10)")
            axes[1].set_xlabel("Target Subject")
            axes[1].set_ylabel("Source Subject")

            sns.heatmap(adj_slope, cmap="coolwarm", center=0, ax=axes[2])
            axes[2].set_title(f"{name}\nAdaptation Velocity (Slope)")
            axes[2].set_xlabel("Target Subject")
            axes[2].set_ylabel("Source Subject")

            plt.tight_layout()
            plot_path = os.path.join(plots_dir, f"{name}_temporal_heatmap.png")
            plt.savefig(plot_path, dpi=150)
            plt.close()

    # Save summary dataframe to CSV
    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = os.path.join(output_dir, "binned_simulation_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Saved full binned summary CSV to: {summary_csv_path}")
    print(f"Temporal binned graph generation complete! Artifacts saved to: {output_dir}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Construct temporal trial-binned performance graphs from simulation CSVs."
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
        help="Output directory for graphml and heatmaps. Default: graph_results/graphs_binned/{dataset}"
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=10,
        help="Number of temporal trial bins to compute (Default: 10)."
    )
    parser.add_argument(
        "--no-snapshots",
        action="store_true",
        help="Do not save per-bin snapshot GraphML files."
    )
    parser.add_argument(
        "--no-heatmaps",
        action="store_true",
        help="Do not generate heatmap PNG plots."
    )

    args = parser.parse_args()

    data_dir = args.data_dir or os.path.join("simulation_results", args.dataset)
    output_dir = args.output_dir or os.path.join("graph_results", "graphs_binned", args.dataset)

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Input directory '{data_dir}' does not exist.")

    build_binned_graphs(
        data_dir=data_dir,
        output_dir=output_dir,
        num_bins=args.num_bins,
        save_snapshots=not args.no_snapshots,
        save_heatmaps=not args.no_heatmaps
    )


if __name__ == "__main__":
    main()
