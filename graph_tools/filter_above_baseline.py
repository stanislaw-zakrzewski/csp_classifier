"""
Filter Graph Edges Above Baseline Accuracy & Top Percentages
============================================================

This script loads cross-subject directed performance graphs (.graphml), compares each edge weight
(accuracy of Source u -> Target v) against Target Subject v's within-subject baseline accuracy stored on Node v,
and removes all edges where the cross-subject accuracy is lower than the baseline.

Additionally, an optional `--top-edges-percent` parameter allows selecting only the top N% highest accuracy
edges out of all graph edges (e.g. 0.05 or 5 for top 5% best edges).

Outputs:
--------
1. Filtered GraphML file (*_above_baseline.graphml).
2. Side-by-side Heatmaps comparing:
   - Full Unfiltered Performance Matrix
   - Filtered Matrix (Edges >= Baseline & Top N% Best Edges)
   - Relative Gain Matrix (Cross-subject Acc - Baseline Acc)
3. Network Graph Diagram visualizing the surviving super-generalizing connections.
4. Summary Report (Markdown/CSV/JSON) detailing retention rates and top surviving donor models.

Usage Examples:
---------------
1. Filter & visualize static models (edges >= baseline):
    python graph_tools/filter_above_baseline.py --dataset Dreyer2023

2. Keep top 5% best edges out of all graph edges (e.g. 366 edges for 7310 total):
    python graph_tools/filter_above_baseline.py --dataset Dreyer2023 --top-edges-percent 0.05

3. Combine baseline filter with top 10% best edges:
    python graph_tools/filter_above_baseline.py --dataset Dreyer2023 --top-edges-percent 10

4. Filter binned graph at a specific trial bin (e.g. Bin 10):
    python graph_tools/filter_above_baseline.py --dataset Dreyer2023 --bin 10
"""

import os
import glob
import json
import argparse
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns


def find_graphml_files(
    input_path: str,
    dataset: str = "Dreyer2023",
    pipeline: str = None,
    bin_idx: int = None
) -> list[str]:
    """Find matching GraphML files based on input arguments."""
    if os.path.isfile(input_path):
        if input_path.endswith('.graphml'):
            return [os.path.abspath(input_path)]
        else:
            raise ValueError(f"Specified file '{input_path}' is not a .graphml file.")

    search_dir = input_path
    if dataset and dataset.lower() != 'all':
        candidates = [
            os.path.join(input_path, "graphs", dataset),
            os.path.join(input_path, "graphs_binned", dataset),
            os.path.join(input_path, "graphs_filtered", dataset),
            os.path.join(input_path, dataset),
            os.path.join("graph_results", "graphs", dataset),
            os.path.join("graph_results", "graphs_binned", dataset),
            os.path.join("graph_results", "graphs_filtered", dataset),
            os.path.join("graph_results", dataset),
            os.path.join("simulation_results", "graphs", dataset),
            os.path.join("simulation_results", dataset)
        ]
        found_dir = None
        for cand in candidates:
            if os.path.exists(cand) and glob.glob(os.path.join(cand, "*.graphml")):
                found_dir = cand
                break
        search_dir = found_dir or search_dir

    if not os.path.exists(search_dir):
        raise FileNotFoundError(f"Directory '{search_dir}' does not exist.")

    files = glob.glob(os.path.join(search_dir, "**", "*.graphml"), recursive=True)

    filtered_files = []
    for f in files:
        basename = os.path.basename(f)
        # Avoid already filtered output files
        if "_above_baseline" in basename:
            continue
        if pipeline and pipeline.lower() not in basename.lower():
            continue
        filtered_files.append(os.path.abspath(f))

    return sorted(filtered_files)


def filter_graph_above_baseline(
    graph_path: str,
    bin_idx: int = None,
    exclude_self: bool = True,
    margin: float = 0.0,
    top_edges_ratio: float = None
) -> tuple[nx.DiGraph, nx.DiGraph, dict]:
    """
    Load a GraphML file and filter out edges below baseline accuracy or not in top N% edges.

    Parameters
    ----------
    graph_path : str
        Path to the input GraphML file.
    bin_idx : int, optional
        Specific bin index (1..10) if analyzing a binned graph. If None, uses default 'weight'.
    exclude_self : bool
        If True, exclude self-loops from graph edge analysis.
    margin : float
        Additional threshold margin (e.g. margin=0.02 requires edge accuracy >= baseline + 0.02).
    top_edges_ratio : float, optional
        Optional ratio of overall best edges to keep (e.g. 0.05 for top 5% best edges).

    Returns
    -------
    G_orig : nx.DiGraph
        Original unfiltered graph.
    G_filtered : nx.DiGraph
        Filtered graph containing retained edges.
    stats : dict
        Summary statistics of the filtering process.
    """
    G_orig = nx.read_graphml(graph_path)
    G_filtered = G_orig.copy()

    # Determine weight attribute key and baseline node attribute key
    if bin_idx is not None:
        weight_key = f'bin_interval_acc_{bin_idx}'
        node_baseline_key = f'baseline_bin_interval_acc_{bin_idx}'
    else:
        weight_key = 'weight'
        node_baseline_key = 'baseline_accuracy'

    candidate_edges = []

    for u, v, d in G_orig.edges(data=True):
        if exclude_self and u == v:
            continue

        # Get edge weight
        w = float(d.get(weight_key, d.get('weight', 0.0)))
        
        # Get target node baseline accuracy
        target_node = G_orig.nodes[v]
        b = float(target_node.get(node_baseline_key, target_node.get('baseline_weight', target_node.get('baseline_accuracy', 0.0))))

        gain = w - b

        # Check baseline threshold condition
        if w >= (b + margin):
            candidate_edges.append({
                'source': u,
                'target': v,
                'edge_accuracy': w,
                'baseline_accuracy': b,
                'relative_gain': gain
            })

    # Sort candidates by edge accuracy descending
    candidate_edges.sort(key=lambda x: x['edge_accuracy'], reverse=True)

    total_possible_edges = len([(u, v) for u, v in G_orig.edges() if (u != v or not exclude_self)])

    # Apply top edges ratio filter if requested
    if top_edges_ratio is not None and 0.0 < top_edges_ratio <= 1.0:
        n_keep = max(1, int(round(total_possible_edges * top_edges_ratio)))
        retained_edge_details = candidate_edges[:n_keep]
    else:
        retained_edge_details = candidate_edges

    retained_set = {(e['source'], e['target']) for e in retained_edge_details}

    # Identify edges to remove
    edges_to_remove = [(u, v) for u, v in G_orig.edges() if (u, v) not in retained_set]
    G_filtered.remove_edges_from(edges_to_remove)

    retained_edges = G_filtered.number_of_edges()
    retention_rate = (retained_edges / total_possible_edges * 100.0) if total_possible_edges > 0 else 0.0

    avg_gain = float(np.mean([e['relative_gain'] for e in retained_edge_details])) if retained_edge_details else 0.0
    max_gain = float(np.max([e['relative_gain'] for e in retained_edge_details])) if retained_edge_details else 0.0
    min_retained_acc = float(np.min([e['edge_accuracy'] for e in retained_edge_details])) if retained_edge_details else 0.0

    stats = {
        'graph_name': os.path.splitext(os.path.basename(graph_path))[0],
        'total_possible_edges': total_possible_edges,
        'retained_edges': retained_edges,
        'removed_edges': len(edges_to_remove),
        'retention_rate_pct': round(retention_rate, 2),
        'top_edges_ratio_applied': top_edges_ratio,
        'min_retained_accuracy': round(min_retained_acc, 4),
        'avg_relative_gain': round(avg_gain, 4),
        'max_relative_gain': round(max_gain, 4),
        'weight_key_used': weight_key,
        'baseline_key_used': node_baseline_key,
        'edge_details': retained_edge_details
    }

    return G_orig, G_filtered, stats


def plot_above_baseline_visualizations(
    G_orig: nx.DiGraph,
    G_filtered: nx.DiGraph,
    stats: dict,
    output_dir: str
):
    """
    Generate side-by-side Heatmaps and a Directed Network Layout Plot.
    """
    os.makedirs(output_dir, exist_ok=True)
    graph_name = stats['graph_name']
    subjects = sorted(list(G_orig.nodes()), key=lambda x: int(x) if x.isdigit() else x)

    weight_key = stats['weight_key_used']
    node_baseline_key = stats['baseline_key_used']

    # 1. Build Adjacency Matrices
    adj_orig = nx.to_pandas_adjacency(G_orig, nodelist=subjects, weight=weight_key)
    adj_filtered = nx.to_pandas_adjacency(G_filtered, nodelist=subjects, weight=weight_key)

    # Build Baseline Vector & Relative Gain Matrix
    baselines = np.array([
        float(G_orig.nodes[s].get(node_baseline_key, G_orig.nodes[s].get('baseline_weight', G_orig.nodes[s].get('baseline_accuracy', 0.0))))
        for s in subjects
    ])

    # Broadcast target baseline across columns
    gain_matrix = adj_orig.values - baselines[np.newaxis, :]
    np.fill_diagonal(gain_matrix, 0)

    # Filtered Gain Matrix (mask below baseline/top filter)
    filtered_gain_matrix = np.where(adj_filtered.values > 0, gain_matrix, np.nan)
    df_filtered_gain = pd.DataFrame(filtered_gain_matrix, index=subjects, columns=subjects)

    # Plot 1: 3-Panel Heatmap Comparison
    fig, axes = plt.subplots(1, 3, figsize=(26, 8))

    # Heatmap 1: Full Unfiltered Accuracies
    sns.heatmap(adj_orig, cmap="viridis", vmin=0, vmax=1, ax=axes[0])
    axes[0].set_title(f"1. Unfiltered Cross-Subject Accuracy\n({graph_name})", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Target Subject")
    axes[0].set_ylabel("Source Subject")

    # Heatmap 2: Retained Accuracies
    filter_label = f" (Top {stats['top_edges_ratio_applied']*100:.1f}% edges)" if stats['top_edges_ratio_applied'] else ""
    sns.heatmap(adj_filtered, cmap="viridis", vmin=0, vmax=1, ax=axes[1], cbar_kws={'label': 'Accuracy'})
    axes[1].set_title(f"2. Retained Transfers{filter_label}\n({stats['retained_edges']} / {stats['total_possible_edges']} Edges: {stats['retention_rate_pct']}%)", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Target Subject")
    axes[1].set_ylabel("Source Subject")

    # Heatmap 3: Relative Gain (Accuracy - Baseline)
    sns.heatmap(df_filtered_gain, cmap="YlGnBu", ax=axes[2], cbar_kws={'label': 'Accuracy Gain over Baseline'})
    axes[2].set_title(f"3. Relative Gain over Baseline (Acc - Baseline)\n(Avg Gain: +{stats['avg_relative_gain']:.4f}, Min Acc Kept: {stats['min_retained_accuracy']:.4f})", fontsize=12, fontweight='bold')
    axes[2].set_xlabel("Target Subject")
    axes[2].set_ylabel("Source Subject")

    # Set tick spacing
    for ax in axes:
        ax.set_xticks(range(0, len(subjects), 10))
        ax.set_xticklabels([subjects[i] for i in range(0, len(subjects), 10)])
        ax.set_yticks(range(0, len(subjects), 10))
        ax.set_yticklabels([subjects[i] for i in range(0, len(subjects), 10)])

    plt.tight_layout()
    heatmap_path = os.path.join(output_dir, f"{graph_name}_above_baseline_heatmaps.png")
    plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 2: Directed Network Diagram of Surviving Super-Generalizing Edges
    fig, ax = plt.subplots(figsize=(14, 12))

    # Node positions using circular layout
    pos = nx.circular_layout(G_filtered)

    # Node sizes & colors based on Out-Degree (Surviving Donors)
    out_degrees = dict(G_filtered.out_degree())
    node_colors = [out_degrees.get(n, 0) for n in G_filtered.nodes()]
    node_sizes = [300 + out_degrees.get(n, 0) * 15 for n in G_filtered.nodes()]

    # Draw Nodes
    nodes_collection = nx.draw_networkx_nodes(
        G_filtered,
        pos,
        node_color=node_colors,
        cmap=plt.cm.plasma,
        node_size=node_sizes,
        alpha=0.9,
        edgecolors='black',
        ax=ax
    )

    # Draw Edges
    nx.draw_networkx_edges(
        G_filtered,
        pos,
        edge_color='#1f77b4',
        alpha=0.25,
        arrows=True,
        arrowsize=10,
        ax=ax
    )

    # Draw Node Labels
    nx.draw_networkx_labels(G_filtered, pos, font_size=8, font_weight='bold', ax=ax)

    plt.colorbar(nodes_collection, ax=ax, label='Surviving Outgoing Edges (Donor Strength)')
    ax.set_title(f"Surviving Super-Generalizing Network Graph\n({graph_name}: {stats['retained_edges']} Edges Kept)", fontsize=14, fontweight='bold')
    ax.axis('off')

    plt.tight_layout()
    network_path = os.path.join(output_dir, f"{graph_name}_network_diagram.png")
    plt.savefig(network_path, dpi=150, bbox_inches='tight')
    plt.close()

    return heatmap_path, network_path


def main():
    parser = argparse.ArgumentParser(
        description="Filter graph edges with accuracy lower than target node baseline accuracy or keep top N% best edges."
    )
    parser.add_argument(
        "--input", "-i",
        default="graph_results",
        help="Path to a .graphml file or directory containing graph files. Default: graph_results"
    )
    parser.add_argument(
        "--dataset", "-d",
        default="Dreyer2023",
        help="Dataset name ('Dreyer2023', 'PhysionetMI', etc.). Default: Dreyer2023"
    )
    parser.add_argument(
        "--pipeline", "-p",
        default=None,
        help="Filter specific pipeline (e.g., 'Cov_Tangent', 'CSP_LDA', 'CSP_SVM')."
    )
    parser.add_argument(
        "--bin",
        type=int,
        default=None,
        help="Specific trial bin index (1..10) if analyzing a binned graph."
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.0,
        help="Additional threshold margin above baseline (e.g. 0.02 for baseline + 2%). Default: 0.0"
    )
    parser.add_argument(
        "--top-edges-percent", "--top-percent", "--top-edges-ratio", "-e",
        type=float,
        default=None,
        help="Optional fraction or percentage of overall best edges to keep (e.g. 0.05 or 5 for top 5%% best edges). Example: 0.05 keeps 366 edges for 7310 total edges."
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help="Output directory for filtered graphml and plots. Default: graph_results/graphs_filtered/{dataset}"
    )

    args = parser.parse_args()

    # Normalize top_edges_percent / top_edges_ratio
    top_edges_ratio = args.top_edges_percent
    if top_edges_ratio is not None:
        if top_edges_ratio > 1.0:
            top_edges_ratio /= 100.0
        if not (0.0 < top_edges_ratio <= 1.0):
            raise ValueError(f"top-edges-percent must be between 0 and 1 (or 0 and 100). Got: {args.top_edges_percent}")

    files = find_graphml_files(
        input_path=args.input,
        dataset=args.dataset,
        pipeline=args.pipeline,
        bin_idx=args.bin
    )

    if not files:
        raise FileNotFoundError(f"No matching .graphml files found for dataset='{args.dataset}' in '{args.input}'.")

    output_dir = args.output_dir or os.path.join("graph_results", "graphs_filtered", args.dataset)
    os.makedirs(output_dir, exist_ok=True)

    print("================================================================================")
    print(" Filter Graph Edges Above Target Baseline & Top Edge Percentages")
    print("================================================================================")
    print(f" Input Files Found   : {len(files)}")
    print(f" Dataset             : {args.dataset}")
    print(f" Target Margin       : +{args.margin:.4f}")
    if top_edges_ratio:
        print(f" Top Edges Filter    : Top {top_edges_ratio * 100:.1f}% best edges per graph")
    else:
        print(f" Top Edges Filter    : None (Keep all edges >= baseline)")
    print(f" Bin Filter          : Bin {args.bin}" if args.bin else " Bin Filter          : None (Final Accuracy)")
    print(f" Output Directory    : {output_dir}")
    print("================================================================================\n")

    all_summary_stats = []

    for f_path in files:
        G_orig, G_filtered, stats = filter_graph_above_baseline(
            graph_path=f_path,
            bin_idx=args.bin,
            margin=args.margin,
            top_edges_ratio=top_edges_ratio
        )

        graph_name = stats['graph_name']
        print(f"Processing '{graph_name}':")
        print(f"  - Total Possible Edges : {stats['total_possible_edges']}")
        print(f"  - Retained Edges       : {stats['retained_edges']} ({stats['retention_rate_pct']}%)")
        print(f"  - Min Accuracy Kept    : {stats['min_retained_accuracy']:.4f}")
        print(f"  - Average Gain Over Bsl: +{stats['avg_relative_gain']:.4f}")
        print(f"  - Max Gain Over Bsl    : +{stats['max_relative_gain']:.4f}")

        # Save filtered GraphML
        filtered_graphml_path = os.path.join(output_dir, f"{graph_name}_above_baseline.graphml")
        nx.write_graphml(G_filtered, filtered_graphml_path)
        print(f"  - Saved filtered GraphML : {filtered_graphml_path}")

        # Generate plots
        heatmap_path, network_path = plot_above_baseline_visualizations(
            G_orig=G_orig,
            G_filtered=G_filtered,
            stats=stats,
            output_dir=output_dir
        )
        print(f"  - Saved Heatmaps         : {heatmap_path}")
        print(f"  - Saved Network Diagram  : {network_path}\n")

        all_summary_stats.append(stats)

    # Save summary report
    summary_df = pd.DataFrame([
        {
            'Graph_Name': s['graph_name'],
            'Total_Edges': s['total_possible_edges'],
            'Retained_Edges': s['retained_edges'],
            'Retention_Rate_Pct': s['retention_rate_pct'],
            'Min_Accuracy_Kept': s['min_retained_accuracy'],
            'Avg_Relative_Gain': s['avg_relative_gain'],
            'Max_Relative_Gain': s['max_relative_gain']
        }
        for s in all_summary_stats
    ])
    summary_csv_path = os.path.join(output_dir, "filtered_graphs_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Saved filtering summary CSV to: {summary_csv_path}")

    def df_to_markdown(df_to_format):
        try:
            return df_to_format.to_markdown(index=False)
        except Exception:
            # Fallback manual markdown table generator if tabulate is missing
            cols = list(df_to_format.columns)
            header = "| " + " | ".join(cols) + " |"
            divider = "| " + " | ".join(["---"] * len(cols)) + " |"
            rows = []
            for _, row in df_to_format.iterrows():
                rows.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
            return "\n".join([header, divider] + rows)

    # Save Markdown report
    md_content = [
        f"# Above-Baseline Filtered Performance Graphs Report ({args.dataset})",
        f"",
        f"- **Dataset**: `{args.dataset}`",
        f"- **Margin Above Baseline**: `+{args.margin:.4f}`",
        f"- **Top Edges Filter**: `{f'Top {top_edges_ratio*100:.1f}% edges' if top_edges_ratio else 'None'}`",
        f"",
        f"## Filtered Graph Retention Summary",
        f"",
        df_to_markdown(summary_df),
        f"",
        f"## Output Files",
        f"- All filtered GraphML files, 3-panel comparison heatmaps, and directed network diagrams have been saved to: `{output_dir}`"
    ]
    summary_md_path = os.path.join(output_dir, "filtered_graphs_report.md")
    with open(summary_md_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(md_content))
    print(f"Saved Markdown report to: {summary_md_path}\n")


if __name__ == "__main__":
    main()
