"""
4-Quadrant Subject Taxonomy Analysis for Cross-Subject EEG Graphs
==================================================================

This script classifies subjects into a 4-Quadrant Subject Taxonomy based on their directed transfer performance:
1. Quadrant I   : Universal Donors      (High Outgoing Accuracy, High Incoming Accuracy)
2. Quadrant II  : Universal Recipients  (Low Outgoing Accuracy, High Incoming Accuracy)
3. Quadrant III : Isolated / Refractory (Low Outgoing Accuracy, Low Incoming Accuracy)
4. Quadrant IV  : Specialist Donors     (High Outgoing Accuracy, Low Incoming Accuracy)

By default, it performs separate 4-quadrant taxonomy analyses for all 6 classifier configurations
(3 pipelines: CSP+LDA, CSP+SVM, Cov+Tangent+LR  x  2 modes: static and adaptive).

Outputs:
--------
- 6 Individual 2D Quadrant Scatter Plot PNGs (one per classifier configuration).
- 1 Combined 6-Panel Comparison Grid PNG (2x3 side-by-side layout comparing all configurations).
- Classification summary tables exported to CSV, Markdown, and JSON.

Usage Examples:
---------------
1. Run taxonomy analysis for all 6 classifier configurations separately (default):
    python graph_tools/subject_taxonomy.py --dataset Dreyer2023

2. Analyze a specific classifier configuration:
    python graph_tools/subject_taxonomy.py --dataset Dreyer2023 --pipeline Cov_Tangent_Space_LR_pipeline_static

3. Customize threshold metric (e.g. median instead of mean):
    python graph_tools/subject_taxonomy.py --dataset Dreyer2023 --threshold-metric median
"""

import os
import glob
import argparse
import json
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns


def find_graph_files(
    input_path: str,
    dataset: str = "Dreyer2023",
    pipeline: str = None
) -> list[str]:
    """Find matching GraphML files based on dataset and optional pipeline filter."""
    if os.path.isfile(input_path):
        if input_path.endswith('.graphml'):
            return [os.path.abspath(input_path)]
        else:
            raise ValueError(f"Specified file '{input_path}' is not a .graphml file.")

    search_dir = input_path
    if dataset and dataset.lower() != 'all':
        candidates = [
            os.path.join(input_path, "graphs", dataset),
            os.path.join(input_path, dataset),
            os.path.join("graph_results", "graphs", dataset),
            os.path.join("graph_results", dataset),
            os.path.join("simulation_results", "graphs", dataset),
            os.path.join("simulation_results", dataset),
            os.path.join(input_path, "graphs_binned", dataset),
            os.path.join("graph_results", "graphs_binned", dataset),
            os.path.join(input_path, "graphs_filtered", dataset),
            os.path.join("graph_results", "graphs_filtered", dataset),
        ]
        found_dir = None
        for cand in candidates:
            if os.path.exists(cand) and glob.glob(os.path.join(cand, "*.graphml")):
                found_dir = cand
                break

        if not found_dir:
            raise FileNotFoundError(
                f"No graph directory containing .graphml files was found for dataset '{dataset}'.\n"
                f"Please ensure graph files exist in 'graph_results/graphs/{dataset}' (e.g. by running build_graphs.py)."
            )
        search_dir = found_dir

    if not os.path.exists(search_dir):
        raise FileNotFoundError(f"Search directory '{search_dir}' does not exist.")

    files = glob.glob(os.path.join(search_dir, "**", "*.graphml"), recursive=True)

    filtered_files = []
    for f in files:
        basename = os.path.basename(f)
        # Avoid already filtered/snapshot output files
        if "_above_baseline" in basename or "_bin_" in basename:
            continue
        # Avoid binned graphml files unless explicitly requested in pipeline argument
        if "binned" in basename.lower() and (not pipeline or "binned" not in pipeline.lower()):
            continue
        if pipeline and pipeline.lower() not in basename.lower():
            continue
        filtered_files.append(os.path.abspath(f))

    return sorted(filtered_files)


def compute_subject_taxonomy_metrics(
    graph_path: str,
    exclude_self: bool = True,
    threshold_metric: str = "mean",
    custom_out_threshold: float = None,
    custom_in_threshold: float = None
) -> tuple[pd.DataFrame, float, float]:
    """
    Load a single GraphML file, calculate incoming & outgoing degree stats, run HITS centrality,
    and classify subjects into the 4 Quadrants.
    """
    G = nx.read_graphml(graph_path)

    # Calculate HITS Hubs & Authorities scores
    try:
        hubs, authorities = nx.hits(G, max_iter=500, normalized=True)
    except Exception:
        hubs = {n: 0.0 for n in G.nodes()}
        authorities = {n: 0.0 for n in G.nodes()}

    subject_stats = {}
    for node in G.nodes():
        subject_stats[node] = {
            'out_weights': [],
            'in_weights': [],
            'hub_score': hubs.get(node, 0.0),
            'auth_score': authorities.get(node, 0.0)
        }

        out_edges = G.out_edges(node, data=True)
        in_edges = G.in_edges(node, data=True)

        if exclude_self:
            out_edges = [(u, v, d) for u, v, d in out_edges if u != v]
            in_edges = [(u, v, d) for u, v, d in in_edges if u != v]

        for _, _, d in out_edges:
            if 'weight' in d:
                subject_stats[node]['out_weights'].append(float(d['weight']))

        for _, _, d in in_edges:
            if 'weight' in d:
                subject_stats[node]['in_weights'].append(float(d['weight']))

    rows = []
    for sub_id, data in subject_stats.items():
        out_w = data['out_weights']
        in_w = data['in_weights']

        out_mean = float(np.mean(out_w)) if out_w else 0.0
        in_mean = float(np.mean(in_w)) if in_w else 0.0
        out_std = float(np.std(out_w)) if out_w else 0.0
        in_std = float(np.std(in_w)) if in_w else 0.0

        rows.append({
            'Subject': str(sub_id),
            'Out_Mean_Acc': out_mean,
            'In_Mean_Acc': in_mean,
            'Out_Std': out_std,
            'In_Std': in_std,
            'Acc_Diff_Out_Minus_In': out_mean - in_mean,
            'Acc_Ratio_Out_Div_In': (out_mean / in_mean) if in_mean > 0 else 0.0,
            'Hub_Score': float(data['hub_score']),
            'Authority_Score': float(data['auth_score'])
        })

    df = pd.DataFrame(rows)

    # Determine thresholds
    if custom_out_threshold is not None:
        out_threshold = custom_out_threshold
    else:
        out_threshold = float(df['Out_Mean_Acc'].median() if threshold_metric.lower() == 'median' else df['Out_Mean_Acc'].mean())

    if custom_in_threshold is not None:
        in_threshold = custom_in_threshold
    else:
        in_threshold = float(df['In_Mean_Acc'].median() if threshold_metric.lower() == 'median' else df['In_Mean_Acc'].mean())

    def assign_quadrant(row):
        is_high_out = row['Out_Mean_Acc'] >= out_threshold
        is_high_in = row['In_Mean_Acc'] >= in_threshold

        if is_high_out and is_high_in:
            return 'Q1: Universal Donor'
        elif not is_high_out and is_high_in:
            return 'Q2: Universal Recipient'
        elif not is_high_out and not is_high_in:
            return 'Q3: Isolated/Refractory'
        else:
            return 'Q4: Specialist Donor'

    df['Quadrant'] = df.apply(assign_quadrant, axis=1)

    # Sort numerically by Subject ID
    def try_int(val):
        try:
            return int(val)
        except ValueError:
            return val

    df['Subject_Num'] = df['Subject'].apply(try_int)
    df = df.sort_values(by=['Quadrant', 'Out_Mean_Acc', 'Subject_Num'], ascending=[True, False, True]).reset_index(drop=True)
    df.drop(columns=['Subject_Num'], inplace=True)
    df.index += 1
    df.index.name = 'Rank'

    return df, out_threshold, in_threshold


def plot_subject_taxonomy_quadrants(
    df: pd.DataFrame,
    out_threshold: float,
    in_threshold: float,
    title_suffix: str = "",
    save_path: str = None
):
    """Generate a 4-Quadrant Scatter Plot visualizing subject incoming vs. outgoing accuracy."""
    fig, ax = plt.subplots(figsize=(11, 9))

    color_map = {
        'Q1: Universal Donor': '#2ca02c',       # Green
        'Q2: Universal Recipient': '#1f77b4',   # Blue
        'Q3: Isolated/Refractory': '#d62728',   # Red
        'Q4: Specialist Donor': '#ff7f0e'       # Orange
    }

    sns.scatterplot(
        data=df,
        x='In_Mean_Acc',
        y='Out_Mean_Acc',
        hue='Quadrant',
        palette=color_map,
        s=110,
        edgecolor='black',
        linewidth=0.8,
        ax=ax
    )

    # Draw threshold boundary lines
    ax.axhline(out_threshold, color='black', linestyle='--', linewidth=1.5, label=f'Out Thresh ({out_threshold:.4f})')
    ax.axvline(in_threshold, color='black', linestyle=':', linewidth=1.5, label=f'In Thresh ({in_threshold:.4f})')

    # Label subjects on the plot
    for _, row in df.iterrows():
        ax.text(
            row['In_Mean_Acc'] + 0.0008,
            row['Out_Mean_Acc'] + 0.0008,
            row['Subject'],
            fontsize=7.5,
            alpha=0.85
        )

    # Shade Quadrants
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    ax.fill_between([in_threshold, x_max], out_threshold, y_max, color='#2ca02c', alpha=0.07)  # Q1
    ax.fill_between([in_threshold, x_max], y_min, out_threshold, color='#1f77b4', alpha=0.07)  # Q2
    ax.fill_between([x_min, in_threshold], y_min, out_threshold, color='#d62728', alpha=0.07)  # Q3
    ax.fill_between([x_min, in_threshold], out_threshold, y_max, color='#ff7f0e', alpha=0.07)  # Q4

    ax.set_title(f"4-Quadrant Subject Taxonomy\n({title_suffix})", fontsize=13, fontweight='bold')
    ax.set_xlabel("Incoming Accuracy (Receptivity / How well other models perform on Subject)", fontsize=10)
    ax.set_ylabel("Outgoing Accuracy (Generalization / How well Subject model performs on others)", fontsize=10)
    ax.legend(title="Subject Category", loc='upper left', bbox_to_anchor=(1.02, 1))
    ax.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved 4-Quadrant taxonomy plot to: {save_path}")

    plt.close()


def plot_multi_panel_taxonomy_comparison(
    all_results: dict,
    dataset: str,
    save_path: str
):
    """
    Generate a 2x3 grid figure comparing all 6 classifier configurations side-by-side.
    (Rows: Static vs. Adaptive; Columns: CSP+LDA, CSP+SVM, Cov+Tangent+LR)
    """
    color_map = {
        'Q1: Universal Donor': '#2ca02c',
        'Q2: Universal Recipient': '#1f77b4',
        'Q3: Isolated/Refractory': '#d62728',
        'Q4: Specialist Donor': '#ff7f0e'
    }

    # Group graphs by pipeline & mode
    columns_order = ['CSP_LDA_pipeline', 'CSP_SVM_pipeline', 'Cov_Tangent_Space_LR_pipeline']
    rows_order = ['static', 'adaptive']

    fig, axes = plt.subplots(2, 3, figsize=(24, 14))

    for r_idx, mode in enumerate(rows_order):
        for c_idx, pipe_prefix in enumerate(columns_order):
            ax = axes[r_idx, c_idx]
            
            matching_key = None
            for key in all_results.keys():
                if pipe_prefix in key and mode in key:
                    matching_key = key
                    break

            if not matching_key or matching_key not in all_results:
                ax.text(0.5, 0.5, f"No Data: {pipe_prefix}_{mode}", ha='center', va='center')
                ax.axis('off')
                continue

            df, out_thresh, in_thresh = all_results[matching_key]

            sns.scatterplot(
                data=df,
                x='In_Mean_Acc',
                y='Out_Mean_Acc',
                hue='Quadrant',
                palette=color_map,
                s=80,
                edgecolor='black',
                linewidth=0.6,
                ax=ax
            )

            ax.axhline(out_thresh, color='black', linestyle='--', linewidth=1.2)
            ax.axvline(in_thresh, color='black', linestyle=':', linewidth=1.2)

            for _, row in df.iterrows():
                ax.text(row['In_Mean_Acc'] + 0.0005, row['Out_Mean_Acc'] + 0.0005, row['Subject'], fontsize=6.5, alpha=0.75)

            # Title
            clean_pipe_name = pipe_prefix.replace("_pipeline", "").replace("_", " ")
            ax.set_title(f"{clean_pipe_name} ({mode.upper()})\nOut Thresh: {out_thresh:.4f} | In Thresh: {in_thresh:.4f}", fontsize=11, fontweight='bold')
            ax.set_xlabel("Incoming Accuracy")
            ax.set_ylabel("Outgoing Accuracy")
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.get_legend().remove()

    # Add shared legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=col, markersize=10, label=label)
               for label, col in color_map.items()]
    fig.legend(handles=handles, title="Subject Taxonomy Category", loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=4, fontsize=12)

    plt.suptitle(f"4-Quadrant Subject Taxonomy 6-Panel Comparison ({dataset})", fontsize=16, fontweight='bold', y=1.05)
    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved 6-Panel comparison taxonomy grid plot to: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Classify subjects into 4-Quadrant Subject Taxonomies separately for all 6 classifier configurations."
    )
    parser.add_argument(
        "--input", "-i",
        default="graph_results",
        help="Path to graphs directory or graph_results folder. Default: graph_results"
    )
    parser.add_argument(
        "--dataset", "-d",
        default="Dreyer2023",
        help="Dataset name ('Dreyer2023', 'PhysionetMI', etc.). Default: Dreyer2023"
    )
    parser.add_argument(
        "--pipeline", "-p",
        default=None,
        help="Filter specific pipeline configuration (e.g., 'Cov_Tangent_Space_LR_pipeline_static', 'CSP_LDA', etc.)."
    )
    parser.add_argument(
        "--threshold-metric",
        default="mean",
        choices=["mean", "median"],
        help="Metric to divide quadrant threshold lines. Default: mean"
    )
    parser.add_argument(
        "--threshold-out",
        type=float,
        default=None,
        help="Custom fixed Outgoing Accuracy threshold float."
    )
    parser.add_argument(
        "--threshold-in",
        type=float,
        default=None,
        help="Custom fixed Incoming Accuracy threshold float."
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help="Output directory for taxonomy reports and plots. Default: graph_results/taxonomy/{dataset}"
    )

    args = parser.parse_args()

    graph_files = find_graph_files(
        input_path=args.input,
        dataset=args.dataset,
        pipeline=args.pipeline
    )

    if not graph_files:
        raise FileNotFoundError(f"No matching GraphML files found for dataset='{args.dataset}' in '{args.input}'.")

    output_dir = args.output_dir or os.path.join("graph_results", "taxonomy", args.dataset)
    os.makedirs(output_dir, exist_ok=True)

    print("================================================================================")
    print(" 4-Quadrant Subject Taxonomy Analysis (Separate Configuration Mode)")
    print("================================================================================")
    print(f" Dataset             : {args.dataset}")
    print(f" Matched Graph Files : {len(graph_files)}")
    for f in graph_files:
        print(f"   - {os.path.basename(f)}")
    print(f" Threshold Metric    : {args.threshold_metric.upper()}")
    print(f" Output Directory    : {output_dir}")
    print("================================================================================\n")

    all_results = {}
    summary_rows = []

    for g_file in graph_files:
        pipe_name = os.path.splitext(os.path.basename(g_file))[0]

        df, out_thresh, in_thresh = compute_subject_taxonomy_metrics(
            graph_path=g_file,
            threshold_metric=args.threshold_metric,
            custom_out_threshold=args.threshold_out,
            custom_in_threshold=args.threshold_in
        )

        all_results[pipe_name] = (df, out_thresh, in_thresh)

        # Print breakdown for this pipeline configuration
        print(f"--- Configuration: '{pipe_name}' ---")
        print(f"  - Outgoing Accuracy Threshold (Y-axis) : {out_thresh:.4f}")
        print(f"  - Incoming Accuracy Threshold (X-axis) : {in_thresh:.4f}")
        counts = df['Quadrant'].value_counts()
        for q_name in ['Q1: Universal Donor', 'Q2: Universal Recipient', 'Q3: Isolated/Refractory', 'Q4: Specialist Donor']:
            c = counts.get(q_name, 0)
            print(f"  {q_name:<25}: {c} subjects ({c / len(df) * 100:.1f}%)")

            summary_rows.append({
                'Pipeline_Config': pipe_name,
                'Quadrant_Category': q_name,
                'Subject_Count': c,
                'Percentage': round(c / len(df) * 100, 2),
                'Out_Threshold': round(out_thresh, 4),
                'In_Threshold': round(in_thresh, 4)
            })
        print()

        # Save individual plot & CSV for this configuration
        pipe_plot_path = os.path.join(output_dir, f"{pipe_name}_taxonomy_quadrant.png")
        plot_subject_taxonomy_quadrants(
            df=df,
            out_threshold=out_thresh,
            in_threshold=in_thresh,
            title_suffix=f"{args.dataset}: {pipe_name}",
            save_path=pipe_plot_path
        )

        pipe_csv_path = os.path.join(output_dir, f"{pipe_name}_taxonomy.csv")
        df.to_csv(pipe_csv_path)
        print(f"  - Saved individual CSV to : {pipe_csv_path}\n")

    # Generate 6-Panel Comparison Grid Figure if multiple graph configurations matched
    if len(all_results) >= 2:
        comparison_plot_path = os.path.join(output_dir, f"{args.dataset}_taxonomy_6_panel_comparison.png")
        plot_multi_panel_taxonomy_comparison(
            all_results=all_results,
            dataset=args.dataset,
            save_path=comparison_plot_path
        )

    # Save overall summary CSV
    df_summary = pd.DataFrame(summary_rows)
    summary_csv_path = os.path.join(output_dir, "taxonomy_summary.csv")
    df_summary.to_csv(summary_csv_path, index=False)
    print(f"Saved taxonomy summary CSV to: {summary_csv_path}")

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

    # Save overall Markdown report
    md_content = [
        f"# 4-Quadrant Subject Taxonomy Multi-Configuration Report ({args.dataset})",
        f"",
        f"- **Dataset**: `{args.dataset}`",
        f"- **Analyzed Configurations**: `{len(all_results)} pipeline configurations`",
        f"- **Threshold Metric**: `{args.threshold_metric.upper()}`",
        f"",
        f"## Quadrant Distribution Breakdown Across Configurations",
        f"",
        df_to_markdown(df_summary),
        f"",
        f"## Output Files & Visualizations",
        f"- Individual plots and CSV files saved in: `{output_dir}`"
    ]
    if len(all_results) >= 2:
        grid_plot_rel = f"{args.dataset}_taxonomy_6_panel_comparison.png"
        md_content.append(f"\n### 6-Panel Comparison Grid\n![Taxonomy Grid]({grid_plot_rel})")

    summary_md_path = os.path.join(output_dir, "taxonomy_report.md")
    with open(summary_md_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(md_content))
    print(f"Saved Markdown summary report to: {summary_md_path}\n")


if __name__ == "__main__":
    main()
