"""
Subject Generalization Ranker
=============================

This script analyzes cross-subject directed performance graphs (.graphml) created from EEG classification simulations.
It ranks subjects based on how effectively a classifier trained on each subject's data generalizes to other target subjects.

Usage Examples:
---------------
1. Rank subjects for a specific dataset (default: PhysionetMI):
    python rank_subjects_by_generalization.py --dataset PhysionetMI

2. Rank subjects considering only top 20% best generalizing edges per subject (ratio=0.2):
    python rank_subjects_by_generalization.py --dataset PhysionetMI --top-edges-percent 0.2

3. Rank subjects for a specific graphml file:
    python rank_subjects_by_generalization.py --input simulation_results/graphs/PhysionetMI/Cov_Tangent_Space_LR_pipeline_adaptive.graphml

4. Filter by pipeline and mode:
    python rank_subjects_by_generalization.py --dataset Dreyer2023 --pipeline CSP_LDA_pipeline --mode adaptive

5. Output top 10 subjects only:
    python rank_subjects_by_generalization.py --top-n 10

6. Save results to CSV or Markdown:
    python rank_subjects_by_generalization.py --output-csv generalization_ranking.csv --output-md generalization_ranking.md

7. Print only ordered subject list (for scripting/piping):
    python rank_subjects_by_generalization.py --simple
"""

import os
import glob
import argparse
import json
import pandas as pd
import numpy as np
import networkx as nx


def find_graphml_files(
    input_path: str,
    dataset: str = None,
    pipeline: str = None,
    mode: str = None
) -> list[str]:
    """Find matching GraphML files based on input parameters."""
    if os.path.isfile(input_path):
        if input_path.endswith('.graphml'):
            return [os.path.abspath(input_path)]
        else:
            raise ValueError(f"Specified file '{input_path}' is not a .graphml file.")

    search_dir = input_path
    if dataset and dataset.lower() != 'all':
        candidates = [
            os.path.join(input_path, dataset),
            os.path.join(input_path, "graphs", dataset),
            os.path.join(input_path, "graphs_binned", dataset),
            os.path.join("graph_results", "graphs", dataset),
            os.path.join("graph_results", "graphs_binned", dataset),
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

    pattern = os.path.join(search_dir, "**", "*.graphml") if os.path.isdir(search_dir) else search_dir
    files = glob.glob(pattern, recursive=True)

    # Filtering by pipeline and mode
    filtered_files = []
    for f in files:
        basename = os.path.basename(f)
        if pipeline and pipeline.lower() not in basename.lower():
            continue
        if mode and mode.lower() not in basename.lower():
            continue
        filtered_files.append(os.path.abspath(f))

    return sorted(filtered_files)


def extract_graph_subject_stats(
    graph_path: str,
    exclude_self: bool = True,
    top_edges_ratio: float = 1.0
) -> pd.DataFrame:
    """
    Extract generalization statistics for each source subject in a single GraphML file.
    
    Parameters
    ----------
    graph_path : str
        Path to the GraphML file.
    exclude_self : bool
        Whether to exclude self-loops (Source == Target).
    top_edges_ratio : float
        Ratio of top outgoing edges to retain (between 0.0 and 1.0).
        For example, 0.2 keeps top 20% best edges per subject.

    Returns
    -------
    pd.DataFrame
        DataFrame with generalization statistics per source subject.
    """
    G = nx.read_graphml(graph_path)
    graph_name = os.path.splitext(os.path.basename(graph_path))[0]
    
    rows = []
    for node in G.nodes():
        out_edges = G.out_edges(node, data=True)
        if exclude_self:
            out_edges = [(u, v, d) for u, v, d in out_edges if u != v]
            
        weights = [d.get('weight', 0.0) for _, _, d in out_edges if 'weight' in d]
        total_targets = len(weights)
        
        if weights:
            weights.sort(reverse=True)
            if 0.0 < top_edges_ratio < 1.0:
                n_keep = max(1, int(round(total_targets * top_edges_ratio)))
                weights = weights[:n_keep]

            mean_acc = float(np.mean(weights))
            median_acc = float(np.median(weights))
            std_acc = float(np.std(weights))
            min_acc = float(np.min(weights))
            max_acc = float(np.max(weights))
            count = len(weights)
        else:
            mean_acc = median_acc = std_acc = min_acc = max_acc = 0.0
            count = 0
            
        rows.append({
            'Subject': str(node),
            'Graph_Name': graph_name,
            'Mean_Accuracy': mean_acc,
            'Median_Accuracy': median_acc,
            'Std_Accuracy': std_acc,
            'Min_Accuracy': min_acc,
            'Max_Accuracy': max_acc,
            'Selected_Edges_Count': count,
            'Total_Target_Count': total_targets
        })
        
    return pd.DataFrame(rows)


def rank_subjects_by_generalization(
    input_path: str = "graph_results/graphs",
    dataset: str = "PhysionetMI",
    pipeline: str = None,
    mode: str = None,
    metric: str = "mean",
    exclude_self: bool = True,
    top_edges_ratio: float = 1.0,
    sort_ascending: bool = False
) -> pd.DataFrame:
    """
    Load GraphML files, compute generalization metrics for each source subject,
    and rank subjects based on their cross-subject classification accuracy.

    Parameters
    ----------
    input_path : str
        Path to a graphml file or root directory containing graphml files.
    dataset : str
        Dataset subfolder ('PhysionetMI', 'Dreyer2023', or 'all').
    pipeline : str
        Optional pipeline filter ('Cov_Tangent_Space_LR_pipeline', 'CSP_LDA_pipeline', etc.).
    mode : str
        Optional mode filter ('static', 'adaptive').
    metric : str
        Metric used for primary ranking ('mean', 'median', 'std', 'min', 'max').
    exclude_self : bool
        If True, exclude self-loops (Source == Target).
    top_edges_ratio : float
        Fraction of top edges (best accuracy targets) to include per subject (0.0 to 1.0).
    sort_ascending : bool
        If True, sort ascending (worst generalizing first). Default False (best first).

    Returns
    -------
    pd.DataFrame
        DataFrame ranked by generalization performance.
    """
    files = find_graphml_files(input_path, dataset=dataset, pipeline=pipeline, mode=mode)
    if not files:
        raise ValueError(f"No matching .graphml files found in '{input_path}' with dataset='{dataset}', pipeline='{pipeline}', mode='{mode}'.")

    all_stats = []
    for f in files:
        df_single = extract_graph_subject_stats(f, exclude_self=exclude_self, top_edges_ratio=top_edges_ratio)
        all_stats.append(df_single)

    combined_df = pd.concat(all_stats, ignore_index=True)

    # Aggregate across evaluated graphs for each subject
    aggregated = combined_df.groupby('Subject').agg(
        Mean_Accuracy=('Mean_Accuracy', 'mean'),
        Median_Accuracy=('Median_Accuracy', 'mean'),
        Std_Accuracy=('Std_Accuracy', 'mean'),
        Min_Accuracy=('Min_Accuracy', 'mean'),
        Max_Accuracy=('Max_Accuracy', 'mean'),
        Evaluated_Graphs_Count=('Graph_Name', 'count'),
        Avg_Selected_Edges=('Selected_Edges_Count', 'mean'),
        Avg_Total_Targets=('Total_Target_Count', 'mean')
    ).reset_index()

    metric_col_map = {
        'mean': 'Mean_Accuracy',
        'median': 'Median_Accuracy',
        'std': 'Std_Accuracy',
        'min': 'Min_Accuracy',
        'max': 'Max_Accuracy'
    }

    sort_col = metric_col_map.get(metric.lower(), 'Mean_Accuracy')
    
    # Custom numeric sorting for Subject IDs if numeric
    def try_int(val):
        try:
            return int(val)
        except ValueError:
            return val

    aggregated['Subject_Num'] = aggregated['Subject'].apply(try_int)
    
    # Primary sort by selected metric, secondary sort by numeric subject ID
    aggregated = aggregated.sort_values(
        by=[sort_col, 'Subject_Num'],
        ascending=[sort_ascending, True]
    ).reset_index(drop=True)

    aggregated.drop(columns=['Subject_Num'], inplace=True)
    aggregated.index += 1
    aggregated.index.name = 'Rank'

    return aggregated


def main():
    parser = argparse.ArgumentParser(
        description="Rank subjects in order of classifier generalization capability based on directed performance graphs."
    )
    parser.add_argument(
        "--input", "-i",
        default="graph_results/graphs",
        help="Path to a .graphml file or directory containing graph files. Default: graph_results/graphs"
    )
    parser.add_argument(
        "--dataset", "-d",
        default="PhysionetMI",
        help="Dataset directory to analyze ('PhysionetMI', 'Dreyer2023', or 'all'). Default: PhysionetMI"
    )
    parser.add_argument(
        "--pipeline", "-p",
        default=None,
        help="Filter graph files by classifier pipeline (e.g., 'Cov_Tangent', 'CSP_LDA', 'CSP_SVM')."
    )
    parser.add_argument(
        "--mode", "-m",
        default=None,
        help="Filter graph files by mode ('static' or 'adaptive')."
    )
    parser.add_argument(
        "--metric",
        default="mean",
        choices=["mean", "median", "std", "min", "max"],
        help="Metric to rank subjects by. Default: mean"
    )
    parser.add_argument(
        "--top-edges-percent", "--top-edges-ratio", "-e",
        type=float,
        default=1.0,
        help="Fraction or percentage of best outgoing edges (highest accuracies) to select for each subject (e.g., 0.2 or 20 for top 20%%). Default: 1.0 (all edges)."
    )
    parser.add_argument(
        "--include-self",
        action="store_true",
        help="Include self-loops (Source == Target) in generalization accuracy calculations."
    )
    parser.add_argument(
        "--ascending",
        action="store_true",
        help="Sort in ascending order (worst generalizing subject first)."
    )
    parser.add_argument(
        "--top-n", "-n",
        type=int,
        default=None,
        help="Limit printed output to top N subjects."
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Save ranked results table to a CSV file."
    )
    parser.add_argument(
        "--output-md",
        default=None,
        help="Save ranked results table to a Markdown file."
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Save ordered list of subjects and detailed statistics to a JSON file."
    )
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Output ONLY the ordered list of subject IDs (convenient for piping or simple copy-paste)."
    )

    args = parser.parse_args()

    exclude_self = not args.include_self

    # Normalize top_edges_percent / top_edges_ratio
    top_edges_ratio = args.top_edges_percent
    if top_edges_ratio > 1.0:
        top_edges_ratio /= 100.0

    if not (0.0 < top_edges_ratio <= 1.0):
        raise ValueError(f"top-edges-percent must be between 0 and 1 (or 0 and 100). Got: {args.top_edges_percent}")

    ranked_df = rank_subjects_by_generalization(
        input_path=args.input,
        dataset=args.dataset,
        pipeline=args.pipeline,
        mode=args.mode,
        metric=args.metric,
        exclude_self=exclude_self,
        top_edges_ratio=top_edges_ratio,
        sort_ascending=args.ascending
    )

    ordered_subject_list = ranked_df['Subject'].tolist()

    if args.simple:
        print(ordered_subject_list)
        return

    print("================================================================================")
    print(f" Subject Generalization Ranking")
    print("================================================================================")
    print(f" Input Path          : {args.input}")
    print(f" Dataset             : {args.dataset}")
    print(f" Pipeline Filter     : {args.pipeline or 'All'}")
    print(f" Mode Filter         : {args.mode or 'All'}")
    print(f" Ranking Metric      : {args.metric.upper()}")
    print(f" Top Edges Filter    : Top {top_edges_ratio * 100:.1f}% best edges per subject")
    print(f" Exclude Self-loops  : {exclude_self}")
    print(f" Total Subjects      : {len(ranked_df)}")
    print("================================================================================\n")

    display_df = ranked_df.head(args.top_n) if args.top_n else ranked_df
    
    # Format floating point numbers for clean terminal display
    pd.set_option('display.max_rows', 150)
    pd.set_option('display.float_format', lambda x: f'{x:.4f}')
    print(display_df[['Subject', 'Mean_Accuracy', 'Median_Accuracy', 'Std_Accuracy', 'Min_Accuracy', 'Max_Accuracy', 'Avg_Selected_Edges', 'Avg_Total_Targets']])

    print("\n--------------------------------------------------------------------------------")
    print("Ordered Subject List (Best to Worst Generalizer):")
    print(ordered_subject_list)
    print("--------------------------------------------------------------------------------")

    if args.output_csv:
        ranked_df.to_csv(args.output_csv)
        print(f"\nSaved CSV report to: {args.output_csv}")

    if args.output_md:
        md_content = [
            f"# Subject Generalization Ranking Report",
            f"",
            f"- **Dataset**: `{args.dataset}`",
            f"- **Pipeline Filter**: `{args.pipeline or 'All'}`",
            f"- **Mode Filter**: `{args.mode or 'All'}`",
            f"- **Ranking Metric**: `{args.metric.upper()}`",
            f"- **Top Edges Filter**: Top `{top_edges_ratio * 100:.1f}%` best edges per subject",
            f"- **Exclude Self-loops**: `{exclude_self}`",
            f"",
            f"## Subject Ranking Table",
            f"",
            display_df[['Subject', 'Mean_Accuracy', 'Median_Accuracy', 'Std_Accuracy', 'Min_Accuracy', 'Max_Accuracy', 'Avg_Selected_Edges', 'Avg_Total_Targets']].to_markdown(),
            f"",
            f"## Ordered Subject List",
            f"",
            f"```json",
            json.dumps(ordered_subject_list),
            f"```"
        ]
        with open(args.output_md, 'w', encoding='utf-8') as f:
            f.write("\n".join(md_content))
        print(f"\nSaved Markdown report to: {args.output_md}")

    if args.output_json:
        data_dict = {
            'metadata': {
                'dataset': args.dataset,
                'pipeline': args.pipeline,
                'mode': args.mode,
                'metric': args.metric,
                'top_edges_ratio': top_edges_ratio,
                'exclude_self': exclude_self
            },
            'ordered_subjects': ordered_subject_list,
            'ranking_details': ranked_df.reset_index().to_dict(orient='records')
        }
        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump(data_dict, f, indent=2)
        print(f"\nSaved JSON export to: {args.output_json}")


if __name__ == "__main__":
    main()
