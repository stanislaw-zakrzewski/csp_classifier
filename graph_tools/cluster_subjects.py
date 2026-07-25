"""
Neural Network Subject Compatibility Clustering for Cross-Subject EEG Graphs
=============================================================================

This script applies Graph Neural Networks (GNN Autoencoders) and Deep Embedded Clustering (DEC)
to directed cross-subject performance graphs to discover clusters of "compatible" subjects.

Supports both Standard Performance Graphs AND Vector-Attributed 10-Bin Temporal Graphs (--binned).

Modes Supported:
----------------
1. 'functional' (Default): Clusters subjects into Donor Clusters (subjects whose models generalize to the same targets)
   and Recipient Clusters (subjects that benefit from the same donor models).
2. 'mutual': Filters for bidirectional compatibility (where W_uv >= Baseline_v AND W_vu >= Baseline_u)
   and clusters subjects into mutually exchangeable sub-cohorts.

Outputs:
--------
Saved to `graph_results/clustering/{dataset}/` (or `graph_results/clustering_binned/{dataset}/` when --binned is set):
- Reordered Clustered Heatmaps showing block-diagonal compatibility clusters (*_clustered_heatmap.png).
- 2D Latent Embedding Scatter Plots (*_donor_recipient_embeddings.png).
- Cluster Adaptation Curve Line Plots (*_cluster_adaptation_curves.png, for --binned).
- Subject Cluster Assignments CSV (subject_cluster_assignments.csv).
- Performance & Baseline Gain Metrics Summary CSV (clustering_summary.csv).
- Master Markdown Analysis Report (clustering_report.md / binned_clustering_report.md).

Usage Examples:
---------------
1. Functional GNN Clustering on Standard Graphs:
    python graph_tools/cluster_subjects.py --dataset Dreyer2023 --num-clusters 5

2. Functional GNN Clustering on 10-Bin Temporal Binned Graphs:
    python graph_tools/cluster_subjects.py --dataset Dreyer2023 --binned --num-clusters 5

3. Mutual Compatibility Clustering on PhysionetMI:
    python graph_tools/cluster_subjects.py --dataset PhysionetMI --mode mutual --num-clusters 5
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

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from scipy.stats import ttest_ind

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)


def find_graph_files(
    input_path: str,
    dataset: str = "Dreyer2023",
    pipeline: str = None,
    is_binned: bool = False
) -> list[str]:
    """Find matching GraphML files for standard or binned graphs."""
    if os.path.isfile(input_path):
        if input_path.endswith('.graphml'):
            return [os.path.abspath(input_path)]
        else:
            raise ValueError(f"Specified file '{input_path}' is not a .graphml file.")

    search_dir = input_path
    if dataset and dataset.lower() != 'all':
        subfolder = "graphs_binned" if is_binned else "graphs"
        candidates = [
            os.path.join(input_path, subfolder, dataset),
            os.path.join(input_path, dataset),
            os.path.join("graph_results", subfolder, dataset),
            os.path.join("graph_results", dataset),
            os.path.join("simulation_results", subfolder, dataset),
            os.path.join("simulation_results", dataset),
        ]
        found_dir = None
        for cand in candidates:
            if os.path.exists(cand) and glob.glob(os.path.join(cand, "*.graphml")):
                found_dir = cand
                break

        if not found_dir:
            raise FileNotFoundError(
                f"No graph directory containing .graphml files was found for dataset '{dataset}' (is_binned={is_binned}).\n"
                f"Please ensure graph files exist in 'graph_results/{subfolder}/{dataset}'."
            )
        search_dir = found_dir

    if not os.path.exists(search_dir):
        raise FileNotFoundError(f"Search directory '{search_dir}' does not exist.")

    files = glob.glob(os.path.join(search_dir, "**", "*.graphml"), recursive=True)

    filtered_files = []
    for f in files:
        basename = os.path.basename(f)
        if "_above_baseline" in basename:
            continue
        if is_binned:
            if "binned" not in basename.lower():
                continue
        else:
            if "binned" in basename.lower() or "_bin_" in basename:
                continue

        if pipeline and pipeline.lower() not in basename.lower():
            continue
        filtered_files.append(os.path.abspath(f))

    return sorted(filtered_files)


class AsymmetricGNNEncoder(nn.Module):
    """
    Dual-Embedding GNN Encoder that maps subject transfer profiles into two distinct latent spaces:
    1. Z_donor: Encodes model donor generalization strength.
    2. Z_recipient: Encodes target model receptivity.
    """
    def __init__(self, in_features: int, hidden_dim: int = 64, latent_dim: int = 16):
        super().__init__()
        # Donor Encoder Pipeline
        self.donor_fc1 = nn.Linear(in_features, hidden_dim)
        self.donor_fc2 = nn.Linear(hidden_dim, latent_dim)

        # Recipient Encoder Pipeline
        self.recip_fc1 = nn.Linear(in_features, hidden_dim)
        self.recip_fc2 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, A_norm: torch.Tensor, X_out: torch.Tensor, X_in: torch.Tensor):
        h_don = F.relu(self.donor_fc1(torch.mm(A_norm, X_out)))
        z_don = self.donor_fc2(h_don)

        h_rec = F.relu(self.recip_fc1(torch.mm(A_norm.t(), X_in)))
        z_rec = self.recip_fc2(h_rec)

        return z_don, z_rec


class DirectedDecoder(nn.Module):
    """Decodes predicted cross-subject transfer matrix (single value or 10-bin vector) from embeddings."""
    def __init__(self, latent_dim: int = 16, out_dim: int = 1):
        super().__init__()
        self.out_dim = out_dim
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim * 2, 32),
            nn.ReLU(),
            nn.Linear(32, out_dim),
            nn.Sigmoid()
        )

    def forward(self, z_don: torch.Tensor, z_rec: torch.Tensor):
        N = z_don.size(0)
        z_don_exp = z_don.unsqueeze(1).expand(N, N, -1)
        z_rec_exp = z_rec.unsqueeze(0).expand(N, N, -1)

        cat_feat = torch.cat([z_don_exp, z_rec_exp], dim=-1)
        pred = self.mlp(cat_feat)
        if self.out_dim == 1:
            pred = pred.squeeze(-1)
        return pred


class DeepGraphClusteringModel(nn.Module):
    """Full PyTorch GNN Autoencoder and Clustering Neural Network."""
    def __init__(self, in_features: int, hidden_dim: int = 64, latent_dim: int = 16, num_clusters: int = 5, out_dim: int = 1):
        super().__init__()
        self.num_clusters = num_clusters
        self.encoder = AsymmetricGNNEncoder(in_features=in_features, hidden_dim=hidden_dim, latent_dim=latent_dim)
        self.decoder = DirectedDecoder(latent_dim=latent_dim, out_dim=out_dim)

        self.donor_centroids = nn.Parameter(torch.Tensor(num_clusters, latent_dim))
        self.recip_centroids = nn.Parameter(torch.Tensor(num_clusters, latent_dim))

        nn.init.xavier_uniform_(self.donor_centroids)
        nn.init.xavier_uniform_(self.recip_centroids)

    def forward(self, A_norm: torch.Tensor, X_out: torch.Tensor, X_in: torch.Tensor):
        z_don, z_rec = self.encoder(A_norm, X_out, X_in)
        pred = self.decoder(z_don, z_rec)
        return z_don, z_rec, pred


def train_gnn_clustering(
    G: nx.DiGraph,
    subjects: list[str],
    num_clusters: int = 5,
    epochs: int = 250,
    lr: float = 0.01,
    mode: str = "functional",
    is_binned: bool = False
) -> tuple[dict, pd.DataFrame]:
    """Train PyTorch GNN Autoencoder and extract subject cluster assignments for standard or binned graphs."""
    N = len(subjects)
    num_bins = 10 if is_binned else 1

    # Build Adjacency Matrix and Feature Tensors
    W_orig = nx.to_pandas_adjacency(G, nodelist=subjects, weight='weight').values
    baselines = np.array([
        float(G.nodes[s].get('baseline_accuracy', G.nodes[s].get('baseline_weight', G.nodes[s].get('baseline_final_interval_acc', 0.0)))) for s in subjects
    ])

    if is_binned:
        # Build 3D Tensor of 10-bin interval accuracies (N x N x 10)
        W_binned_3d = np.zeros((N, N, 10))
        for i, u in enumerate(subjects):
            for j, v in enumerate(subjects):
                if G.has_edge(u, v):
                    edge_data = G[u][v]
                    for b in range(10):
                        W_binned_3d[i, j, b] = float(edge_data.get(f'bin_interval_acc_{b+1}', edge_data.get('weight', 0.0)))

        # Node baseline 10-bin vectors (N x 10)
        baseline_bins = np.zeros((N, 10))
        baseline_slopes = np.zeros((N, 1))
        for i, s in enumerate(subjects):
            node_data = G.nodes[s]
            for b in range(10):
                baseline_bins[i, b] = float(node_data.get(f'baseline_bin_interval_acc_{b+1}', baselines[i]))
            baseline_slopes[i, 0] = float(node_data.get('baseline_adaptation_slope', 0.0))

        # Flattened outgoing (N x 10N) and incoming features
        X_out_np = np.hstack([W_binned_3d.reshape(N, -1), baseline_bins, baseline_slopes])
        X_in_np = np.hstack([W_binned_3d.transpose(1, 0, 2).reshape(N, -1), baseline_bins, baseline_slopes])
        W_target = torch.tensor(W_binned_3d, dtype=torch.float32)
        in_feat_dim = X_out_np.shape[1]
        out_dim = 10
    else:
        X_out_np = W_orig.copy()
        X_in_np = W_orig.T.copy()
        for i, b in enumerate(baselines):
            X_out_np[i, i] = b
            X_in_np[i, i] = b
        W_target = torch.tensor(W_orig, dtype=torch.float32)
        in_feat_dim = N
        out_dim = 1

    # Normalize Adjacency Matrix
    A_raw = W_orig.copy()
    np.fill_diagonal(A_raw, 0)
    deg_out = np.sum(A_raw, axis=1)
    deg_out[deg_out == 0] = 1.0
    D_inv = np.diag(1.0 / np.sqrt(deg_out))
    A_norm_np = D_inv @ A_raw @ D_inv

    A_norm = torch.tensor(A_norm_np, dtype=torch.float32)
    X_out = torch.tensor(X_out_np, dtype=torch.float32)
    X_in = torch.tensor(X_in_np, dtype=torch.float32)

    # Instantiate PyTorch Model
    model = DeepGraphClusteringModel(in_features=in_feat_dim, hidden_dim=64, latent_dim=16, num_clusters=num_clusters, out_dim=out_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # Train Loop
    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        z_don, z_rec, W_pred = model(A_norm, X_out, X_in)

        if is_binned:
            mask = (1.0 - torch.eye(N)).unsqueeze(-1).expand(-1, -1, 10)
            recon_loss = F.mse_loss(W_pred * mask, W_target * mask)
        else:
            mask = 1.0 - torch.eye(N)
            recon_loss = F.mse_loss(W_pred * mask, W_target * mask)

        loss = recon_loss
        loss.backward()
        optimizer.step()

    # Extract Trained Embeddings
    model.eval()
    with torch.no_grad():
        z_don, z_rec, W_pred = model(A_norm, X_out, X_in)
        z_don_np = z_don.cpu().numpy()
        z_rec_np = z_rec.cpu().numpy()

    # Clustering
    if mode.lower() == "mutual":
        z_joint = np.hstack([z_don_np, z_rec_np])
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        mutual_clusters = kmeans.fit_predict(z_joint)
        donor_clusters = mutual_clusters
        recip_clusters = mutual_clusters
    else:
        kmeans_don = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        donor_clusters = kmeans_don.fit_predict(z_don_np)

        kmeans_rec = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        recip_clusters = kmeans_rec.fit_predict(z_rec_np)

    cluster_df = pd.DataFrame({
        'Subject': subjects,
        'Baseline_Accuracy': baselines,
        'Donor_Cluster': donor_clusters,
        'Recipient_Cluster': recip_clusters
    })

    # t-SNE Embeddings
    tsne_don = TSNE(n_components=2, random_state=42, perplexity=min(30, max(5, N // 4)))
    z_don_2d = tsne_don.fit_transform(z_don_np)
    cluster_df['z_donor_x'] = z_don_2d[:, 0]
    cluster_df['z_donor_y'] = z_don_2d[:, 1]

    tsne_rec = TSNE(n_components=2, random_state=42, perplexity=min(30, max(5, N // 4)))
    z_rec_2d = tsne_rec.fit_transform(z_rec_np)
    cluster_df['z_recip_x'] = z_rec_2d[:, 0]
    cluster_df['z_recip_y'] = z_rec_2d[:, 1]

    # Evaluation Metrics
    within_cluster_accs = []
    between_cluster_accs = []
    within_cluster_gains = []
    within_crossovers = []
    overall_crossovers = []

    # Additional metrics for binned mode
    zero_shot_within = []
    zero_shot_between = []
    final_within = []
    final_between = []

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            w_ij = W_orig[i, j]
            b_j = baselines[j]
            gain_ij = w_ij - b_j
            is_crossover = (w_ij >= b_j)
            overall_crossovers.append(is_crossover)

            is_intra = (recip_clusters[i] == recip_clusters[j] or donor_clusters[i] == recip_clusters[j])

            if is_intra:
                within_cluster_accs.append(w_ij)
                within_cluster_gains.append(gain_ij)
                within_crossovers.append(is_crossover)
            else:
                between_cluster_accs.append(w_ij)

            if is_binned:
                w_zero = float(W_binned_3d[i, j, 0])
                w_final = float(W_binned_3d[i, j, 9])
                if is_intra:
                    zero_shot_within.append(w_zero)
                    final_within.append(w_final)
                else:
                    zero_shot_between.append(w_zero)
                    final_between.append(w_final)

    within_mean_acc = float(np.mean(within_cluster_accs)) if within_cluster_accs else 0.0
    between_mean_acc = float(np.mean(between_cluster_accs)) if between_cluster_accs else 0.0
    within_gain = float(np.mean(within_cluster_gains)) if within_cluster_gains else 0.0
    within_ratio = (within_mean_acc / between_mean_acc) if between_mean_acc > 0 else 1.0

    within_cross_rate = float(np.mean(within_crossovers)) if within_crossovers else 0.0
    overall_cross_rate = float(np.mean(overall_crossovers)) if overall_crossovers else 0.0
    crossover_lift = (within_cross_rate / overall_cross_rate) if overall_cross_rate > 0 else 1.0

    try:
        t_stat, p_val = ttest_ind(within_cluster_accs, between_cluster_accs, equal_var=False)
        p_val = float(p_val)
    except Exception:
        p_val = 1.0

    try:
        sil_don = float(silhouette_score(z_don_np, donor_clusters))
        sil_rec = float(silhouette_score(z_rec_np, recip_clusters))
    except Exception:
        sil_don = 0.0
        sil_rec = 0.0

    metrics = {
        'within_cluster_mean_acc': round(within_mean_acc, 4),
        'between_cluster_mean_acc': round(between_mean_acc, 4),
        'within_cluster_ratio': round(within_ratio, 4),
        'within_cluster_baseline_gain': round(within_gain, 4),
        'within_crossover_rate': round(within_cross_rate, 4),
        'overall_crossover_rate': round(overall_cross_rate, 4),
        'crossover_lift': round(crossover_lift, 4),
        'p_value': round(p_val, 6),
        'donor_silhouette_score': round(sil_don, 4),
        'recipient_silhouette_score': round(sil_rec, 4)
    }

    if is_binned:
        zs_w = float(np.mean(zero_shot_within)) if zero_shot_within else 0.0
        zs_b = float(np.mean(zero_shot_between)) if zero_shot_between else 0.0
        zs_ratio = (zs_w / zs_b) if zs_b > 0 else 1.0

        fin_w = float(np.mean(final_within)) if final_within else 0.0
        fin_b = float(np.mean(final_between)) if final_between else 0.0
        fin_ratio = (fin_w / fin_b) if fin_b > 0 else 1.0

        metrics['zero_shot_within_acc'] = round(zs_w, 4)
        metrics['zero_shot_between_acc'] = round(zs_b, 4)
        metrics['zero_shot_ratio'] = round(zs_ratio, 4)
        metrics['final_within_acc'] = round(fin_w, 4)
        metrics['final_between_acc'] = round(fin_b, 4)
        metrics['final_ratio'] = round(fin_ratio, 4)

    return metrics, cluster_df


def plot_clustered_heatmap(
    G: nx.DiGraph,
    subjects: list[str],
    cluster_df: pd.DataFrame,
    pipe_name: str,
    output_path: str
):
    """Generate a re-ordered transfer heatmap sorted by cluster assignment."""
    sorted_df = cluster_df.sort_values(by=['Recipient_Cluster', 'Donor_Cluster', 'Subject']).reset_index(drop=True)
    sorted_subjects = sorted_df['Subject'].tolist()

    W_orig = nx.to_pandas_adjacency(G, nodelist=sorted_subjects, weight='weight')

    for s in sorted_subjects:
        if s in G.nodes and 'baseline_accuracy' in G.nodes[s]:
            W_orig.loc[s, s] = float(G.nodes[s]['baseline_accuracy'])

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(W_orig, cmap="viridis", vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'Accuracy'})

    ax.set_title(f"Clustered Transfer Heatmap: {pipe_name}\n(Re-ordered by GNN Subject Compatibility Cluster)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Target Subject (Sorted by Cluster)", fontsize=11)
    ax.set_ylabel("Source Subject (Sorted by Cluster)", fontsize=11)

    tick_step = max(1, len(sorted_subjects) // 10)
    ax.set_xticks(ticks=range(0, len(sorted_subjects), tick_step))
    ax.set_xticklabels([sorted_subjects[i] for i in range(0, len(sorted_subjects), tick_step)])
    ax.set_yticks(ticks=range(0, len(sorted_subjects), tick_step))
    ax.set_yticklabels([sorted_subjects[i] for i in range(0, len(sorted_subjects), tick_step)])

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_latent_embeddings(
    cluster_df: pd.DataFrame,
    pipe_name: str,
    output_path: str
):
    """Generate 2D t-SNE scatter plots for Donor and Recipient latent GNN embeddings."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    sns.scatterplot(
        data=cluster_df,
        x='z_donor_x',
        y='z_donor_y',
        hue='Donor_Cluster',
        palette='tab10',
        s=100,
        edgecolor='black',
        linewidth=0.8,
        ax=axes[0]
    )
    axes[0].set_title(f"{pipe_name}\nDonor Latent Embeddings (Z_donor)", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("t-SNE Dimension 1")
    axes[0].set_ylabel("t-SNE Dimension 2")
    axes[0].grid(True, linestyle='--', alpha=0.4)

    for _, row in cluster_df.iterrows():
        axes[0].text(row['z_donor_x'] + 0.5, row['z_donor_y'] + 0.5, str(row['Subject']), fontsize=7.5, alpha=0.8)

    sns.scatterplot(
        data=cluster_df,
        x='z_recip_x',
        y='z_recip_y',
        hue='Recipient_Cluster',
        palette='tab10',
        s=100,
        edgecolor='black',
        linewidth=0.8,
        ax=axes[1]
    )
    axes[1].set_title(f"{pipe_name}\nRecipient Latent Embeddings (Z_recipient)", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("t-SNE Dimension 1")
    axes[1].set_ylabel("t-SNE Dimension 2")
    axes[1].grid(True, linestyle='--', alpha=0.4)

    for _, row in cluster_df.iterrows():
        axes[1].text(row['z_recip_x'] + 0.5, row['z_recip_y'] + 0.5, str(row['Subject']), fontsize=7.5, alpha=0.8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_cluster_adaptation_curves(
    G: nx.DiGraph,
    subjects: list[str],
    cluster_df: pd.DataFrame,
    pipe_name: str,
    output_path: str
):
    """Plot 10-bin trial adaptation curves for each Recipient Cluster vs baseline."""
    num_recip_k = cluster_df['Recipient_Cluster'].nunique()

    fig, ax = plt.subplots(figsize=(12, 7))
    bins = list(range(1, 11))

    for rec_m in range(num_recip_k):
        rec_subs = cluster_df[cluster_df['Recipient_Cluster'] == rec_m]['Subject'].astype(str).tolist()
        if not rec_subs:
            continue

        donor_subs = cluster_df[cluster_df['Donor_Cluster'] == rec_m]['Subject'].astype(str).tolist()
        if not donor_subs:
            donor_subs = subjects

        # Compute average 10-bin interval accuracy trajectory
        bin_accs = np.zeros(10)
        counts = 0
        for u in donor_subs:
            for v in rec_subs:
                if u != v and G.has_edge(u, v):
                    edge_data = G[u][v]
                    for b in range(10):
                        bin_accs[b] += float(edge_data.get(f'bin_interval_acc_{b+1}', edge_data.get('weight', 0.0)))
                    counts += 1
        if counts > 0:
            bin_accs /= counts
            ax.plot(bins, bin_accs, marker='o', linewidth=2, label=f'Recipient Cluster {rec_m} ({len(rec_subs)} subs)')

    # Compute overall baseline curve
    base_accs = np.zeros(10)
    for s in subjects:
        node_data = G.nodes[s]
        for b in range(10):
            base_accs[b] += float(node_data.get(f'baseline_bin_interval_acc_{b+1}', node_data.get('baseline_accuracy', 0.0)))
    base_accs /= len(subjects)

    ax.plot(bins, base_accs, color='black', linestyle='--', linewidth=2.5, label='Overall Target Baseline')

    ax.set_title(f"10-Bin Temporal Trial Adaptation Curves: {pipe_name}\n(Cluster Transfer Trajectories vs Baseline)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Trial Window Bin (1 to 10)", fontsize=11)
    ax.set_ylabel("Mean Interval Accuracy", fontsize=11)
    ax.set_xticks(bins)
    ax.set_ylim(0.4, 1.0)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='lower right', fontsize=10)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Cluster subjects on standard or 10-bin temporal performance graphs using PyTorch Graph Neural Networks (GNNs)."
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
        help="Filter specific pipeline configuration."
    )
    parser.add_argument(
        "--mode", "-m",
        default="functional",
        choices=["functional", "mutual"],
        help="Clustering mode: 'functional' or 'mutual'. Default: functional"
    )
    parser.add_argument(
        "--num-clusters", "-k",
        type=int,
        default=5,
        help="Number of subject clusters to form. Default: 5"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=250,
        help="GNN pre-training epochs. Default: 250"
    )
    parser.add_argument(
        "--binned",
        action="store_true",
        help="Use 10-bin vector-attributed temporal trial graphs from graphs_binned."
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help="Custom output directory for clustering artifacts."
    )

    args = parser.parse_args()

    graph_files = find_graph_files(
        input_path=args.input,
        dataset=args.dataset,
        pipeline=args.pipeline,
        is_binned=args.binned
    )

    if not graph_files:
        raise FileNotFoundError(f"No matching GraphML files found for dataset='{args.dataset}' (binned={args.binned}) in '{args.input}'.")

    sub_dir = "clustering_binned" if args.binned else "clustering"
    output_dir = args.output_dir or os.path.join("graph_results", sub_dir, args.dataset)
    os.makedirs(output_dir, exist_ok=True)

    print("================================================================================")
    print(f" PyTorch GNN Subject Compatibility Clustering (Binned={args.binned})")
    print("================================================================================")
    print(f" Dataset             : {args.dataset}")
    print(f" Clustering Mode     : {args.mode.upper()}")
    print(f" Target Clusters (K) : {args.num_clusters}")
    print(f" Matched Graph Files : {len(graph_files)}")
    for f in graph_files:
        print(f"   - {os.path.basename(f)}")
    print(f" Output Directory    : {output_dir}")
    print("================================================================================\n")

    all_metrics = []
    all_subject_assignments = []

    for g_file in graph_files:
        pipe_name = os.path.splitext(os.path.basename(g_file))[0].replace("_binned", "")
        G = nx.read_graphml(g_file)
        subjects = sorted(list(G.nodes()), key=lambda x: int(x) if x.isdigit() else x)

        print(f"Training GNN for pipeline '{pipe_name}' ({len(subjects)} subjects, binned={args.binned})...")
        metrics, cluster_df = train_gnn_clustering(
            G=G,
            subjects=subjects,
            num_clusters=args.num_clusters,
            epochs=args.epochs,
            mode=args.mode,
            is_binned=args.binned
        )

        cluster_df['Pipeline_Config'] = pipe_name
        all_subject_assignments.append(cluster_df)

        metrics['Pipeline_Graph'] = pipe_name
        metrics['Num_Clusters'] = args.num_clusters
        all_metrics.append(metrics)

        print(f"  - Within-Cluster Mean Acc   : {metrics['within_cluster_mean_acc']:.4f}")
        print(f"  - Between-Cluster Mean Acc  : {metrics['between_cluster_mean_acc']:.4f}")
        print(f"  - Within-Cluster Ratio      : {metrics['within_cluster_ratio']:.4f}x")
        print(f"  - Within-Cluster Baseline Gain: +{metrics['within_cluster_baseline_gain']:.4f}")
        if args.binned:
            print(f"  - Zero-Shot Ratio (Bin 1)   : {metrics['zero_shot_ratio']:.4f}x")
            print(f"  - Final Ratio (Bin 10)      : {metrics['final_ratio']:.4f}x")
        print(f"  - Donor Silhouette Score    : {metrics['donor_silhouette_score']:.4f}")
        print(f"  - Recipient Silhouette Score: {metrics['recipient_silhouette_score']:.4f}")

        # Save clustered heatmap
        heatmap_path = os.path.join(output_dir, f"{pipe_name}_clustered_heatmap.png")
        plot_clustered_heatmap(G, subjects, cluster_df, pipe_name, heatmap_path)

        # Save 2D Latent Embeddings scatter plot
        embed_plot_path = os.path.join(output_dir, f"{pipe_name}_donor_recipient_embeddings.png")
        plot_latent_embeddings(cluster_df, pipe_name, embed_plot_path)

        if args.binned:
            curve_plot_path = os.path.join(output_dir, f"{pipe_name}_cluster_adaptation_curves.png")
            plot_cluster_adaptation_curves(G, subjects, cluster_df, pipe_name, curve_plot_path)

        print()

    # Save summary tables
    summary_df = pd.DataFrame(all_metrics)
    summary_csv_path = os.path.join(output_dir, "clustering_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)

    combined_assignments = pd.concat(all_subject_assignments, ignore_index=True)
    assignments_csv_path = os.path.join(output_dir, "subject_cluster_assignments.csv")
    combined_assignments.to_csv(assignments_csv_path, index=False)

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

    # Save master markdown report
    report_filename = "binned_clustering_report.md" if args.binned else "clustering_report.md"
    md_content = [
        f"# PyTorch GNN Subject Compatibility Clustering Report ({args.dataset}, Binned={args.binned})",
        f"",
        f"- **Dataset**: `{args.dataset}`",
        f"- **Temporal 10-Bin Mode**: `{args.binned}`",
        f"- **Clustering Mode**: `{args.mode.upper()}`",
        f"- **Target Clusters (K)**: `{args.num_clusters}`",
        f"- **Analyzed Graphs**: `{len(graph_files)} pipeline configurations`",
        f"",
        f"## Clustering Performance Metrics Summary",
        f"",
        df_to_markdown(summary_df),
        f"",
        f"## Key Takeaways & Compatibility Insights",
        f"- **Within-Cluster Ratio**: Ratio of transfer accuracy between subjects inside the same cluster vs. subjects in different clusters.",
        f"- **Within-Cluster Baseline Gain**: Average accuracy improvement gained over target subject baseline training when selecting models from the assigned cluster.",
        f"",
        f"## Generated Artifacts",
        f"- All clustered heatmaps, t-SNE latent embedding plots, and CSV tables saved in: `{output_dir}`"
    ]

    report_md_path = os.path.join(output_dir, report_filename)
    with open(report_md_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(md_content))

    print(f"Saved clustering metrics summary CSV to: {summary_csv_path}")
    print(f"Saved subject cluster assignments CSV to: {assignments_csv_path}")
    print(f"Saved Markdown clustering report to: {report_md_path}\n")


if __name__ == "__main__":
    main()
