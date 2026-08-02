# Master Documentation Index & Repository Map

Welcome to the **CSP Classifier & GNN Transfer Learning Repository Documentation Index**. This index organizes all theoretical guides, architectural specifications, CLI script manuals, and experimental report outputs.

---

## 🗺️ Documentation Quick Map

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DOCUMENTATION INDEX HUB                            │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
         ┌─────────────────────────────┼─────────────────────────────┐
         ▼                             ▼                             ▼
 📖 Theoretical Guides        🛠️ Tooling & Scripts         📊 Experiment Reports
 (graph_results/graphs/)      (graph_tools/README.md)       (graph_results/*/)
 - Max-Coverage Guide         - 8 Command-Line Tools        - Wearable Reports
 - GNN Clustering Guide       - CLI Arguments & Usage       - Crossover Reports
 - Wearable Selection Guide   - Graph Construction          - Cluster Reports
 - Master Takeaways           - GNN Clustering
```

---

## 1. 📖 Theoretical Guides & Architecture Specifications

| Document | Description | Key Focus |
| :--- | :--- | :--- |
| **[gnn_clustering_master_takeaways.md](file:///c:/Users/stz/Documents/GitHub/csp_classifier/graph_results/gnn_clustering_master_takeaways.md)** | **Master Summary** of GNN subject clustering across all 6 BCI datasets. | Dual-GNN math, donor selection gain, floor/ceiling effects. |
| **[max_coverage_wearable_guide.md](file:///c:/Users/stz/Documents/GitHub/csp_classifier/graph_results/graphs/max_coverage_wearable_guide.md)** | **Theoretical & Math Guide** for Submodular Max-Coverage Ensemble Selection. | Max-Coverage math, submodular $1-1/e$ proof, 2-phase wearable workflow. |
| **[wearable_5_classifier_selection_guide.md](file:///c:/Users/stz/Documents/GitHub/csp_classifier/graph_results/graphs/wearable_5_classifier_selection_guide.md)** | **Wearable Selection Guide** for deploying 5 pre-trained BCI classifiers. | Zero-Overlap Leave-N-Out ($v \notin \mathcal{S}^*$), GNN latent facility location. |
| **[gnn_subject_clustering_guide.md](file:///c:/Users/stz/Documents/GitHub/csp_classifier/graph_results/graphs/gnn_subject_clustering_guide.md)** | **GNN & Clustering Architecture** technical guide. | PyTorch GNN encoder, asymmetric adjacency matrix, DEC loss ($\text{KL}(P \parallel Q)$). |
| **[temporal_trial_binned_graphs.md](file:///c:/Users/stz/Documents/GitHub/csp_classifier/graph_results/graphs/temporal_trial_binned_graphs.md)** | **Temporal Trial-Binned Graph** architecture specification. | 10-bin trial window segmentation, vector-attributed temporal GraphML files. |
| **[data_mining_roadmap.md](file:///c:/Users/stz/Documents/GitHub/csp_classifier/graph_results/graphs/data_mining_roadmap.md)** | **Data Mining Roadmap** for transfer networks. | Graph topology analysis, centrality metrics, and motif mining. |

---

## 2. 🛠️ Tooling & Script Manuals

### A. Graph & Network Analysis Tools (**[graph_tools/README.md](file:///c:/Users/stz/Documents/GitHub/csp_classifier/graph_tools/README.md)**)
Provides CLI usage instructions, arguments, and examples for 8 graph analysis command-line scripts:
1. **`build_graphs.py`**: Converts raw simulation CSVs into GraphML directed performance networks.
2. **`build_binned_graphs.py`**: Converts trial-level CSVs into 10-bin temporal GraphML networks.
3. **`cluster_subjects.py`**: Runs PyTorch GNN embeddings + Deep Embedded Clustering (DEC).
4. **`interpret_clusters.py`**: Generates donor selection rules and proficiency tier breakdowns.
5. **`rank_subjects_by_generalization.py`**: Computes PageRank and hub centrality across donor networks.
6. **`subject_taxonomy.py`**: Classifies subjects into High-Responders, Adapters, and BCI Illiterates.
7. **`select_wearable_classifiers.py`**: Selects 5 optimal wearable classifiers via Submodular Greedy & GNN Facility Location.
8. **`baseline_crossover_analysis.py`**: Analyzes at which 10-trial bin zero-knowledge baselines catch up to transfer models.

### B. PyTorch Deep Learning Tools (**[eegnet_tools/README.md](file:///c:/Users/stz/Documents/GitHub/csp_classifier/eegnet_tools/README.md)**)
Provides complete architecture specifications, 9-model baseline matrix, and CLI instructions for EEGNet deep learning tools:
9. **`eegnet_tools/eegnet_model.py`**: PyTorch EEGNet-8,2 architecture module.
10. **`eegnet_tools/train_eegnet_models.py`**: Pre-training engine for multi-subject pooled PyTorch EEGNets & classical pipelines.
11. **`eegnet_tools/simulate_eegnet_adaptive.py`**: Online adaptive simulation with full EEGNet model fine-tuning ($\text{lr}=10^{-4}$).
12. **`eegnet_tools/evaluate_eegnet_benchmarks.py`**: Aggregates metrics and generates master benchmark report (`eegnet_master_benchmark_report.md`).

---

## 3. 📊 Generated Experiment Reports (By Dataset)

Each dataset folder in `graph_results/` contains automatically generated experiment reports:

### A. Deep Learning EEGNet Master Benchmark Reports (`graph_results/eegnet_benchmark/{dataset}/`)
- **`eegnet_master_benchmark_report.md`**: Master report benchmarking PyTorch EEGNets (Strategies A, B, C) against single-subject and pooled classical models (Baselines 1–6).
- Artifacts: `eegnet_vs_classical_trajectories.png`, `eegnet_benchmark_summary.csv`.

### B. Wearable Selection Reports (`graph_results/wearable_selection/{dataset}/`)
- **`wearable_selection_report.md`**: Direct comparison of 5-classifier wearable selection (Methods A, B, C, D) against Target Scratch Baseline Models.
- Artifacts: `{pipeline}_wearable_coverage_curve.png`, `{pipeline}_target_coverage_map.png`, `{pipeline}_wearable_vs_baseline_trials.png`.

### C. Baseline Crossover Analysis Reports (`graph_results/crossover_analysis/{dataset}/`)
- **`baseline_crossover_report.md`**: Bin-by-bin trajectory analysis determining when zero-knowledge baseline models overtake pre-trained transfer models.
- Artifacts: `{pipeline}_crossover_trajectories.png`, `{pipeline}_crossover_histogram.png`.

### C. GNN Cluster Interpretation Reports (`graph_results/clustering/{dataset}/`)
- **`cluster_interpretation_report.md`**: Cluster proficiency tier breakdowns, donor selection recommendation rules, and baseline accuracy gains.

---

## 💡 Recommended Reading Order

1. **New to the Project?** Start with **[gnn_clustering_master_takeaways.md](file:///c:/Users/stz/Documents/GitHub/csp_classifier/graph_results/gnn_clustering_master_takeaways.md)** for high-level findings.
2. **Interested in Wearable BCI Selection?** Read **[max_coverage_wearable_guide.md](file:///c:/Users/stz/Documents/GitHub/csp_classifier/graph_results/graphs/max_coverage_wearable_guide.md)**.
3. **Want to Run Commands?** Consult **[graph_tools/README.md](file:///c:/Users/stz/Documents/GitHub/csp_classifier/graph_tools/README.md)**.
