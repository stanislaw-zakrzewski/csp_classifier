# Graph Tools Directory (`graph_tools/`)

This directory houses all scripts and modules for building, processing, and analyzing cross-subject performance graphs created from EEG classification simulations.

---

## Tools Included

### 1. `build_graphs.py`
Constructs **standard directed performance graphs** from raw simulation CSV files.

**Key Features**:
- **Graph Construction**: Builds GraphML files for each pipeline configuration (3 pipelines $\times$ 2 modes: static and adaptive).
- **Node Baseline Attributes**: Attaches target subject within-subject baseline accuracies directly to nodes (`baseline_accuracy`, `baseline_classifier`).
- **Visualizations & Summaries**: Generates side-by-side accuracy heatmap PNG plots and summary CSV tables.

**Usage**:
```bash
# Build standard performance graphs for Dreyer2023
python graph_tools/build_graphs.py --dataset Dreyer2023

# Build graphs for PhysionetMI
python graph_tools/build_graphs.py --dataset PhysionetMI

# Custom input and output directories
python graph_tools/build_graphs.py --data-dir simulation_results/GutmannFlury2025_MI --output-dir graph_results/graphs/GutmannFlury2025_MI
```

---

### 2. `build_binned_graphs.py`
Constructs **temporal trial-binned performance graphs** from simulation results by splitting trial runs into $N$ equal windows (default: 10 bins).

**Key Features**:
- **Vector-Attributed Graphs**: Computes local interval accuracy per bin, cumulative accuracy at each checkpoint, zero-shot accuracy (Bin 1), and adaptation velocity slope on **edges**, and matching within-subject **baseline performance trajectory attributes on nodes** (`baseline_weight`, `baseline_bin_interval_acc_1..10`, `baseline_adaptation_slope`, etc.).
- **Node Attributes**: Stores within-subject baseline classifier metrics directly inside each node $v$, allowing self-contained relative gain ($\text{Acc}_{uv} - \text{Baseline}_v$) and breakeven crossover analysis.
- **Snapshot Graphs**: Saves 10 individual GraphML snapshots representing graph state evolution over time.
- **Visualizations**: Generates heatmaps comparing Zero-Shot (Bin 1), Mid-term (Bin 5), Final (Bin 10), and Adaptation Velocity Slope.
- **Summary Tables**: Exports full dataset performance summary CSVs.

**Usage**:
```bash
# Build 10-bin temporal graphs for Dreyer2023
python graph_tools/build_binned_graphs.py --dataset Dreyer2023

# Build 10-bin temporal graphs for PhysionetMI
python graph_tools/build_binned_graphs.py --dataset PhysionetMI

# Custom number of bins
python graph_tools/build_binned_graphs.py --dataset Dreyer2023 --num-bins 10
```

---

### 2. `rank_subjects_by_generalization.py`
Ranks subjects in descending order of generalization performance based on outgoing edge accuracies.

**Usage**:
```bash
# Rank subjects for Dreyer2023
python graph_tools/rank_subjects_by_generalization.py --dataset Dreyer2023

# Rank considering top 20% best edges per subject
python graph_tools/rank_subjects_by_generalization.py --dataset Dreyer2023 --top-edges-percent 0.2

# Output simple ordered subject list
python graph_tools/rank_subjects_by_generalization.py --dataset Dreyer2023 --simple
```

---

### 3. `subject_taxonomy.py`
Categorizes subjects into a **4-Quadrant Subject Taxonomy** based on static classifier models (*_static.graphml):
- **Quadrant I: Universal Donors** (High Outgoing, High Incoming Accuracy)
- **Quadrant II: Universal Recipients** (Low Outgoing, High Incoming Accuracy)
- **Quadrant III: Isolated / Refractory** (Low Outgoing, Low Incoming Accuracy)
- **Quadrant IV: Specialist Donors** (High Outgoing, Low Incoming Accuracy)

**Usage**:
```bash
# Run 4-Quadrant Subject Taxonomy analysis for Dreyer2023 static models
python graph_tools/subject_taxonomy.py --dataset Dreyer2023

# Save detailed report to Markdown and CSV
python graph_tools/subject_taxonomy.py --dataset Dreyer2023 --output-md dreyer_taxonomy.md --output-csv dreyer_taxonomy.csv
```

---

### 4. `filter_above_baseline.py`
Filters performance graphs by removing all cross-subject transfer edges ($u \to v$) where the classifier accuracy is lower than Target Subject $v$'s within-subject baseline model accuracy stored on Node $v$.

**Outputs**:
- Filtered GraphML files (`*_above_baseline.graphml`).
- 3-Panel Heatmaps comparing Unfiltered Accuracies vs. Filtered Accuracies vs. Relative Gain ($\text{Acc}_{uv} - \text{Baseline}_v$).
- Directed Network Diagrams showing surviving super-generalizing connections.
- Retention rate summary tables.

**Usage**:
```bash
# Filter all graphs for Dreyer2023 and generate plots
python graph_tools/filter_above_baseline.py --dataset Dreyer2023

# Filter specific GraphML file
python graph_tools/filter_above_baseline.py --input graph_results/graphs/Dreyer2023/Cov_Tangent_Space_LR_pipeline_static.graphml

# Filter with an additional accuracy margin (e.g. baseline + 2%)
python graph_tools/filter_above_baseline.py --dataset Dreyer2023 --margin 0.02
```

---

### 5. `cluster_subjects.py`
Applies **PyTorch Graph Neural Networks (GNN Autoencoders)** and **Deep Embedded Clustering (DEC)** to discover clusters of compatible subjects.

**Key Features**:
- **Dual Asymmetric Embeddings**: Learns $Z_{\text{donor}}$ (outgoing generalization capability) and $Z_{\text{recipient}}$ (target receptivity) to handle directed transfer asymmetry ($u \to v$ vs $v \to u$).
- **Standard & 10-Bin Temporal Support**: Supports both standard graphs and 10-bin temporal trial graphs (`--binned`).
- **Visualizations**: Generates reordered block-diagonal heatmaps, 2D t-SNE latent embedding plots, and 10-bin cluster adaptation curves (`*_cluster_adaptation_curves.png`).

**Usage**:
```bash
# GNN Clustering on Standard Graphs
python graph_tools/cluster_subjects.py --dataset Dreyer2023 --num-clusters 5

# GNN Clustering on 10-Bin Temporal Binned Graphs
python graph_tools/cluster_subjects.py --dataset Dreyer2023 --binned --num-clusters 5
```

---

### 6. `interpret_clusters.py`
Analyzes GNN clustering outputs and generates actionable donor model selection recommendation rules and cluster proficiency tier breakdowns.

**Usage**:
```bash
# Interpret standard clustering results
python graph_tools/interpret_clusters.py --dataset Dreyer2023

# Interpret 10-bin temporal binned clustering results
python graph_tools/interpret_clusters.py --dataset Dreyer2023 --binned
```

---

### 7. `select_wearable_classifiers.py`
Selects an optimal subset of $K=5$ pre-trained classifiers to deploy on a wearable BCI device using Submodular Max-Coverage and GNN Latent Space Facility Location under a strict Zero-Overlap protocol ($v \notin \mathcal{S}^*$).

**Usage**:
```bash
python graph_tools/select_wearable_classifiers.py --dataset Dreyer2023 --num-classifiers 5
```

---

### 8. `baseline_crossover_analysis.py`
Determines the exact 10-trial bin index at which a zero-knowledge baseline classifier (adapting online from scratch) overtakes pre-trained adaptive cross-subject classifiers (both Average of All Donors and Average of Top 10% Donors).

**Usage**:
```bash
python graph_tools/baseline_crossover_analysis.py --dataset Dreyer2023
python graph_tools/baseline_crossover_analysis.py --dataset PhysionetMI
```

---

## Documentation & Comprehensive Findings

For full mathematical formulations, cross-dataset benchmarks, and clinical deployment guidelines, refer to:
- **[gnn_clustering_master_takeaways.md](file:///c:/Users/stz/Documents/GitHub/csp_classifier/graph_results/gnn_clustering_master_takeaways.md)**: Master takeaways summary comparing all 6 datasets.
- **[max_coverage_wearable_guide.md](file:///c:/Users/stz/Documents/GitHub/csp_classifier/graph_results/graphs/max_coverage_wearable_guide.md)**: Max-Coverage math formulation, deployment workflow, and generalization guide.
- **[wearable_5_classifier_selection_guide.md](file:///c:/Users/stz/Documents/GitHub/csp_classifier/graph_results/graphs/wearable_5_classifier_selection_guide.md)**: Guide for 5-classifier wearable ensemble selection.
- **[gnn_subject_clustering_guide.md](file:///c:/Users/stz/Documents/GitHub/csp_classifier/graph_results/graphs/gnn_subject_clustering_guide.md)**: PyTorch GNN architecture and DEC clustering guide.






