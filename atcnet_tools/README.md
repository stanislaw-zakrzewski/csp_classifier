# PyTorch ATCNet Transfer Learning Tools (`atcnet_tools`)

This package provides a comprehensive suite of PyTorch deep learning models, multi-subject pre-training engines, online adaptive simulation engines, and master benchmark report generators for **ATCNet (Attention-based Temporal Convolutional Network)** (Altaheri et al., 2023).

---

## 1. Model Architecture: ATCNet

ATCNet combines spatial-temporal convolutions, Multi-Head Self-Attention (MHSA), and Dilated Temporal Convolutional Networks (TCNs) into a wearable-optimized architecture (~115k parameters):

1. **Conv Block**: Temporal Conv2d $\to$ Depthwise Spatial Conv2d ($C \times 1$, max-norm $\le 1.0$) $\to$ Pointwise Conv2d $\to$ AvgPool2d $\to$ Dropout.
2. **Attention Module**: Sliding-window Multi-Head Self-Attention (MHSA) module operating on temporal feature maps.
3. **Dilated TCN Module**: 1D Causal Dilated Convolutions (dilation = 1, 2) with residual connections ($O(N)$ linear temporal complexity).
4. **Classification Head**: Dense Linear Layer (max-norm $\le 0.5$) $\to$ Log-Softmax.

---

## 2. Model Strategies & Pre-Computed Baseline Reuse

The framework benchmarks **ATCNet strategies** and compares them directly against pre-computed **EEGNet** and **Classical Baselines** without re-running unnecessary computations:

| Strategy | Architecture | Training Data Strategy | Description |
| :--- | :--- | :--- | :--- |
| **Strategy A** | **PyTorch ATCNet** | **5 Cluster-Mode Pooled Cohorts** | 5 specialized ATCNet models (1 per GNN cluster mode) |
| **Strategy B** | **PyTorch ATCNet** | **Top-5 Submodular Pooled Cohort** | 1 Submodular Foundation ATCNet model trained on Top-5 pooled data (`[2, 14, 43, 80, 82]`) |
| **Strategy C** | **PyTorch ATCNet** | Single Subject Datasets | Dedicated single-subject ATCNet models |
| **Baseline 1** | PyTorch ATCNet | Target Subject $v$ (Scratch) | Cold-start ATCNet trained from scratch ($\text{lr} = 10^{-3}$, 5 mini-epochs) |
| **Pre-Computed Baselines** | EEGNet, CSP+LDA, Cov+LR | Pre-Computed from `graph_results/` | Read directly from existing benchmark summaries |

---

## 3. Quickstart CLI Guide

### 1. Pre-Train ATCNet Models (`train_atcnet_models.py`)
```bash
# Pre-train all modes (cluster, submodular, single) with GPU or CPU acceleration
python atcnet_tools/train_atcnet_models.py --dataset Dreyer2023 --mode all --epochs 80 --batch-size 128 --device auto
```

### 2. Run Online Adaptive Simulation (`simulate_atcnet_adaptive.py`)
```bash
# Run trial-by-trial adaptive simulation (1-step fine-tuning every 8 trials)
python atcnet_tools/simulate_atcnet_adaptive.py --dataset Dreyer2023 --adapt-interval 8 --device auto
```

### 3. Generate Master Benchmark Report (`evaluate_atcnet_benchmarks.py`)
```bash
# Aggregate simulation metrics and generate trajectory plot vs pre-computed baselines
python atcnet_tools/evaluate_atcnet_benchmarks.py --dataset Dreyer2023
```

---

## 4. Output Directory Structure

```
c:\Users\stz\Documents\GitHub\csp_classifier\
 ├── trained_pipelines/atcnet/{dataset}/
 │    ├── subject_{id}/ATCNet_model.pt
 │    ├── cluster_mode_{id}/ATCNet_cluster_model.pt
 │    ├── submodular_top5_pooled/ATCNet_submodular_top5_model.pt
 │    └── atcnet_pretraining_accuracies.csv
 │
 ├── simulation_results/atcnet/{dataset}/
 │    ├── {subject_id}.csv (Trial-by-trial predictions)
 │
 └── graph_results/atcnet_benchmark/{dataset}/
      ├── atcnet_benchmark_summary.csv
      ├── atcnet_vs_eegnet_vs_classical_trajectories.png
      └── atcnet_master_benchmark_report.md
```
