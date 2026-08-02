# Cluster Selection Validation Suite (`cluster_selection_tools`)

A dedicated framework for empirically testing and validating how to select the best GNN cluster model for an unseen new target subject.

---

## 1. Evaluated Selection Strategies

1. **Option 1 (Zero-Shot GNN Distance)**: Projects resting/baseline covariance features into GNN embedding space and selects cluster $c_1^* = \arg\min_c \|\mathbf{z}_v - \boldsymbol{\mu}_c\|_2$.
2. **Option 2 (Soft Cluster Mixture Ensemble)**: Predicts using a soft GNN kernel weighted ensemble of all cluster models $\sum_c q_{v,c} M_c(x)$.
3. **Option 3 (First 8-Trial Confidence Selection)**: Runs all cluster models on the first 8 incoming trials and selects cluster $c_3^*$ with the highest confidence / lowest prediction entropy.
4. **Oracle Control (Upper Bound)**: Evaluates all cluster models on the full session and identifies the true best cluster model $c_{\text{oracle}}^*$.

---

## 2. Quickstart Execution

```bash
# Step 1: Run validation tests across target subjects
python cluster_selection_tools/validate_cluster_selection.py --dataset Dreyer2023 --model-type ATCNet --device auto

# Step 2: Generate Master Recommendation Report
python cluster_selection_tools/evaluate_selection_report.py
```
