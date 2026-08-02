# Cross-Session Multi-Day Transfer Suite (`cross_session_tools`)

A dedicated framework for evaluating **cross-session multi-day transfer stability** and **time-decay degradation** across EEGNet, ATCNet, CSP+LDA, and Covariance Tangent Space LR models on the **`Yang2025`** motor imagery dataset.

---

## 1. Experimental Setup & Session Division

In the `Yang2025` dataset, each subject recorded **600 total trial epochs** divided into **3 distinct multi-day sessions** (200 trials per session):

| Session | Session Name | Role in Experiment | Trial Count |
| :--- | :--- | :--- | :---: |
| **Session 0** | **Day 1** | **Pre-Training Set** (All models trained strictly on Day 1) | 200 trials / subject |
| **Session 1** | **Day 2** | **Out-of-Session Test Set 1** (Short-Term Multi-Day Gap) | 200 trials / subject |
| **Session 2** | **Day 3** | **Out-of-Session Test Set 2** (Long-Term Multi-Day Gap) | 200 trials / subject |

---

## 2. Models & Strategies Evaluated

| Strategy / Baseline | Description | Pre-Training Set (Day 1) | Test Set (Day 2 & 3) |
| :--- | :--- | :--- | :--- |
| **Strategy C (Matched Single)** | Single-subject models (ATCNet, EEGNet, CSP+LDA, Cov+LR) | Subject $v$ Day 1 (200 trials) | Subject $v$ Day 2 & Day 3 |
| **Strategy A (GNN Cluster Pooled)** | GNN Cluster-Mode models | Cluster Donors Day 1 (~12,000 trials) | Target Subject $v$ Day 2 & Day 3 |
| **Strategy B (Submodular Top-5)** | Top-5 Submodular model | Submodular Donors Day 1 (1,000 trials) | Target Subject $v$ Day 2 & Day 3 |
| **Baseline 1 (Scratch Cold-Start)** | Cold-start models | None (Day 1 excluded) | Trained from scratch on Day 2 & Day 3 |
| **Strategy C (Mismatched Donors)** | Mismatched donor models | Single Donor $d \neq v$ Day 1 | Target Subject $v$ Day 2 & Day 3 |

---

## 3. Quickstart CLI Execution

### Step 1: Pre-Train Session 0 (Day 1) Models
```bash
python cross_session_tools/train_cross_session_models.py --dataset Yang2025 --model all --epochs 80 --batch-size 128 --device auto
```

### Step 2: Run Out-of-Session Adaptive Simulation (Day 2 & Day 3)
```bash
python cross_session_tools/simulate_cross_session_adaptive.py --dataset Yang2025 --model all --adapt-mode head_only --adapt-interval 8 --device auto
```

### Step 3: Generate Master Benchmark Reports & Time-Decay Analysis
```bash
python cross_session_tools/evaluate_cross_session_benchmarks.py --dataset Yang2025
```

---

## 4. Output Artifacts Directory Structure

```
c:\Users\stz\Documents\GitHub\csp_classifier\
 ├── trained_pipelines/cross_session/Yang2025/
 │    ├── subject_{1..51}/
 │    ├── cluster_mode_{0..4}/
 │    └── submodular_top5_pooled/
 ├── simulation_results/cross_session/Yang2025/
 │    └── {1..51}.csv
 └── graph_results/cross_session_benchmark/Yang2025/
      ├── cross_session_day2_summary.csv
      ├── cross_session_day3_summary.csv
      ├── cross_session_decay_summary.csv
      ├── cross_session_day2_trajectories.png
      ├── cross_session_day3_trajectories.png
      └── cross_session_master_benchmark_report.md
```
