# Population Transfer Upper-Limit Suite (`population_transfer_tools`)

A dedicated framework for investigating the **theoretical upper limit of zero-shot cross-subject transfer**. For each dataset, a model (ATCNet, EEGNet, CSP+LDA, Cov+LR) is pre-trained on **ALL available subjects EXCEPT 5 randomly selected held-out test subjects** ($N - 5$ donors, e.g. 82 donors in `Dreyer2023`, 104 donors in `PhysionetMI`).

---

## 1. Experimental Design & Convergence Optimization

1. **Held-Out Test Set**: 5 test subjects are randomly selected and saved to `held_out_subjects.json`.
2. **Population Donor Pool ($N - 5$ Donors)**:
   - Trial data is concatenated across all remaining $N - 5$ donor subjects ($15,000$ to $20,000+$ trial epochs).
3. **Enhanced Convergence (120 Epochs)**:
   - PyTorch models (ATCNet and EEGNet) are pre-trained for **120 epochs** (up from 80) with AdamW ($\text{lr} = 10^{-3}$) and GPU acceleration (`--device auto`) to ensure full convergence on large donor pools.
4. **Out-of-Sample Simulation**:
   - Evaluates population models **EXCLUSIVELY on the 5 held-out test subjects** using Parameter-Efficient Head-Only Adaptation (`--adapt-mode head_only`).

---

## 2. Quickstart CLI Execution

### Step 1: Pre-Train Population Models on N - 5 Donors (120 Epochs)
```bash
python population_transfer_tools/train_population_models.py --dataset Dreyer2023 --epochs 120 --batch-size 128 --device auto
```

### Step 2: Run Adaptive Simulation Exclusively on Held-Out Test Subjects
```bash
python population_transfer_tools/simulate_population_adaptive.py --dataset Dreyer2023 --adapt-mode head_only --adapt-interval 8 --device auto
```

### Step 3: Generate Master Upper-Limit Benchmark Reports & Trajectories
```bash
python population_transfer_tools/evaluate_population_benchmarks.py --dataset Dreyer2023
```

---

## 3. Output Directory Structure

```
c:\Users\stz\Documents\GitHub\csp_classifier\
 ├── trained_pipelines/population_transfer/{dataset}/
 │    ├── held_out_subjects.json
 │    ├── ATCNet_population_model.pt
 │    ├── EEGNet_population_model.pt
 │    ├── CSP_LDA_population_model.pkl
 │    └── Cov_Tangent_Space_LR_population_model.pkl
 ├── simulation_results/population_transfer/{dataset}/
 │    └── {held_out_sub_ids}.csv
 └── graph_results/population_transfer_benchmark/{dataset}/
      ├── population_benchmark_summary.csv
      ├── population_upper_limit_trajectories.png
      └── population_upper_limit_benchmark_report.md
```
