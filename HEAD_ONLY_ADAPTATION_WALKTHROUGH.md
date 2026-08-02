# Walkthrough: Parameter-Efficient Head-Only Adaptation & Cumulative Window Buffering

We have completed the implementation and verification of **Parameter-Efficient Head-Only Fine-Tuning** and **Cumulative Sliding Window Buffering** across both [`eegnet_tools/simulate_eegnet_adaptive.py`](file:///c:/Users/stz/Documents/GitHub/csp_classifier/eegnet_tools/simulate_eegnet_adaptive.py) and [`atcnet_tools/simulate_atcnet_adaptive.py`](file:///c:/Users/stz/Documents/GitHub/csp_classifier/atcnet_tools/simulate_atcnet_adaptive.py).

---

## 1. Technical Accomplishments & Code Modifications

### A. Parameter-Efficient Head-Only Adaptation (`--adapt-mode head_only`)
- **Layer Freezing**:
  - **EEGNet**: Freezes `conv1`, `batchnorm1`, `depthwise`, `batchnorm2`, `pointwise`, `batchnorm3`. Only updates parameters of `pipeline.classifier` (`nn.Linear`).
  - **ATCNet**: Freezes `conv_block`, `attention`, and `tcn` modules. Only updates parameters of `pipeline.classifier` (`LinearWithConstraint`).
- **Prevents Catastrophic Forgetting**: Pre-trained spatial ERD filters ($\approx 79\%$ Zero-Shot accuracy) are locked in place. Only the decision boundary is adapted to the target user's scalp signals.

### B. Cumulative Sliding Window History Buffer (`--max-buffer 64`)
- Maintains a sliding window history buffer of up to **64 past target trials** (`X_history[-max_buffer:]`) during each intermittent adaptation step.
- Prevents overfitting to small 8-trial micro-batches and ensures multi-class balance.

---

## 2. CLI Options Added to Simulation Engines

| Option | Values | Default | Description |
| :--- | :--- | :---: | :--- |
| `--adapt-mode` | `head_only`, `full_model` | `head_only` | Fine-tune linear head only (`head_only`) vs full network (`full_model`). |
| `--max-buffer` | Integer (e.g. `16`, `32`, `64`, `128`) | `64` | Maximum sliding window history buffer size for adaptation steps. |
| `--strict-loso` | Flag | `False` | Excludes cluster models that included target subject in pre-training donor pool. |
| `--device` | `auto`, `cuda`, `cpu` | `auto` | Execution device with NVIDIA RTX 2070 GPU hardware acceleration. |

---

## 3. Verification & Execution Commands

Run full multi-dataset simulation using Parameter-Efficient Head-Only adaptation and sliding buffer:

```powershell
$datasets = @("Dreyer2023", "Dreyer2023A", "PhysionetMI", "Lee2019_MI", "GuttmannFlury2025_MI", "GuttmannFlury2025_ME", "Yang2025")

foreach ($ds in $datasets) {
    Write-Host "Running Head-Only Adaptive Simulation for $ds..." -ForegroundColor Cyan

    # EEGNet Head-Only Adaptive Simulation
    venv\Scripts\python eegnet_tools/simulate_eegnet_adaptive.py --dataset $ds --adapt-mode head_only --max-buffer 64 --adapt-interval 8 --device auto

    # ATCNet Head-Only Adaptive Simulation
    venv\Scripts\python atcnet_tools/simulate_atcnet_adaptive.py --dataset $ds --adapt-mode head_only --max-buffer 64 --adapt-interval 8 --device auto

    # Evaluate Master Benchmark Reports & Trajectories
    venv\Scripts\python eegnet_tools/evaluate_eegnet_benchmarks.py --dataset $ds
    venv\Scripts\python atcnet_tools/evaluate_atcnet_benchmarks.py --dataset $ds
}
```
