# Future Research Roadmap: Architectural Enhancements, Novel Experiments & Real-World BCI Transfer Learning

This document outlines high-impact future research directions, deep learning model enhancements, GNN clustering innovations, cross-dataset transfer benchmarks, and wearable edge optimization paradigms for the BCI transfer learning framework.

---

## 1. Architectural & Deep Learning Enhancements

### A. Parameter-Efficient Fine-Tuning (LoRA / Adapters for BCI)
- **Current State**: Online adaptation fine-tunes all $2,018$ parameters of the EEGNet model (`full-model fine-tuning`).
- **Improvement**: Freeze the pre-trained spatial/temporal convolution backbone and insert **LoRA (Low-Rank Adaptation, $r=2$ or $4$)** or tiny bottleneck adapter layers into the linear classifier head.
- **Why it matters**: LoRA reduces gradient compute cost, prevents catastrophic forgetting of multi-subject donor spatial filters, and guarantees that fine-tuning stays regularized.

### B. Wearable-Optimized Attention Models (ATCNet vs. EEGConformer)
- **ATCNet (Attention-based Temporal Convolutional Network)**: Combines spatial-temporal convolutions, Dilated TCNs, and Multi-Head Self-Attention (~115k parameters).
- **EEGConformer**: Full Convolutional Transformer (~1.5M+ parameters).
- **Recommendation**: **ATCNet** is significantly more suitable for low-power wearable BCI deployment due to its compact parameter footprint and $O(N)$ temporal convolution efficiency.

---

## 2. GNN Latent Space & Dynamic Routing (Mixture-of-Experts)

### A. Dynamic GNN Target Routing Network (Soft MoE)
- **Current State**: Target subjects are assigned to 1 GNN cluster model.
- **Improvement**: Train a lightweight **GNN Gating Network (MoE Router)** that takes a new user's first 5–10 trial signals and computes dynamic soft attention weights $[\alpha_0, \alpha_1, \alpha_2, \alpha_3, \alpha_4]$ across the 5 cluster foundation models:
  $$\hat{y} = \sum_{c=0}^4 \alpha_c \cdot \text{EEGNet}_c(\mathbf{x})$$
- **Why it matters**: Eliminates hard cluster boundaries and creates a smooth, ensemble mixture of foundation experts tailored dynamically to each user.

### B. Supervised Contrastive Pre-Training (SupCon for EEG)
- **Experiment**: Incorporate Supervised Contrastive Learning (SupCon) during pre-training.
- **Why it matters**: Forces feature representations of $\text{Left Hand}$ vs. $\text{Right Hand}$ to separate into tight, distinct hyperspheres in latent space regardless of donor subject identity, boosting zero-shot transfer.

---

## 3. Cross-Dataset Foundation Benchmarking & Multi-Class Expansion

### A. Cross-Dataset Zero-Shot Generalization (Out-of-Study Transfer)
- **Current State**: Models are pre-trained and evaluated within the same dataset study.
- **Experiment**: Pre-train a unified **Foundation EEGNet** on `PhysionetMI` + `Lee2019_MI` ($160+$ subjects), and evaluate **zero-shot transfer onto completely unseen external datasets** (`Dreyer2023`, `GuttmannFlury2025`).
- **Why it matters**: Validates true dataset-agnostic foundation transfer across different EEG hardware devices and sampling environments.

### B. Multi-Class Motor Imagery Expansion (4-Class MI)
- **Experiment**: Extend the framework from 2-class ($\text{Left}$ vs $\text{Right}$) to 4-class motor imagery ($\text{Left Hand}$, $\text{Right Hand}$, $\text{Both Feet}$, $\text{Tongue}$) on datasets like `PhysionetMI` or `BCI Competition IV 2a`.

---

## 4. Edge Hardware & Robustness Stress-Testing

### A. Model Quantization Benchmark (FP32 $\to$ INT8)
- **Experiment**: Apply PyTorch dynamic/static INT8 quantization or ONNX Runtime Micro export to pre-trained EEGNet checkpoints.
- **Why it matters**: Reduces memory footprint from $8\text{ KB}$ down to **$2\text{ KB}$** and measures inference latency on real ARM Cortex-M microcontrollers or mobile CPUs.

### B. Electrode Reduction & Noise Degradation Robustness
- **Experiment**: Test model accuracy degradation when channel count is reduced from 11 channels down to **4 channels** (`C3`, `C4`, `Cz`, `FC3`) or when synthetic Gaussian scalp impedance noise is injected.

---

## 5. Neural Interpretability & Spatial Saliency

### A. Spatial Filter Topography Visualization (Grad-CAM / Saliency Maps)
- **Experiment**: Compute Grad-CAM or Deep Taylor Decomposition on the depthwise spatial convolution weights ($C \times 1$).
- **Why it matters**: Generates scalp activation heatmaps confirming that pre-trained foundation models are genuinely focusing on contralateral sensorimotor cortex channels ($C3$/$C4$) rather than ocular/EMG artifacts.
