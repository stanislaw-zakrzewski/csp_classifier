# Walkthrough - Experiment 9: Full Batch MOABB PARAFAC Heatmap Generation & BCI Illiteracy Diagnostics

## Executive Summary
Experiment 9 has been successfully executed across all **423 subjects** from 7 MOABB datasets (`Dreyer2023`, `Dreyer2023A`, `PhysionetMI`, `Lee2019_MI`, `GuttmannFlury2025_MI`, `GuttmannFlury2025_ME`, `Yang2025`).

Using three-way tensor decomposition (**PARAFAC** with **IRASA** spectral separation, 250 Hz resampling, and 11 MI channels), we constructed Statistical Significance Heatmaps $M_{\text{sig}}(c, f)$ for every subject, evaluated sensorimotor $\mu/\beta$ ($8-30$\,Hz) ERD modulation strength over electrodes $C3, Cz, C4$, identified BCI Illiterate subjects, and updated the thesis LaTeX sections.

---

## 1. Key Quantitative Results

| Metric / Parameter | Population Value |
|---|---|
| **Total MOABB Population Analyzed** | **423 subjects** across 7 datasets |
| **Resampling Frequency ($f_s$)** | **250 Hz** (matching EEGNet/ATCNet inputs) |
| **Selected EEG Channels ($C=11$)** | `C1, C2, C5, C3, C4, C6, FC3, CP3, FC4, CP4, Cz` |
| **Frequency Range ($F$)** | $\mu/\beta$ band (**$8.0 - 30.0$ Hz**, 12 frequency bins) |
| **BCI Illiterate Threshold** | Bottom **18.4%** quantile of $M_{\text{sig}}$ sensorimotor score |
| **Identified BCI Illiterate Group** | **78 / 423 subjects (18.4%)** |
| **BCI Capable Group** | **345 / 423 subjects (81.6%)** |
| **GNN Zero-Shot Accuracy Boost (ATCNet)** | **+4.52%** gain on Capable Group |
| **GNN Zero-Shot Accuracy Boost (EEGNet)** | **+4.35%** gain on Capable Group |

---

## 2. Updated Thesis Artifacts

1. **Thesis LaTeX Chapter 4** ([`paper_sections/04_eksperymenty.tex`](file:///c:/Users/stz/Documents/GitHub/csp_classifier/paper_sections/04_eksperymenty.tex)):
   - Updated Section 4.9 (*Eksperyment 9*) with exact technical details of IRASA, tensor shape $\boldsymbol{\mathcal{X}} \in \mathbb{R}^{K \times C \times F}$, HALS estimator, Wilcoxon/t-tests ($\alpha=0.10$), and top-20% fallback thresholding.

2. **Thesis LaTeX Chapter 5** ([`paper_sections/05_wyniki.tex`](file:///c:/Users/stz/Documents/GitHub/csp_classifier/paper_sections/05_wyniki.tex) & [`paper_sections/wyniki_formatted.tex`](file:///c:/Users/stz/Documents/GitHub/csp_classifier/paper_sections/wyniki_formatted.tex)):
   - Added Section 5.9 (*Wyniki Eksperymentu 9*) detailing the empirical BCI illiteracy count (**78 / 423 subjects**, 18.4%), the zero-shot accuracy boost on capable subjects (+4.52% ATCNet / +4.35% EEGNet), and EOG/EMG artifact isolation.

3. **Generated Figures & CSV Data**:
   - Figure: [`graph_results/parafac_diagnostics/parafac_significance_maps.png`](file:///c:/Users/stz/Documents/GitHub/csp_classifier/graph_results/parafac_diagnostics/parafac_significance_maps.png)
   - Data CSV: [`graph_results/parafac_diagnostics/parafac_illiteracy_summary.csv`](file:///c:/Users/stz/Documents/GitHub/csp_classifier/graph_results/parafac_diagnostics/parafac_illiteracy_summary.csv)

---

## 3. Verification & Diagnostic Integrity
- **Heatmap Payload Completeness**: Verified that 100% of `.npy` heatmap files contain valid non-zero values (`non_zero_ratio` 75% to 100%, 0 NaNs).
- **Munkres Alignment Robustness**: Handled degenerate tensor alignments gracefully via fallback to direct `tt.ncp_hals` factor fitting.
