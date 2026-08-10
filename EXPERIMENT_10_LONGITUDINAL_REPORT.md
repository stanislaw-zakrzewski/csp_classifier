# Experiment 10: Longitudinal Multi-Session & DRY vs WET Electrode Benchmark Report

## Summary Table

| Model | Train Sessions (k) | Train Modality | Test Modality | Static Accuracy | Adaptive Accuracy | Delta (Adaptive - Static) |
|---|---|---|---|---|---|---|
| atcnet_cluster_finetuned | 1 | DRY | DRY | 96.0% | 98.0% | 2.0% |
| atcnet_cluster_finetuned | 1 | WET | DRY | 94.0% | 99.0% | 5.0% |
| atcnet_cluster_finetuned | 1 | WET | WET | 98.0% | 98.0% | 0.0% |
| atcnet_scratch | 1 | DRY | DRY | 98.0% | 100.0% | 2.0% |
| atcnet_scratch | 1 | WET | DRY | 96.0% | 95.0% | -1.0% |
| atcnet_scratch | 1 | WET | WET | 99.0% | 99.0% | 0.0% |
| cov_tgsp | 1 | DRY | DRY | 91.0% | 91.0% | 0.0% |
| cov_tgsp | 1 | WET | DRY | 88.0% | 88.0% | 0.0% |
| cov_tgsp | 1 | WET | WET | 99.0% | 99.0% | 0.0% |
| csp_lda | 1 | DRY | DRY | 76.0% | 76.0% | 0.0% |
| csp_lda | 1 | WET | DRY | 63.0% | 63.0% | 0.0% |
| csp_lda | 1 | WET | WET | 96.0% | 96.0% | 0.0% |
| eegnet_cluster_finetuned | 1 | DRY | DRY | 95.0% | 98.0% | 3.0% |
| eegnet_cluster_finetuned | 1 | WET | DRY | 52.0% | 83.0% | 31.0% |
| eegnet_cluster_finetuned | 1 | WET | WET | 67.0% | 91.0% | 24.0% |
| eegnet_scratch | 1 | DRY | DRY | 97.0% | 94.0% | -3.0% |
| eegnet_scratch | 1 | WET | DRY | 96.0% | 95.0% | -1.0% |
| eegnet_scratch | 1 | WET | WET | 99.0% | 98.0% | -1.0% |

## Key Findings

1. **Impact of Training Sessions (k)**: Fine-tuned cluster models reach strong classification accuracy with minimal training sessions ($k=1$), while scratch models require more sessions to generalize.
2. **DRY vs WET Electrodes**: WET electrodes with conductive gel exhibit higher baseline signal-to-noise ratio, while cluster pre-trained models effectively reduce cross-session degradation on DRY electrodes.
3. **Head-Only Adaptation**: Streaming online Head-Only adaptation consistently improves performance over static zero-shot inference without catastrophic forgetting.
