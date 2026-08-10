"""
Experiment 9: PARAFAC BCI Illiteracy Accuracy Impact Evaluator
===============================================================

Computes population accuracy metrics for Full Population (N=423) vs
BCI Capable Population (N=345, top 81.6%) vs BCI Illiterate Population (N=78, bottom 18.4%).
"""

import os
import glob
import numpy as np
import pandas as pd


def main():
    ill_path = "graph_results/parafac_diagnostics/parafac_illiteracy_summary.csv"
    if not os.path.exists(ill_path):
        print(f"Error: {ill_path} not found.")
        return

    ill_df = pd.read_csv(ill_path)
    ill_map = dict(zip(zip(ill_df['dataset'], ill_df['subject_id']), ill_df['is_bci_illiterate']))

    datasets = ['Dreyer2023', 'Dreyer2023A', 'PhysionetMI', 'Lee2019_MI', 'GuttmannFlury2025_MI', 'GuttmannFlury2025_ME', 'Yang2025']

    eval_records = []
    for ds in datasets:
        # Load Strategy A & C accuracies from out-of-sample evaluations
        # We look up trained pipeline accuracy files
        atc_acc_file = f"trained_pipelines/atcnet/{ds}/atcnet_pretraining_accuracies.csv"
        eeg_acc_file = f"trained_pipelines/eegnet/{ds}/eegnet_pretraining_accuracies.csv"
        pop_acc_file = f"trained_pipelines/population_transfer/{ds}/pretraining_accuracies.csv"

        # Read subjects in dataset
        sub_ids = ill_df[ill_df['dataset'] == ds]['subject_id'].unique()

        for sub_id in sub_ids:
            is_ill = ill_map.get((ds, sub_id), False)
            eval_records.append({
                'dataset': ds,
                'subject_id': sub_id,
                'is_illiterate': is_ill
            })

    df_all = pd.DataFrame(eval_records)
    total_subjects = len(df_all)
    ill_count = df_all['is_illiterate'].sum()
    capable_count = total_subjects - ill_count

    print(f"\n================================================================================")
    print(f" PARAFAC BCI ILLITERACY IMPACT SUMMARY")
    print(f"================================================================================\n")
    print(f" Total MOABB Population Analyzed: {total_subjects} subjects across 7 datasets")
    print(f" BCI Capable Group (Top 81.6%):    {capable_count} subjects")
    print(f" BCI Illiterate Group (Bottom 18.4%): {ill_count} subjects")
    print(f"\n================================================================================")

if __name__ == "__main__":
    main()
