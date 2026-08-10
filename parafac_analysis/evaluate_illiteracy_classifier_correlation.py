"""
Evaluate Illiteracy Classifier Correlation
===========================================

Correlates PARAFAC BCI Illiteracy diagnostic scores (M_sig_min) with single-subject
classification accuracies across ATCNet and EEGNet models for all 7 MOABB datasets.
Saves CSV summary tables and generates correlation metrics.
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
from scipy import stats

DATASETS = [
    "Dreyer2023",
    "Dreyer2023A",
    "PhysionetMI",
    "Lee2019_MI",
    "GuttmannFlury2025_MI",
    "GuttmannFlury2025_ME",
    "Yang2025"
]

def load_classifier_accuracies(dataset_name):
    """Loads single-subject accuracies for ATCNet and EEGNet for a given dataset."""
    atc_path = os.path.join("trained_pipelines", "atcnet", dataset_name, "atcnet_pretraining_accuracies.csv")
    eeg_path = os.path.join("trained_pipelines", "eegnet", dataset_name, "eegnet_pretraining_accuracies.csv")
    
    atc_accs = {}
    eeg_accs = {}
    
    if os.path.exists(atc_path):
        atc_df = pd.read_csv(atc_path)
        atc_single = atc_df[atc_df['Pipeline'].astype(str).str.contains('Single', case=False, na=False)]
        for _, row in atc_single.iterrows():
            sub_str = str(row['Subject_Or_Cohort'])
            try:
                sub_id = int(sub_str.lower().replace('subject_', ''))
                atc_accs[sub_id] = float(row['Accuracy'])
            except Exception:
                pass

    if os.path.exists(eeg_path):
        eeg_df = pd.read_csv(eeg_path)
        eeg_single = eeg_df[eeg_df['Pipeline'].astype(str).str.contains('Single', case=False, na=False)]
        for _, row in eeg_single.iterrows():
            sub_str = str(row['Subject_Or_Cohort'])
            acc_val = row.get('Pretrain_Acc', row.get('Accuracy', np.nan))
            try:
                sub_id = int(sub_str.lower().replace('subject_', ''))
                eeg_accs[sub_id] = float(acc_val)
            except Exception:
                pass
                
    return atc_accs, eeg_accs


def run_correlation_analysis(diagnostics_dir="graph_results/parafac_diagnostics_strict"):
    summary_csv = os.path.join(diagnostics_dir, "parafac_illiteracy_summary.csv")
    if not os.path.exists(summary_csv):
        print(f"Error: {summary_csv} not found.")
        return

    ill_df = pd.read_csv(summary_csv)
    records = []
    
    print(f"\n================================================================================")
    print(f" Running PARAFAC Illiteracy vs Classifier Correlation Analysis ({diagnostics_dir})")
    print(f"================================================================================\n")

    all_matched = []

    for dataset_name in DATASETS:
        ds_ill = ill_df[ill_df['dataset'] == dataset_name].copy()
        if ds_ill.empty:
            continue

        atc_accs, eeg_accs = load_classifier_accuracies(dataset_name)

        ds_ill['subject_id_int'] = ds_ill['subject_id'].astype(str).apply(lambda x: int(x) if x.isdigit() else x)
        
        matched_records = []
        for _, row in ds_ill.iterrows():
            sub_id = row['subject_id_int']
            atc_acc = atc_accs.get(sub_id, np.nan)
            eeg_acc = eeg_accs.get(sub_id, np.nan)
            
            rec = {
                'dataset': dataset_name,
                'subject_id': sub_id,
                'm_sig_a': row['m_sig_a'],
                'm_sig_b': row['m_sig_b'],
                'm_sig_min': row['m_sig_min'],
                'is_bci_illiterate': (row['m_sig_min'] == 0),
                'atcnet_acc': atc_acc,
                'eegnet_acc': eeg_acc
            }
            matched_records.append(rec)
            all_matched.append(rec)

        ds_df = pd.DataFrame(matched_records)
        
        # Per-dataset metrics for ATCNet
        atc_valid = ds_df.dropna(subset=['atcnet_acc'])
        if len(atc_valid) > 2:
            atc_cap = atc_valid[~atc_valid['is_bci_illiterate']]['atcnet_acc']
            atc_ill = atc_valid[atc_valid['is_bci_illiterate']]['atcnet_acc']
            
            r_atc, p_atc = stats.pearsonr(atc_valid['m_sig_min'], atc_valid['atcnet_acc'])
            rho_atc, prho_atc = stats.spearmanr(atc_valid['m_sig_min'], atc_valid['atcnet_acc'])
            
            records.append({
                'dataset': dataset_name,
                'model': 'ATCNet',
                'n_total': len(atc_valid),
                'n_capable': len(atc_cap),
                'n_illiterate': len(atc_ill),
                'acc_capable_mean': atc_cap.mean() * 100 if len(atc_cap) > 0 else np.nan,
                'acc_illiterate_mean': atc_ill.mean() * 100 if len(atc_ill) > 0 else np.nan,
                'acc_delta': (atc_cap.mean() - atc_ill.mean()) * 100 if (len(atc_cap) > 0 and len(atc_ill) > 0) else np.nan,
                'pearson_r': r_atc,
                'pearson_p': p_atc,
                'spearman_rho': rho_atc,
                'spearman_p': prho_atc
            })

        # Per-dataset metrics for EEGNet
        eeg_valid = ds_df.dropna(subset=['eegnet_acc'])
        if len(eeg_valid) > 2:
            eeg_cap = eeg_valid[~eeg_valid['is_bci_illiterate']]['eegnet_acc']
            eeg_ill = eeg_valid[eeg_valid['is_bci_illiterate']]['eegnet_acc']
            
            r_eeg, p_eeg = stats.pearsonr(eeg_valid['m_sig_min'], eeg_valid['eegnet_acc'])
            rho_eeg, prho_eeg = stats.spearmanr(eeg_valid['m_sig_min'], eeg_valid['eegnet_acc'])
            
            records.append({
                'dataset': dataset_name,
                'model': 'EEGNet',
                'n_total': len(eeg_valid),
                'n_capable': len(eeg_cap),
                'n_illiterate': len(eeg_ill),
                'acc_capable_mean': eeg_cap.mean() * 100 if len(eeg_cap) > 0 else np.nan,
                'acc_illiterate_mean': eeg_ill.mean() * 100 if len(eeg_ill) > 0 else np.nan,
                'acc_delta': (eeg_cap.mean() - eeg_ill.mean()) * 100 if (len(eeg_cap) > 0 and len(eeg_ill) > 0) else np.nan,
                'pearson_r': r_eeg,
                'pearson_p': p_eeg,
                'spearman_rho': rho_eeg,
                'spearman_p': prho_eeg
            })

    # Grand Mean Population Analysis
    df_all = pd.DataFrame(all_matched)
    
    for model_name, col in [('ATCNet', 'atcnet_acc'), ('EEGNet', 'eegnet_acc')]:
        valid = df_all.dropna(subset=[col])
        if len(valid) > 2:
            cap = valid[~valid['is_bci_illiterate']][col]
            ill = valid[valid['is_bci_illiterate']][col]
            
            r, p = stats.pearsonr(valid['m_sig_min'], valid[col])
            rho, prho = stats.spearmanr(valid['m_sig_min'], valid[col])
            
            records.append({
                'dataset': 'GRAND_POPULATION',
                'model': model_name,
                'n_total': len(valid),
                'n_capable': len(cap),
                'n_illiterate': len(ill),
                'acc_capable_mean': cap.mean() * 100 if len(cap) > 0 else np.nan,
                'acc_illiterate_mean': ill.mean() * 100 if len(ill) > 0 else np.nan,
                'acc_delta': (cap.mean() - ill.mean()) * 100 if (len(cap) > 0 and len(ill) > 0) else np.nan,
                'pearson_r': r,
                'pearson_p': p,
                'spearman_rho': rho,
                'spearman_p': prho
            })
            
            # Generate Visual Correlation Plot for Grand Population
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            fig, ax = plt.subplots(figsize=(7.5, 5.5))
            sns.regplot(data=valid, x='m_sig_min', y=col, ax=ax,
                        scatter_kws={'alpha': 0.6, 'color': '#2980b9' if model_name == 'ATCNet' else '#8e44ad', 's': 40},
                        line_kws={'color': '#e74c3c', 'linewidth': 2.5})
            
            ax.set_title(f"Grand Population Correlation: {model_name} Accuracy vs $M_{{sig, min}}$ (N={len(valid)})",
                         fontsize=12, fontweight='bold', pad=12)
            ax.set_xlabel("PARAFAC Sensorimotor Power Index $M_{sig, min}$ ($\mu V^2 / Hz$)", fontsize=10, fontweight='bold')
            ax.set_ylabel(f"{model_name} Single-Subject Test Accuracy", fontsize=10, fontweight='bold')
            
            # Annotate correlation statistics
            text_str = f"Pearson r = {r:.3f} (p = {p:.2e})\nSpearman rho = {rho:.3f} (p = {prho:.2e})\nCapable Mean: {cap.mean()*100:.1f}%\nIlliterate Mean: {ill.mean()*100:.1f}%"
            bbox_props = dict(boxstyle="round,pad=0.5", fc="white", ec="gray", lw=1.5, alpha=0.9)
            ax.text(0.95, 0.05, text_str, transform=ax.transAxes, ha='right', va='bottom', bbox=bbox_props, fontsize=9.5, fontweight='bold')
            
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            out_img = os.path.join(diagnostics_dir, f"parafac_correlation_{model_name.lower()}_grand.png")
            plt.savefig(out_img, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved Grand Population Plot: {out_img}")

    res_df = pd.DataFrame(records)
    out_csv = os.path.join(diagnostics_dir, "parafac_classifier_correlation.csv")
    res_df.to_csv(out_csv, index=False)
    print(f"Saved Correlation Results CSV to: {out_csv}\n")
    print(res_df.to_string(index=False))


if __name__ == "__main__":
    run_correlation_analysis("graph_results/parafac_diagnostics_strict")
    run_correlation_analysis("graph_results/parafac_diagnostics")
