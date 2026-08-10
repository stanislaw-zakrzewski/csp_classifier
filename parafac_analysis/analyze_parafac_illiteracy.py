"""
Experiment 9: BCI Illiteracy & PARAFAC Statistical Diagnostic Analyzer
=======================================================================

Loads generated PARAFAC statistical significance heatmaps, evaluates sensorimotor mu/beta
ERD modulation strength over C3, Cz, C4, identifies BCI Illiterate subjects (< 18.4%),
correlates illiteracy with classification accuracy, and generates topomap figures.
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mne


def analyze_heatmaps(diagnostics_dir="graph_results/parafac_diagnostics"):
    heatmap_files = glob.glob(os.path.join(diagnostics_dir, "*", "subject_*_heatmap.npy"))
    if not heatmap_files:
        print(f"No heatmap files found under {diagnostics_dir}")
        return None

    records = []
    for filepath in heatmap_files:
        try:
            data = np.load(filepath, allow_pickle=True).item()
            subject_id = data['subject_id']
            dataset_name = data['dataset_name']
            heatmap = data['heatmap']  # Shape: (channels, frequencies)
            heatmap_a = data.get('heatmap_a', heatmap)
            heatmap_b = data.get('heatmap_b', heatmap)
            channels = data['channels']
            frequencies = np.array(data['frequencies'])

            # Find indices for sensorimotor channels
            sm_channels = ["C3", "Cz", "C4"]
            ch_indices = [i for i, ch in enumerate(channels) if ch in sm_channels]
            if len(ch_indices) == 0:
                ch_indices = list(range(min(3, len(channels))))

            c3_indices = [i for i, ch in enumerate(channels) if ch in ["C3", "Cz"]]
            c4_indices = [i for i, ch in enumerate(channels) if ch in ["C4", "Cz"]]
            if len(c3_indices) == 0: c3_indices = ch_indices
            if len(c4_indices) == 0: c4_indices = ch_indices

            # Mu/Beta frequency band (8-30 Hz)
            freq_indices = np.where((frequencies >= 8) & (frequencies <= 30))[0]

            # Compute Sensorimotor Significance Power Indices
            if len(freq_indices) > 0 and len(ch_indices) > 0:
                m_sig_both = float(np.mean(heatmap[np.ix_(ch_indices, freq_indices)]))
                m_sig_a = float(np.mean(heatmap_a[np.ix_(c4_indices, freq_indices)]))
                m_sig_b = float(np.mean(heatmap_b[np.ix_(c3_indices, freq_indices)]))
                m_sig_min = min(m_sig_a, m_sig_b)
            else:
                m_sig_both = float(np.mean(heatmap))
                m_sig_a = float(np.mean(heatmap_a))
                m_sig_b = float(np.mean(heatmap_b))
                m_sig_min = min(m_sig_a, m_sig_b)

            # Use m_sig_min as the primary bilateral BCI capacity score
            records.append({
                'dataset': dataset_name,
                'subject_id': subject_id,
                'filepath': filepath,
                'm_sig': m_sig_min,  # Primary bilateral modulation metric
                'm_sig_both': m_sig_both,
                'm_sig_a': m_sig_a,
                'm_sig_b': m_sig_b,
                'm_sig_min': m_sig_min
            })
        except Exception as e:
            print(f"Error reading {filepath}: {e}")

    df = pd.DataFrame(records)
    if df.empty:
        return df

    # Determine BCI Illiteracy threshold (bottom 18.4% quantile)
    illiteracy_threshold = df['m_sig'].quantile(0.184)
    df['is_bci_illiterate'] = df['m_sig'] <= illiteracy_threshold

    # Save summary CSV
    output_csv = os.path.join(diagnostics_dir, "parafac_illiteracy_summary.csv")
    df.to_csv(output_csv, index=False)
    print(f"Saved PARAFAC Illiteracy Summary CSV: {output_csv}")

def plot_distribution(df, illiteracy_threshold, out_dir):
    plt.figure(figsize=(7, 5))
    sns.histplot(df['m_sig'], kde=True, color='teal', bins=15)
    plt.axvline(illiteracy_threshold, color='red', linestyle='--', linewidth=2, label=f'Q18.4% Cutoff ({illiteracy_threshold:.3f})')
    plt.title("Bilateral Score ($M_{sig, min}$) Distribution", fontsize=12, fontweight='bold')
    plt.xlabel("Bilateral Power Index ($\mu V^2 / Hz$)", fontsize=10)
    plt.ylabel("Subject Count", fontsize=10)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "parafac_distribution.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def plot_boxplot(df, illiteracy_threshold, out_dir):
    plt.figure(figsize=(6, 5))
    sns.boxplot(x='is_bci_illiterate', y='m_sig', data=df, hue='is_bci_illiterate', palette=['#4c72b0', '#c44e52'], legend=False)
    plt.title("BCI Illiterate vs Capable Distribution", fontsize=12, fontweight='bold')
    plt.xlabel("Group", fontsize=10)
    plt.ylabel("Mean $M_{sig, min}$ Value ($\mu V^2 / Hz$)", fontsize=10)
    plt.xticks([0, 1], ['Capable (81.6%)', 'Illiterate (18.4%)'])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "parafac_boxplot.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def plot_topomaps_split(df, out_dir, dataset_title="7 Datasets Combined"):
    ch_names = ["C1", "C2", "C5", "C3", "C4", "C6", "FC3", "CP3", "FC4", "CP4", "Cz"]
    info = mne.create_info(ch_names=ch_names, sfreq=250, ch_types='eeg')
    montage = mne.channels.make_standard_montage('standard_1005')
    info.set_montage(montage)

    capable_df = df[~df['is_bci_illiterate']]
    if capable_df.empty:
        capable_df = df

    ch_a_capable = np.zeros(len(ch_names))
    ch_b_capable = np.zeros(len(ch_names))
    count_capable = 0

    ch_a_all = np.zeros(len(ch_names))
    ch_b_all = np.zeros(len(ch_names))
    count_all = 0

    for idx, row in df.iterrows():
        try:
            d = np.load(row['filepath'], allow_pickle=True).item()
            hm_a = d.get('heatmap_a', d['heatmap'])
            hm_b = d.get('heatmap_b', d['heatmap'])
            freqs = np.array(d['frequencies'])
            f_idx = np.where((freqs >= 8) & (freqs <= 30))[0]
            if len(f_idx) > 0:
                mean_a = np.mean(hm_a[:, f_idx], axis=1)
                mean_b = np.mean(hm_b[:, f_idx], axis=1)
                
                ch_a_all += mean_a
                ch_b_all += mean_b
                count_all += 1

                if not row['is_bci_illiterate']:
                    ch_a_capable += mean_a
                    ch_b_capable += mean_b
                    count_capable += 1
        except Exception:
            pass

    if count_capable > 0:
        ch_a_capable /= count_capable
        ch_b_capable /= count_capable

    if count_all > 0:
        ch_a_all /= count_all
        ch_b_all /= count_all

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))

    im_a0, _ = mne.viz.plot_topomap(ch_a_capable, info, axes=axes[0, 0], show=False, cmap='Blues')
    fig.colorbar(im_a0, ax=axes[0, 0], shrink=0.7, label='Power ($\mu V^2 / Hz$)')
    axes[0, 0].set_title("Left Hand Topomap\n(Non-BCI Illiterate)", fontsize=11, fontweight='bold')

    im_b0, _ = mne.viz.plot_topomap(ch_b_capable, info, axes=axes[0, 1], show=False, cmap='Oranges')
    fig.colorbar(im_b0, ax=axes[0, 1], shrink=0.7, label='Power ($\mu V^2 / Hz$)')
    axes[0, 1].set_title("Right Hand Topomap\n(Non-BCI Illiterate)", fontsize=11, fontweight='bold')

    im_a1, _ = mne.viz.plot_topomap(ch_a_all, info, axes=axes[1, 0], show=False, cmap='Blues')
    fig.colorbar(im_a1, ax=axes[1, 0], shrink=0.7, label='Power ($\mu V^2 / Hz$)')
    axes[1, 0].set_title("Left Hand Topomap\n(All Subjects)", fontsize=11, fontweight='bold')

    im_b1, _ = mne.viz.plot_topomap(ch_b_all, info, axes=axes[1, 1], show=False, cmap='Oranges')
    fig.colorbar(im_b1, ax=axes[1, 1], shrink=0.7, label='Power ($\mu V^2 / Hz$)')
    axes[1, 1].set_title("Right Hand Topomap\n(All Subjects)", fontsize=11, fontweight='bold')

    fig.suptitle(f"Motor Imagery Sensorimotor Topomaps ({dataset_title})", fontsize=13, fontweight='bold', y=0.99)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "parafac_topomaps_split.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def plot_quadrants(df, thresh_a, thresh_b, out_dir, dataset_title="7 Datasets Combined"):
    n_total = len(df)
    q1 = len(df[(df['m_sig_a'] > 0) & (df['m_sig_b'] > 0)])
    q2 = len(df[(df['m_sig_a'] == 0) & (df['m_sig_b'] > 0)])
    q3 = len(df[(df['m_sig_a'] > 0) & (df['m_sig_b'] == 0)])
    q4 = len(df[(df['m_sig_a'] == 0) & (df['m_sig_b'] == 0)])

    matrix_counts = np.array([[q2, q1], [q4, q3]])
    matrix_pcts = matrix_counts / n_total * 100

    labels = np.array([
        [f"Left-Side Illiterate\n\nN = {q2}\n({matrix_pcts[0,0]:.1f}%)", f"Non BCI Illiterate (Capable)\n\nN = {q1}\n({matrix_pcts[0,1]:.1f}%)"],
        [f"Both-Side Illiterate\n\nN = {q4}\n({matrix_pcts[1,0]:.1f}%)", f"Right-Side Illiterate\n\nN = {q3}\n({matrix_pcts[1,1]:.1f}%)"]
    ])

    fig, ax = plt.subplots(figsize=(7.5, 6))
    sns.heatmap(matrix_counts, annot=labels, fmt='', cmap='Blues', cbar=False,
                annot_kws={'size': 11, 'weight': 'bold'},
                xticklabels=['Left Hand Illiterate', 'Left Hand Capable'],
                yticklabels=['Right Hand Capable', 'Right Hand Illiterate'],
                linewidths=4, linecolor='white', ax=ax)

    ax.set_title(f"4-Quadrant BCI Illiteracy Diagnostic Matrix\n({dataset_title})", fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel("")
    ax.set_ylabel("")

    plt.tight_layout()
    out_path = os.path.join(out_dir, "parafac_illiteracy_quadrants.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def analyze_heatmaps(diagnostics_dir="graph_results/parafac_diagnostics"):
    heatmap_files = glob.glob(os.path.join(diagnostics_dir, "*", "subject_*_heatmap.npy"))
    if not heatmap_files:
        print(f"No heatmap files found under {diagnostics_dir}")
        return None

    records = []
    for filepath in heatmap_files:
        try:
            data = np.load(filepath, allow_pickle=True).item()
            subject_id = data['subject_id']
            dataset_name = data.get('dataset', os.path.basename(os.path.dirname(filepath)))

            hm = data['heatmap']
            hm_a = data.get('heatmap_a', hm)
            hm_b = data.get('heatmap_b', hm)

            channels = data['channels']
            frequencies = np.array(data['frequencies'])

            c3_idx = [i for i, c in enumerate(channels) if c in ['C3', 'Cz']]
            c4_idx = [i for i, c in enumerate(channels) if c in ['C4', 'Cz']]
            freq_idx = np.where((frequencies >= 8) & (frequencies <= 30))[0]

            if len(c4_idx) > 0 and len(freq_idx) > 0:
                m_sig_a = float(np.mean(hm_a[np.ix_(c4_idx, freq_idx)]))
            else:
                m_sig_a = 0.0

            if len(c3_idx) > 0 and len(freq_idx) > 0:
                m_sig_b = float(np.mean(hm_b[np.ix_(c3_idx, freq_idx)]))
            else:
                m_sig_b = 0.0

            m_sig_min = min(m_sig_a, m_sig_b)
            m_sig_both = float(np.mean(hm[np.ix_([i for i, c in enumerate(channels) if c in ['C3', 'Cz', 'C4']], freq_idx)]))

            records.append({
                'dataset': dataset_name,
                'subject_id': subject_id,
                'filepath': filepath,
                'm_sig': m_sig_min,
                'm_sig_both': m_sig_both,
                'm_sig_a': m_sig_a,
                'm_sig_b': m_sig_b,
                'm_sig_min': m_sig_min
            })
        except Exception as e:
            print(f"Error reading {filepath}: {e}")

    df = pd.DataFrame(records)
    if df.empty:
        return df

    illiteracy_threshold = df['m_sig'].quantile(0.184)
    df['is_bci_illiterate'] = df['m_sig'] <= illiteracy_threshold

    output_csv = os.path.join(diagnostics_dir, "parafac_illiteracy_summary.csv")
    df.to_csv(output_csv, index=False)
    print(f"Saved PARAFAC Illiteracy Summary CSV: {output_csv}")

    thresh_a = df['m_sig_a'].quantile(0.184)
    thresh_b = df['m_sig_b'].quantile(0.184)

    print("\n--- Generating Grand Mean Diagnostic Plots ---")
    plot_distribution(df, illiteracy_threshold, diagnostics_dir)
    plot_boxplot(df, illiteracy_threshold, diagnostics_dir)
    plot_topomaps_split(df, diagnostics_dir, dataset_title="7 Datasets Combined")
    plot_quadrants(df, thresh_a, thresh_b, diagnostics_dir, dataset_title="7 Datasets Combined")

    datasets = df['dataset'].unique()
    for ds in datasets:
        ds_df = df[df['dataset'] == ds]
        if len(ds_df) < 2:
            continue
        ds_dir = os.path.join(diagnostics_dir, ds)
        os.makedirs(ds_dir, exist_ok=True)
        print(f"\n--- Generating Per-Dataset Diagnostic Plots for: {ds} ---")
        ds_thresh_a = ds_df['m_sig_a'].quantile(0.184)
        ds_thresh_b = ds_df['m_sig_b'].quantile(0.184)
        ds_ill_thresh = ds_df['m_sig'].quantile(0.184)
        plot_distribution(ds_df, ds_ill_thresh, ds_dir)
        plot_boxplot(ds_df, ds_ill_thresh, ds_dir)
        plot_topomaps_split(ds_df, ds_dir, dataset_title=ds)
        plot_quadrants(ds_df, ds_thresh_a, ds_thresh_b, ds_dir, dataset_title=ds)

    return df


def main():
    df = analyze_heatmaps()
    if df is not None and not df.empty:
        total = len(df)
        illiterate_count = df['is_bci_illiterate'].sum()
        print(f"\n================================================================================")
        print(f" Analyzed {total} Subjects across PARAFAC Diagnostics")
        print(f" BCI Illiterate Subjects (< 18.4% cutoff): {illiterate_count} / {total} ({illiterate_count/total*100:.1f}%)")
        print(f"================================================================================\n")


if __name__ == "__main__":
    main()
