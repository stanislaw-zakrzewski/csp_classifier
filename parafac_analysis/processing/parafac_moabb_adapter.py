"""
PARAFAC MOABB Adapter Module
============================

Performs IRASA spectral decomposition and PARAFAC non-negative tensor factorization
directly on MNE Raw/Epochs objects provided by MOABB datasets without requiring manual
EDF file prompts or GUI dialogues.
"""

import os
import time
import numpy as np
import pandas as pd
from scipy import stats
import tensortools as tt
from mne import Epochs, preprocessing, pick_channels
from yasa import irasa

ALPHA_NORMALITY = 0.05
ALPHA_HYPOTHESIS = 0.1
DEFAULT_RESAMPLE_FREQ = 250

DEFAULT_CHANNELS = ["C1", "C2", "C5", "C3", "C4", "C6", "FC3", "CP3", "FC4", "CP4", "Cz"]


def decompose_parafac(x, ranks, replica_count):
    try:
        ensemble = tt.Ensemble(fit_method="ncp_hals")
        ensemble.fit(x, ranks=ranks, replicates=replica_count, verbose=False)
        return ensemble.results
    except Exception:
        # Fallback when kruskal_align in tensortools raises IndexError on degenerate factors
        results = {}
        for r in ranks:
            models = []
            for _ in range(replica_count):
                m = tt.ncp_hals(x, rank=r, verbose=False)
                models.append(m)
            results[r] = models
        return results


def combine_atom(decompositions, atom_index, atom_aggregator):
    trial_amp = np.mean(decompositions[0][:, atom_index]) if len(decompositions[0]) > 0 else 1.0
    channel_factors = decompositions[1][:, atom_index]
    frequency_factors = decompositions[2][:, atom_index]

    for channel_factor_index, channel_factor in enumerate(channel_factors):
        atom_aggregator[channel_factor_index] += trial_amp * frequency_factors * channel_factor


def is_a_label(el):
    return el['relation'] == 'greater'


def is_b_label(el):
    return el['relation'] == 'less'


def process_moabb_raw(
    raw_signal,
    events,
    id_dict,
    subject_id,
    dataset_name,
    output_dir,
    band=(8, 30),
    selected_channels=None,
    starting_rank=5,
    end_rank=12,
    replicas=3,
    iterations=10,
    t_min=0.0,
    t_max=4.0,
    resample_freq=250,
    use_fallback=True,
    verbose="ERROR"
):
    """
    Processes a raw MNE signal from MOABB dataset through IRASA and PARAFAC tensor factorization.
    """
    if selected_channels is None:
        selected_channels = DEFAULT_CHANNELS

    # 1. Filter bandpass (8-30 Hz)
    s_freq = raw_signal.info['sfreq']
    filtered = raw_signal.copy().filter(
        band[0], band[1],
        l_trans_bandwidth=2, h_trans_bandwidth=2,
        filter_length=int(s_freq * 2), fir_design='firwin',
        skip_by_annotation='edge', verbose=verbose
    )

    # 2. Resample to target frequency (250 Hz)
    if resample_freq > 0 and s_freq != resample_freq:
        filtered = filtered.resample(sfreq=resample_freq, verbose=verbose)
        events_copy = events.copy()
        events_copy[:, 0] = np.round(events_copy[:, 0] * (resample_freq / float(s_freq))).astype(int)
        s_freq = resample_freq
    else:
        events_copy = events.copy()

    # 3. Pick selected channels present in raw signal
    available_ch = [ch for ch in selected_channels if ch in filtered.ch_names]
    if len(available_ch) == 0:
        available_ch = filtered.ch_names[:11]
    filtered.pick_channels(available_ch)

    # 4. Epoching
    epochs = Epochs(
        filtered, events_copy, event_id=id_dict,
        tmin=t_min, tmax=t_max, proj=True, baseline=None,
        preload=True, verbose=verbose
    )

    epochs_data = epochs.get_data() * 1e6  # Convert Volts -> microVolts (uV)
    labels = epochs.events[:, -1]

    # 5. IRASA Oscillatory PSD extraction
    osc_epochs_data = []
    freqs_sample, _, _ = irasa(
        epochs_data[0], float(s_freq),
        ch_names=available_ch, band=band, win_sec=0.5,
        return_fit=False
    )
    decomposition_frequencies = freqs_sample.tolist() if hasattr(freqs_sample, 'tolist') else list(freqs_sample)

    for epoch_data in epochs_data:
        freqs, psd_ap, psd_osc = irasa(
            epoch_data, float(s_freq),
            ch_names=available_ch, band=band, win_sec=0.5,
            return_fit=False
        )
        osc_epochs_data.append(psd_osc.values if hasattr(psd_osc, 'values') else psd_osc)
    tensor_data = np.array(osc_epochs_data)  # Shape: (trials, channels, frequencies)

    # 6. PARAFAC Decomposition
    present_labels = sorted(list(set(labels)))
    if len(present_labels) < 2:
        print(f"[{dataset_name}] Subject {subject_id}: fewer than 2 distinct event classes present in epochs ({present_labels}). Skipping.")
        return None

    a_label, b_label = present_labels[0], present_labels[1]
    a_label_name = f"Class_{a_label}"
    b_label_name = f"Class_{b_label}"

    combined_atoms_a = []
    combined_atoms_b = []
    combined_atoms_both = []
    statistically_significant_all = []

    for current_rank in range(starting_rank, end_rank + 1):
        parafac_decompositions = decompose_parafac(tensor_data, range(current_rank, current_rank + 1), replicas)

        for replica in range(replicas):
            statistically_significant = []
            trial_factors = parafac_decompositions[current_rank][replica].factors[0]

            label_loc = {lbl: np.where(labels == lbl)[0] for lbl in present_labels}
            atoms_p_value_pairs = []

            for atom_index in range(trial_factors.shape[1]):
                a_factors = trial_factors[label_loc[a_label], atom_index]
                b_factors = trial_factors[label_loc[b_label], atom_index]

                if len(a_factors) == 0 or len(b_factors) == 0:
                    continue

                normality1 = stats.normaltest(a_factors)[1] if len(a_factors) >= 8 else 1.0
                normality2 = stats.normaltest(b_factors)[1] if len(b_factors) >= 8 else 1.0

                if normality1 < ALPHA_NORMALITY and normality2 < ALPHA_NORMALITY:
                    pvalue_less = stats.ttest_ind(a_factors, b_factors, alternative='less').pvalue
                    pvalue_greater = stats.ttest_ind(a_factors, b_factors, alternative='greater').pvalue
                else:
                    pvalue_less = stats.ranksums(a_factors, b_factors, alternative='less')[1]
                    pvalue_greater = stats.ranksums(a_factors, b_factors, alternative='greater')[1]

                atoms_p_value_pairs.append([pvalue_greater, pvalue_less])

                if pvalue_greater <= ALPHA_HYPOTHESIS:
                    statistically_significant.append({
                        'rank': current_rank, 'replica': replica, 'atom': atom_index,
                        'relation': 'greater', 'pvalue': pvalue_greater
                    })
                if pvalue_less <= ALPHA_HYPOTHESIS:
                    statistically_significant.append({
                        'rank': current_rank, 'replica': replica, 'atom': atom_index,
                        'relation': 'less', 'pvalue': pvalue_less
                    })

            # Fallback: If no atoms passed strict threshold and use_fallback is True, pick top 20% most significant atoms
            if use_fallback and len(statistically_significant) == 0 and len(atoms_p_value_pairs) > 0:
                min_p_values = np.min(atoms_p_value_pairs, axis=1)
                twenty_percent = max(1, int(0.2 * len(atoms_p_value_pairs)))
                sorted_p = np.sort(min_p_values)
                cutoff_val = sorted_p[min(twenty_percent - 1, len(sorted_p) - 1)]
                for atom_index, (pvalue_greater, pvalue_less) in enumerate(atoms_p_value_pairs):
                    if pvalue_greater <= pvalue_less and pvalue_greater <= cutoff_val:
                        statistically_significant.append({
                            'rank': current_rank, 'replica': replica, 'atom': atom_index,
                            'relation': 'greater', 'pvalue': pvalue_greater
                        })
                    elif pvalue_less < pvalue_greater and pvalue_less <= cutoff_val:
                        statistically_significant.append({
                            'rank': current_rank, 'replica': replica, 'atom': atom_index,
                            'relation': 'less', 'pvalue': pvalue_less
                        })

            rank_replica_factors = parafac_decompositions[current_rank][replica].factors
            atom_a = np.zeros((len(available_ch), len(decomposition_frequencies)))
            atom_b = np.zeros((len(available_ch), len(decomposition_frequencies)))
            atom_both = np.zeros((len(available_ch), len(decomposition_frequencies)))

            for sig_atom in statistically_significant:
                if sig_atom['relation'] == 'greater':
                    combine_atom(rank_replica_factors, sig_atom['atom'], atom_a)
                    combine_atom(rank_replica_factors, sig_atom['atom'], atom_both)
                elif sig_atom['relation'] == 'less':
                    combine_atom(rank_replica_factors, sig_atom['atom'], atom_b)
                    combine_atom(rank_replica_factors, sig_atom['atom'], atom_both)

            combined_atoms_a.append(atom_a)
            combined_atoms_b.append(atom_b)
            combined_atoms_both.append(atom_both)
            statistically_significant_all.extend(statistically_significant)

    heatmap_a = np.mean(combined_atoms_a, axis=0) if len(combined_atoms_a) > 0 else np.zeros((len(available_ch), len(decomposition_frequencies)))
    heatmap_b = np.mean(combined_atoms_b, axis=0) if len(combined_atoms_b) > 0 else np.zeros((len(available_ch), len(decomposition_frequencies)))
    heatmap_both = np.mean(combined_atoms_both, axis=0) if len(combined_atoms_both) > 0 else np.zeros((len(available_ch), len(decomposition_frequencies)))

    # Save heatmap payload
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"subject_{subject_id}_heatmap.npy")
    payload = {
        'subject_id': subject_id,
        'dataset_name': dataset_name,
        'heatmap': heatmap_both,
        'heatmap_a': heatmap_a,
        'heatmap_b': heatmap_b,
        'frequencies': decomposition_frequencies,
        'channels': available_ch,
        'starting_rank': starting_rank,
        'end_rank': end_rank,
        'replicas': replicas,
        'iterations': iterations,
        'a_label_name': a_label_name,
        'b_label_name': b_label_name,
        't_min': t_min,
        't_max': t_max,
        'resample_freq': resample_freq
    }
    np.save(save_path, payload, allow_pickle=True)
    return payload
