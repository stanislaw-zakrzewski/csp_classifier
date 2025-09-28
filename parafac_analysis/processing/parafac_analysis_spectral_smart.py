import time
import numpy as np
import tensortools as tt
from matplotlib import pyplot as plt
from mne import Epochs, preprocessing
from scipy import stats
import pandas as pd
import seaborn as sns
import itertools
from skopt.space import Integer
from skopt.utils import use_named_args
from skopt import gp_minimize
from yasa import irasa

ALPHA_NORMALITY = .05
ALPHA_HYPOTHESIS = .1


FREQUENCY = 256

def process(subject, band, selected_channels, label_names, starting_rank, end_rank, replicas, iterations, t_min, t_max,montage,
            verbose='DEBUG', save_file_name=None, visualize=False):
    raw_signal = subject.get_raw_copy()
    s_freq = subject.sampling_frequency
    if montage == 'standard_1020':
        raw_signal = raw_signal.drop_channels(['X5'], on_missing='ignore')
        raw_signal = raw_signal.drop_channels(['X3'], on_missing='ignore')



    filtered_raw_signal = raw_signal.filter(band[0], band[
        1], l_trans_bandwidth=2, h_trans_bandwidth=2, filter_length=s_freq * 2, fir_design='firwin',
                                            skip_by_annotation='edge', verbose=verbose)

    events = subject.events
    if FREQUENCY > 0:
        filtered_raw_signal = filtered_raw_signal.resample(sfreq=FREQUENCY)  # resample
        events[:, 0] = np.round(events[:, 0] * (FREQUENCY / float(s_freq))).astype(
            int)
        s_freq = FREQUENCY

    filtered_raw_signal.set_montage(montage)

    filtered_raw_signal = preprocessing.compute_current_source_density(filtered_raw_signal)

    if len(selected_channels) > 0:
        for channel in subject.electrode_names:
            if channel not in selected_channels:
                filtered_raw_signal.drop_channels([channel], on_missing='ignore')

    picks = list(range(len(filtered_raw_signal.info['ch_names'])))

    # picks = pick_types(filtered_raw_signal.info, meg=False, eeg=True, stim=False, eog=False,
    #                    exclude='bads')



    epochs = Epochs(filtered_raw_signal, subject.events, subject.id_dict, t_min, t_max, proj=True, picks=picks,
                    baseline=None, preload=True, verbose=verbose)

    epochs_data = epochs.get_data()

    labels = epochs.events[:, -1]

    osc_epochs_data = []
    decomposition_frequencies = list(range(band[0], band[1] + 1))
    for epoch_data in epochs_data:
        decomposition_frequencies, psd_frac, psd_osc = irasa(epoch_data, float(s_freq),
                                                      ch_names=selected_channels, band=band, win_sec=.5,
                                                      return_fit=False)
        np.save('irasa_data.npy', {'freq': decomposition_frequencies, 'frac': psd_frac, 'osc': psd_osc, 'original': epoch_data}, allow_pickle=True)

        osc_epochs_data.append(psd_osc)
    epochs_data = np.array(osc_epochs_data)

    # yf_old = rfft(epochs_data)
    #
    # yf_frequencies = rfftfreq(epochs_data.shape[-1], 1 / subject.sampling_frequency)
    #
    # def find_nearest(array, value):
    #     array = np.asarray(array)
    #     idx = (np.abs(array - value)).argmin()
    #     return idx
    #
    # lim_min = find_nearest(yf_frequencies, band[0])
    # lim_max = find_nearest(yf_frequencies, band[1]) + 1
    # decomposition_frequencies = yf_frequencies[lim_min:lim_max]
    #
    # yf = np.abs(yf_old[:, :, lim_min:lim_max])
    #
    # epochs_data = yf

    perform_parafac_decomposition(epochs_data, labels, selected_channels, label_names, starting_rank, end_rank,
                                  replicas, iterations, save_file_name, decomposition_frequencies, t_min, t_max,montage,
                                  visualize)


def decompose(x, ranks, replica_count):
    ensemble = tt.Ensemble(fit_method="ncp_hals")
    ensemble.fit(x, ranks=ranks, replicates=replica_count)
    return ensemble.results


def combine_atom(decompositions, atom_index, atom_aggregator):
    channel_factors = decompositions[1][:, atom_index]
    frequency_factors = decompositions[2][:, atom_index]

    for channel_factor_index, channel_factor in enumerate(channel_factors):
        atom_aggregator[channel_factor_index] += frequency_factors * channel_factor


def visualize_combined_atoms_as_heatmap(combined_atoms_a, combined_atoms_b, selected_channels, frequencies,
                                        a_label_name, b_label_name, title):
    df_a_data = {'Channel': [], "Frequency": [], "Amplitude": []}
    df_b_data = {'Channel': [], "Frequency": [], "Amplitude": []}
    for channel_index, channel_name in enumerate(selected_channels):
        for frequency_index, frequency in enumerate(frequencies):
            df_a_data['Channel'].append(channel_name)
            df_a_data['Frequency'].append(round(frequency, 1))
            df_a_data['Amplitude'].append(combined_atoms_a[channel_index][frequency_index])
            df_b_data['Channel'].append(channel_name)
            df_b_data['Frequency'].append(round(frequency, 1))
            df_b_data['Amplitude'].append(combined_atoms_b[channel_index][frequency_index])
    df_a = pd.DataFrame(df_a_data)
    df_a['Channel'] = pd.Categorical(df_a['Channel'], categories=selected_channels)
    df_a = df_a.sort_values('Channel')
    df_b = pd.DataFrame(df_b_data)
    df_b['Channel'] = pd.Categorical(df_b['Channel'], categories=selected_channels)
    df_b = df_b.sort_values('Channel')
    vmin = min(min(df_a['Amplitude']), min(df_b['Amplitude']))
    vmax = min(max(df_a['Amplitude']), max(df_b['Amplitude']))
    fig, axes = plt.subplots(ncols=3,
                             gridspec_kw=dict(width_ratios=[len(selected_channels), len(selected_channels), 0.5]))
    sns.heatmap(df_a.pivot(index='Frequency', columns='Channel', values='Amplitude'), ax=axes[0], cbar=False, vmin=vmin)
    axes[0].set_title(a_label_name)
    axes[0].invert_yaxis()
    g2 = sns.heatmap(df_b.pivot(index='Frequency', columns='Channel', values='Amplitude'), ax=axes[1], cbar=False,
                     vmax=vmax)
    g2.set(ylabel=None)
    axes[1].set_title(b_label_name)
    axes[1].invert_yaxis()
    # fig.colorbar(axes[1].collections[0], cax=axes[2])
    # fig.suptitle('Rank: {}'.format(title))
    # plt.show()


def is_a_label(el):
    return el['relation'] == 'greater'


def is_b_label(el):
    return el['relation'] == 'less'


def average_heatmaps(heatmaps):
    return np.sum(heatmaps, axis=0) / len(heatmaps)


def calculate_cosine_similarity(matrix_a, matrix_b):
    return (matrix_a * matrix_b).sum(axis=1) / np.linalg.norm(matrix_a, axis=1) / np.linalg.norm(matrix_b,
                                                                                                 axis=1)


def average_similarity(similarity_data, name):
    non_nan_similarity_data = np.where(np.isnan(similarity_data), 0, similarity_data)
    try:
        averaged_similarity = np.mean(non_nan_similarity_data)
        return averaged_similarity
    except:
        print('Nan averaged similarity for {} task, zeroing'.format(name))
        return 0


def perform_parafac_decomposition(x, y, selected_channels, label_names, starting_rank, end_rank, replicas,
                                  iterations, save_file_name, selected_frequencies, t_min, t_max, montage,visualize):
    label_keys = list(label_names.keys())
    label_keys.sort()
    a_label = label_keys[0]
    a_label_name = label_names[a_label]
    b_label = label_keys[1]
    b_label_name = label_names[b_label]

    space = [Integer(starting_rank, end_rank, name='rank')]

    combined_atoms_a = {}
    combined_atoms_b = {}
    combined_atoms_both = {}

    @use_named_args(space)
    def objective(**params):
        current_rank = params['rank']
        print('Processing rank {}'.format(current_rank))
        statistically_significant = []
        combined_atoms_a[current_rank] = []
        combined_atoms_b[current_rank] = []
        combined_atoms_both[current_rank] = []
        parafac_decompositions = decompose(x, range(current_rank, current_rank + 1), replicas)

        for replica in range(replicas):
            statistically_significant.append([])
            trial_factors = parafac_decompositions[current_rank][replica].factors[0]

            label_loc = {}  # indexes of given label key
            for label_key in set(y):
                label_loc[label_key] = np.where(y == label_key)[0]

            atoms_p_value_pairs = []
            for atom_index in range(len(trial_factors[0])):
                label_trial_factors = {}
                for label_key in label_loc.keys():
                    label_indexes = list(label_loc[label_key])
                    label_atoms = trial_factors[label_indexes]  # get trial factors for given label key
                    label_atoms = label_atoms[:, atom_index]  # select current atom
                    label_trial_factors[label_key] = label_atoms

                a_label_trial_factors = label_trial_factors[a_label]
                b_label_trial_factors = label_trial_factors[b_label]

                normality1 = stats.normaltest(a_label_trial_factors)[1]
                normality2 = stats.normaltest(b_label_trial_factors)[1]
                if normality1 < ALPHA_NORMALITY and normality2 < ALPHA_NORMALITY:
                    pvalue_less = stats.ttest_ind(a_label_trial_factors, b_label_trial_factors,
                                                  alternative='less').pvalue
                    pvalue_greater = stats.ttest_ind(a_label_trial_factors, b_label_trial_factors,
                                                     alternative='greater').pvalue
                else:
                    pvalue_less = stats.ranksums(a_label_trial_factors, b_label_trial_factors, alternative='less')[1]
                    pvalue_greater = \
                        stats.ranksums(a_label_trial_factors, b_label_trial_factors, alternative='greater')[1]
                atoms_p_value_pairs.append([pvalue_greater, pvalue_less])
                if pvalue_greater <= ALPHA_HYPOTHESIS:
                    statistically_significant[-1].append(
                        {'rank': current_rank, 'replica': replica, 'atom': atom_index, 'relation': 'greater',
                         'pvalue': pvalue_greater})
                if pvalue_less <= ALPHA_HYPOTHESIS:
                    statistically_significant[-1].append(
                        {'rank': current_rank, 'replica': replica, 'atom': atom_index, 'relation': 'less',
                         'pvalue': pvalue_less})
            if len(statistically_significant[-1]) == 0:
                min_p_values = np.min(atoms_p_value_pairs, axis=1)
                twenty_percent = max(1, int(.2 * len(trial_factors[0])))
                min_value = np.sort(min_p_values, axis=None)[twenty_percent-1]
                for atom_index, atoms_p_value_pair in enumerate(atoms_p_value_pairs):
                    pvalue_greater, pvalue_less = atoms_p_value_pair
                    if pvalue_greater > pvalue_less:
                        if pvalue_less <= min_value:
                            statistically_significant[-1].append(
                                {'rank': current_rank, 'replica': replica, 'atom': atom_index, 'relation': 'less',
                                 'pvalue': pvalue_less})
                    if pvalue_greater < pvalue_less:
                        if pvalue_greater <= min_value:
                            statistically_significant[-1].append(
                                {'rank': current_rank, 'replica': replica, 'atom': atom_index, 'relation': 'greater',
                                 'pvalue': pvalue_greater})

            if current_rank == 4:
                a0 = [[0 for _ in selected_frequencies] for _ in selected_channels]
                a1 = [[0 for _ in selected_frequencies] for _ in selected_channels]
                a2 = [[0 for _ in selected_frequencies] for _ in selected_channels]
                a3 = [[0 for _ in selected_frequencies] for _ in selected_channels]
                rrr = parafac_decompositions[current_rank][0].factors.factors
                combine_atom(rrr, 0, a0)
                combine_atom(rrr, 1, a1)
                combine_atom(rrr, 2, a2)
                combine_atom(rrr, 3, a3)
                a0 = np.array(a0)
                a1 = np.array(a1)
                a2 = np.array(a2)
                a3 = np.array(a3)
                a0 *= rrr[0][0][0]
                a1 *= rrr[0][0][1]
                a2 *= rrr[0][0][2]
                a3 *= rrr[0][0][3]
                o = {
                    'frequencies': selected_frequencies,
                    'channels': selected_channels,
                    'data': x,
                    'labels': y,
                    'label_names': label_names,
                    'decompositions': rrr,
                    'significant': statistically_significant
                }
                np.save('decomposition', o, allow_pickle=True)


            percentage_of_significant_atoms = round(len(statistically_significant[-1]) / current_rank * 100, 2)
            a_label_cardinality = len(list(filter(is_a_label, statistically_significant[-1])))
            b_label_cardinality = len(list(filter(is_b_label, statistically_significant[-1])))
            print('Statistically significant count: {}/{} {}% ({} {} and {} {})'.format(
                len(statistically_significant[-1]),
                current_rank,
                percentage_of_significant_atoms,
                a_label_cardinality,
                a_label_name,
                b_label_cardinality, b_label_name))

            rank_replica_decompositions = parafac_decompositions[current_rank][0].factors.factors

            combined_atoms_a[current_rank].append([[0 for _ in selected_frequencies] for _ in selected_channels])
            combined_atoms_b[current_rank].append([[0 for _ in selected_frequencies] for _ in selected_channels])
            combined_atoms_both[current_rank].append([[0 for _ in selected_frequencies] for _ in selected_channels])
            for statistically_significant_decomposition in statistically_significant[-1]:
                if statistically_significant_decomposition['rank'] == current_rank:
                    if statistically_significant_decomposition['relation'] == 'greater':
                        combine_atom(rank_replica_decompositions, statistically_significant_decomposition['atom'],
                                     combined_atoms_a[current_rank][-1])
                    else:
                        combine_atom(rank_replica_decompositions, statistically_significant_decomposition['atom'],
                                     combined_atoms_b[current_rank][-1])
                    combine_atom(rank_replica_decompositions, statistically_significant_decomposition['atom'],
                                 combined_atoms_both[current_rank][-1])

        similarities_a = []
        similarities_b = []
        similarities_both = []

        for pair in list(itertools.combinations(range(replicas), 2)):
            p1, p2 = pair
            a1 = np.array(combined_atoms_a[current_rank][p1])
            a2 = np.array(combined_atoms_a[current_rank][p2])
            b1 = np.array(combined_atoms_b[current_rank][p1])
            b2 = np.array(combined_atoms_b[current_rank][p2])
            both1 = np.array(combined_atoms_both[current_rank][p1])
            both2 = np.array(combined_atoms_both[current_rank][p2])
            similarities_a.append(calculate_cosine_similarity(a1, a2))
            similarities_b.append(calculate_cosine_similarity(b1, b2))
            similarities_both.append(calculate_cosine_similarity(both1, both2))

        average_similarity_a = min(1, average_similarity(similarities_a, a_label_name))
        average_similarity_b = min(1, average_similarity(similarities_b, b_label_name))
        average_similarity_both = min(1, average_similarity(similarities_both,
                                                     "{} and {}".format(a_label_name, b_label_name)))

        print('Averaged similarity {}: {}'.format(a_label_name, average_similarity_a))
        print('Averaged similarity {}: {}'.format(b_label_name, average_similarity_b))
        print('Averaged similarity combined: {}'.format(average_similarity_both))
        percentage_significant = 0
        for stst in statistically_significant:
            percentage_significant += len(stst)
        percentage_significant /= (current_rank * len(statistically_significant))

        if visualize:
            averaged_atoms_a = average_heatmaps(combined_atoms_a[current_rank])
            averaged_atoms_b = average_heatmaps(combined_atoms_b[current_rank])
            visualize_combined_atoms_as_heatmap(averaged_atoms_a, averaged_atoms_b, selected_channels,
                                                selected_frequencies, a_label_name, b_label_name, current_rank)

        def prot(val):
            if not isinstance(val, float): return 1
            a = .9 * val + .1
            if a < .1: return .1
            if a > 1: return 1
            return a

        return prot(1 - average_similarity_a) * prot(1 - average_similarity_b) * prot(
            1 - percentage_significant) * prot(((current_rank / end_rank) + 1) / 2.)

    res_gp = gp_minimize(objective, space, n_calls=iterations, random_state=0)
    best_rank = res_gp.x[0]
    print('Best rank: {}'.format(best_rank))
    averaged_heatmap = average_heatmaps(combined_atoms_both[best_rank])
    averaged_heatmap_a = average_heatmaps(combined_atoms_a[best_rank])
    averaged_heatmap_b = average_heatmaps(combined_atoms_b[best_rank])

    heatmap_data = {
        'heatmap': averaged_heatmap,
        'heatmap_a': averaged_heatmap_a,
        'heatmap_b': averaged_heatmap_b,
        'frequencies': selected_frequencies,
        'channels': selected_channels,
        'starting_rank': starting_rank,
        'end_rank': end_rank,
        'replica_count': replicas,
        'optimizer_iterations': iterations,
        'a_label_name': a_label_name,
        'b_label_name': b_label_name,
        't_min': t_min,
        't_max': t_max,
        'montage': montage
    }
    t = time.localtime()
    timestamp = time.strftime('%Y-%m-%dT%H-%M-%S', t)
    with open('parafac_analysis/{}.npy'.format(
            save_file_name if save_file_name is not None else timestamp), 'wb') as out_file:
        np.save(out_file, heatmap_data, allow_pickle=True)

    visualize_combined_atoms_as_heatmap(averaged_heatmap_a, averaged_heatmap_b, selected_channels,
                                        selected_frequencies, a_label_name, b_label_name, f'best rank: {best_rank}')


def adapt_selected_channels(subject, selected_channels):
    subject_channels = subject.electrode_names
    not_found = {}
    for selected_channel in selected_channels:
        if selected_channel not in subject_channels:
            not_found[selected_channel] = None

    for not_found_key in not_found:
        uppercase_key = not_found_key.upper()
        if uppercase_key in subject_channels:
            selected_channels = list(map(lambda x: x.replace(not_found_key, uppercase_key), selected_channels))

    return selected_channels
