import time
import matplotlib
import numpy as np
import tensortools as tt
from matplotlib import pyplot as plt
from mne import Epochs, pick_types
from scipy import stats
from scipy.fft import rfft

from data_classes.subject import Subject
from tkinter import filedialog as fd
import pandas as pd
import seaborn as sns


def process(subject, band, selected_channels, label_names, starting_rank, verbose='DEBUG'):
    tmin, tmax = .0, 2.

    raw_signal = subject.get_raw_copy()

    if len(selected_channels) > 0:
        for channel in subject.electrode_names:
            if channel not in selected_channels:
                raw_signal.drop_channels([channel])

    filtered_raw_signal = raw_signal.filter(band[0], band[
        1], l_trans_bandwidth=2, h_trans_bandwidth=2, filter_length=subject.sampling_frequency * 2, fir_design='firwin',
                                            skip_by_annotation='edge', verbose=verbose)

    picks = pick_types(filtered_raw_signal.info, meg=False, eeg=True, stim=False, eog=False,
                       exclude='bads')

    epochs = Epochs(filtered_raw_signal, subject.events, subject.id_dict, tmin, tmax, proj=True, picks=picks,
                    baseline=None, preload=True, verbose=verbose)

    epochs_data = epochs.get_data()
    labels = np.array(epochs.events[:, -1])
    yf = rfft(epochs_data)
    epochs_data = np.abs(yf[:, :, band[0]:band[1]])

    return perform_parafac_decomposition(epochs_data, labels, selected_channels, label_names, starting_rank, band)


def decompose(x, ranks, replica_count):
    ensemble = tt.Ensemble(fit_method="ncp_hals")
    ensemble.fit(x, ranks=ranks, replicates=replica_count)
    return ensemble.results


def combine_atom(decompositions, atom_index, atom_aggregator):
    channel_factors = decompositions[1][:, atom_index]
    frequency_factors = decompositions[2][:, atom_index]

    for channel_factor_index, channel_factor in enumerate(channel_factors):
        atom_aggregator[channel_factor_index] += frequency_factors * channel_factor


def visualize_combined_atoms_as_heatmap(combined_atoms_left, combined_atoms_right, selected_channels, frequencies, a_label_name, b_label_name):
    df_left_data = {'Channel': [], "Frequency": [], "Amplitude": []}
    df_right_data = {'Channel': [], "Frequency": [], "Amplitude": []}
    for channel_index, channel_name in enumerate(selected_channels):
        for frequency_index, frequency in enumerate(frequencies):
            df_left_data['Channel'].append(channel_name)
            df_left_data['Frequency'].append(frequency)
            df_left_data['Amplitude'].append(combined_atoms_left[channel_index][frequency_index])
            df_right_data['Channel'].append(channel_name)
            df_right_data['Frequency'].append(frequency)
            df_right_data['Amplitude'].append(combined_atoms_right[channel_index][frequency_index])
    df_left = pd.DataFrame(df_left_data)
    df_left['Channel'] = pd.Categorical(df_left['Channel'], categories=selected_channels)
    df_left = df_left.sort_values('Channel')
    df_right = pd.DataFrame(df_right_data)
    df_right['Channel'] = pd.Categorical(df_right['Channel'], categories=selected_channels)
    df_right = df_right.sort_values('Channel')
    vmin = min(min(df_left['Amplitude']), min(df_right['Amplitude']))
    vmax = min(max(df_left['Amplitude']), max(df_right['Amplitude']))
    fig, axes = plt.subplots(ncols=3,
                             gridspec_kw=dict(width_ratios=[len(selected_channels), len(selected_channels), 0.5]))
    sns.heatmap(df_left.pivot(index='Frequency', columns='Channel', values='Amplitude'), ax=axes[0], cbar=False,
                vmin=vmin)
    axes[0].set_title(a_label_name)
    axes[0].invert_yaxis()
    g2 = sns.heatmap(df_right.pivot(index='Frequency', columns='Channel', values='Amplitude'), ax=axes[1], cbar=False,
                     vmax=vmax)
    g2.set(ylabel=None)
    axes[1].set_title(b_label_name)
    axes[1].invert_yaxis()
    fig.colorbar(axes[1].collections[0], cax=axes[2])
    # plt.ylim(0, len(frequencies))
    plt.show()


def perform_parafac_decomposition(x, y, selected_channels, label_names, starting_rank, band):
    selected_frequencies = range(band[0], band[1])
    label_keys = list(label_names.keys())
    label_keys.sort()
    a_label = label_keys[0]
    a_label_name = label_names[a_label]
    b_label = label_keys[1]
    b_label_name = label_names[b_label]

    current_rank = starting_rank
    statistically_significant = None
    parafac_decompositions = None

    def is_alabel(el):
        return el['relation'] == 'greater'

    def is_blabel(el):
        return el['relation'] == 'less'

    while current_rank is not None:
        if current_rank is None:
            continue
        print('Processing rank {}'.format(current_rank))
        statistically_significant = []
        parafac_decompositions = decompose(x, range(current_rank, current_rank + 1), 1)
        trial_factors = parafac_decompositions[current_rank][0].factors[0]
        label_loc = {}
        for label in set(y):
            label_loc[label] = np.where(y == label)[0]
        for atom_index in range(len(trial_factors[0])):
            labelll = {}
            for label_name in label_loc.keys():
                aa = list(label_loc[label_name])
                label_atoms = trial_factors[aa]
                label_atoms = label_atoms[:, atom_index]
                labelll[label_name] = label_atoms

            normality1 = stats.normaltest(labelll[a_label])[1]
            normality2 = stats.normaltest(labelll[b_label])[1]
            if normality1 < .05 and normality2 < .05:
                pvalue_less = stats.ttest_ind(labelll[a_label], labelll[b_label], alternative='less').pvalue
                pvalue_greater = stats.ttest_ind(labelll[a_label], labelll[b_label], alternative='greater').pvalue
            else:
                pvalue_less = stats.ranksums(labelll[a_label], labelll[b_label], alternative='less')[1]
                pvalue_greater = stats.ranksums(labelll[a_label], labelll[b_label], alternative='greater')[1]
            if pvalue_greater <= .05:
                weight = np.average(np.abs(trial_factors[:, atom_index]))
                statistically_significant.append(
                    {'rank': current_rank, 'replica': 0, 'atom': atom_index, 'relation': 'greater',
                     'pvalue': pvalue_greater, 'weight': weight})
            if pvalue_less <= .05:
                weight = np.average(np.abs(trial_factors[:, atom_index]))
                statistically_significant.append(
                    {'rank': current_rank, 'replica': 0, 'atom': atom_index, 'relation': 'less',
                     'pvalue': pvalue_less, 'weight': weight})
        percentage_of_significant_atoms = round(len(statistically_significant) / current_rank * 100, 2)
        alabel_count = len(list(filter(is_alabel, statistically_significant)))
        blabel_count = len(list(filter(is_blabel, statistically_significant)))
        print('Statistically significant count: {}/{} {}% ({} {} and {} {})'.format(len(statistically_significant),
                                                                                    current_rank,
                                                                                    percentage_of_significant_atoms,
                                                                                    alabel_count,
                                                                                    a_label_name,
                                                                                    blabel_count, b_label_name))

        rank_replica_decompositions = parafac_decompositions[current_rank][0].factors.factors

        combined_atoms_left = [[0 for _ in selected_frequencies] for _ in selected_channels]
        combined_atoms_right = [[0 for _ in selected_frequencies] for _ in selected_channels]
        for statistically_significant_decomposition in statistically_significant:
            if statistically_significant_decomposition['rank'] == current_rank and \
                    statistically_significant_decomposition[
                        'replica'] == 0:
                if statistically_significant_decomposition['relation'] == 'greater':
                    combine_atom(rank_replica_decompositions, statistically_significant_decomposition['atom'],
                                 combined_atoms_left)
                else:
                    combine_atom(rank_replica_decompositions, statistically_significant_decomposition['atom'],
                                 combined_atoms_right)
        visualize_combined_atoms_as_heatmap(combined_atoms_left, combined_atoms_right, selected_channels,
                                            selected_frequencies, a_label_name, b_label_name)

        current_rank = input("Select current frequency (leaving empty end search):")
        if current_rank != '':
            current_rank = int(current_rank)
        else:
            current_rank = None

    # # Perform PARAFAC decomposition
    # parafac_decompositions = decompose(x, ranks, replica_count)
    #
    # statistically_significant = []
    # for rank in ranks:
    #     for replica in range(replica_count):
    #         trial_factors = parafac_decompositions[rank][replica].factors[0]
    #         label_loc = {}
    #         for label in set(y):
    #             label_loc[label] = np.where(y == label)[0]
    #         for atom_index in range(len(trial_factors[0])):
    #             labelll = {}
    #             for label_name in label_loc.keys():
    #                 aa = list(label_loc[label_name])
    #                 label_atoms = trial_factors[aa]
    #                 label_atoms = label_atoms[:, atom_index]
    #                 labelll[label_name] = label_atoms
    #
    #             normality1 = stats.normaltest(labelll[a_label])[1]
    #             normality2 = stats.normaltest(labelll[b_label])[1]
    #             if normality1 < .05 and normality2 < .05:
    #                 pvalue_less = stats.ttest_ind(labelll[a_label], labelll[b_label], alternative='less').pvalue
    #                 pvalue_greater = stats.ttest_ind(labelll[a_label], labelll[b_label], alternative='greater').pvalue
    #             else:
    #                 pvalue_less = stats.ranksums(labelll[a_label], labelll[b_label], alternative='less')[1]
    #                 pvalue_greater = stats.ranksums(labelll[a_label], labelll[b_label], alternative='greater')[1]
    #             if pvalue_greater <= .05:
    #                 weight = np.average(np.abs(trial_factors[:, atom_index]))
    #                 statistically_significant.append(
    #                     {'rank': rank, 'replica': replica, 'atom': atom_index, 'relation': 'greater',
    #                      'pvalue': pvalue_greater, 'weight': weight})
    #             if pvalue_less <= .05:
    #                 weight = np.average(np.abs(trial_factors[:, atom_index]))
    #                 statistically_significant.append(
    #                     {'rank': rank, 'replica': replica, 'atom': atom_index, 'relation': 'less',
    #                      'pvalue': pvalue_less, 'weight': weight})

    if parafac_decompositions and statistically_significant:
        data_file = {
            'decompositions': parafac_decompositions,
            'statistically_significant_decompositions': statistically_significant,
            'metadata': {
                'max_rank': current_rank,
                'replica_count': 1,
                'selected_channels': selected_channels,
                'a_label': a_label_name,
                'b_label': b_label_name,
                'highpass_cutoff': band[0],
                'lowpass_cutoff': band[1]
            }
        }
        t = time.localtime()
        timestamp = time.strftime('%Y-%m-%dT%H-%M-%S', t)
        with open('parafac_analysis/results/{}.npy'.format(timestamp), 'wb') as out_file:
            np.save(out_file, data_file, allow_pickle=True)

    return
    #
    # visualize_all_decompositions(statistically_significant, max_rank, replica_count)


def visualize_all_decompositions(statistically_significant_atoms, max_rank, replica_count):
    splits = split_into_groups(statistically_significant_atoms, max_rank)
    colors = get_colors(len(splits))
    labels = get_labels(len(splits))
    bottom = np.zeros(max_rank)

    for split_index, split in enumerate(splits):
        x = range(1, len(split) + 1)
        plt.bar(x, split, label=labels[split_index], color=colors[split_index], bottom=bottom)
        plt.title('Statistically significant atoms per rank for {} replicas'.format(replica_count))
        bottom = bottom + split
    plt.legend()
    plt.show()


def visualize_best_decomposition(parafac_decomposition_results, sorted_statistically_significant_atoms,
                                 best_decomposition, selected_channels, a_label_name, b_label_name):
    for atom_metadata in sorted_statistically_significant_atoms:
        rank = atom_metadata['rank']
        replica = atom_metadata['replica']
        if rank == best_decomposition['rank'] and replica == best_decomposition['replica']:
            atom_index = atom_metadata['atom']
            relation = atom_metadata['relation']
            pvalue = atom_metadata['pvalue']
            factors = parafac_decomposition_results[rank][replica].factors

            fig, (ax1, ax2) = plt.subplots(2)
            fig.suptitle(
                'Atom #{} - {} {} than {} (rank: {}, replica: {}, pvalue: {})'.format(atom_index, a_label_name,
                                                                                      relation, b_label_name,
                                                                                      rank, replica, pvalue))
            fig.set_figwidth(25)
            fig.set_figheight(10)
            x1 = selected_channels
            y1 = factors[1][:, atom_index]
            y2 = factors[2][:, atom_index]
            x2 = range(1, len(y2) + 1)
            ax1.plot(x1, y1)
            ax2.plot(x2, y2)
            plt.show()
            input("Press Enter to continue...")


def split_into_groups(to_split, max_rank, after_splitting=None, cutoff=.01, step=10):
    if after_splitting is None:
        after_splitting = []

    split = np.zeros(max_rank)
    next_to_split = []
    for ts in to_split:
        rank_index = ts['rank'] - 1
        pvalue = ts['pvalue']
        if pvalue > cutoff:
            split[rank_index] += 1
        else:
            next_to_split.append(ts)
    after_splitting.append(split)
    if len(next_to_split) == 0:
        return after_splitting
    else:
        return split_into_groups(next_to_split, max_rank, after_splitting, cutoff / step)


def get_colors(number_of_colors):
    cmap = matplotlib.colormaps.get_cmap('plasma')
    color_positions = np.linspace(0, .99, number_of_colors)
    colors = []
    for color_position in color_positions:
        colors.append(cmap(color_position))
    return colors


def get_labels(number_of_labels, cutoff=100, step=10):
    labels = []
    current_cutoff = cutoff
    for _ in range(number_of_labels):
        labels.append('pvalue > {}'.format(1 / current_cutoff))
        current_cutoff *= step
    return labels


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


def decomposition(subject, selected_channels, selected_frequency_band, starting_rank):
    label_names = {}
    for label_name in subject.id_dict:
        label_names[subject.id_dict[label_name]] = label_name
    selected_channels = adapt_selected_channels(subject, selected_channels)

    print("Available labels:")
    unique_events, event_counts = np.unique(subject.events[:, 2], return_counts=True)
    cardinalities = dict(zip(unique_events, event_counts))
    for label_name in label_names:
        print('\t* {}: {} (cardinality: {})'.format(label_name, label_names[label_name], cardinalities[label_name]))
    selected_label_1 = int(input("Select first label:"))
    selected_label_2 = int(input("Select second label:"))
    selected_labels = {
        selected_label_1: label_names[selected_label_1],
        selected_label_2: label_names[selected_label_2]
    }

    process(subject, selected_frequency_band, selected_channels, selected_labels, starting_rank)


def main():
    subject_path = fd.askopenfilename(filetypes=[("European Data Format files", "*.edf")])
    subject = Subject(subject_path)
    print("Fill PARAFAC decomposition parameters")
    starting_rank = int(input('Starting rank: '))

    selected_frequency_band = input('Selected frequency band (default: "2,24"): ')
    if selected_frequency_band:
        selected_frequency_band = selected_frequency_band.split(',')
        selected_frequency_band = (int(selected_frequency_band[0]), int(selected_frequency_band[1]))
    else:
        selected_frequency_band = (2, 24)

    print('Available channels: {}'.format(', '.join(subject.electrode_names)))
    selected_channels = input('Select channels (default: "C5,C3,C1,Cz,C2,C4,C6"): ')
    if selected_channels:
        selected_channels = selected_channels.split(',')
    else:
        selected_channels = ['C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6']

    decomposition(subject, selected_channels, selected_frequency_band, starting_rank)


main()
