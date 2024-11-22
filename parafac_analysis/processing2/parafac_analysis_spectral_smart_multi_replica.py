import itertools
import time
import matplotlib
import numpy as np
import tensortools as tt
from matplotlib import pyplot as plt
from mne import Epochs, pick_types
from scipy import stats
from scipy.fft import rfft
from pyedflib import highlevel

from data_classes.subject import Subject
from tkinter import filedialog as fd
import pandas as pd
import seaborn as sns


def process(subject, band, selected_channels, label_names, starting_rank, end_rank, step, replicas, verbose='ERROR'):
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

    epochs_data = epochs.get_data(verbose=verbose)
    labels = np.array(epochs.events[:, -1])
    yf_old = rfft(epochs_data)

    yf2 = np.load(r'parafac_analysis/reconstructions/2024-10-06T10-02-43.npy')
    yf = np.abs(yf_old[:, :, band[0]:band[1]])
    epochs_data = yf
    with open('parafac_analysis/reconstructions/initial.npy', 'wb') as out_file:
        np.save(out_file, yf, allow_pickle=True)

    perform_parafac_decomposition(epochs_data, labels, selected_channels, label_names, starting_rank, end_rank, step,
                                  band, replicas)


def decompose(x, ranks, replica_count):
    ensemble = tt.Ensemble(fit_method="ncp_hals")
    ensemble.fit(x, ranks=ranks, replicates=replica_count, verbose=False)
    return ensemble.results


def combine_atom(decompositions, atom_index, atom_aggregator):
    channel_factors = decompositions[1][:, atom_index]
    frequency_factors = decompositions[2][:, atom_index]

    for channel_factor_index, channel_factor in enumerate(channel_factors):
        atom_aggregator[channel_factor_index] += frequency_factors * channel_factor


def visualize_combined_atoms_as_heatmap(combined_atoms_left, combined_atoms_right, selected_channels, frequencies,
                                        a_label_name, b_label_name):
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


def perform_parafac_decomposition(x, y, selected_channels, label_names, starting_rank, end_rank, step, band, replicas):
    selected_frequencies = range(band[0], band[1])
    label_keys = list(label_names.keys())
    label_keys.sort()
    a_label = label_keys[0]
    a_label_name = label_names[a_label]
    b_label = label_keys[1]
    b_label_name = label_names[b_label]

    statistically_significant = None
    parafac_decompositions = None

    def is_alabel(el):
        return el['relation'] == 'greater'

    def is_blabel(el):
        return el['relation'] == 'less'

    last_rank = 0
    sim_l = []
    sim_r = []
    sim_b = []
    coverage = {'rank': [], 'percentage': []}
    ranks = range(starting_rank, end_rank, step)
    for current_rank in ranks:
        print('Processing rank {}'.format(current_rank))
        statistically_significant = []
        combined_atoms_both = []
        combined_atoms_left = []
        combined_atoms_right = []
        parafac_decompositions = decompose(x, range(current_rank, current_rank + 1), replicas)
        for replica in range(replicas):
            statistically_significant.append([])
            trial_factors = parafac_decompositions[current_rank][replica].factors[0]
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
                    statistically_significant[-1].append(
                        {'rank': current_rank, 'replica': 0, 'atom': atom_index, 'relation': 'greater',
                         'pvalue': pvalue_greater, 'weight': weight})
                if pvalue_less <= .05:
                    weight = np.average(np.abs(trial_factors[:, atom_index]))
                    statistically_significant[-1].append(
                        {'rank': current_rank, 'replica': 0, 'atom': atom_index, 'relation': 'less',
                         'pvalue': pvalue_less, 'weight': weight})
            percentage_of_significant_atoms = round(len(statistically_significant[-1]) / current_rank * 100, 2)
            alabel_count = len(list(filter(is_alabel, statistically_significant[-1])))
            blabel_count = len(list(filter(is_blabel, statistically_significant[-1])))
            coverage['rank'].append(current_rank)
            coverage['percentage'].append(len(statistically_significant[-1]) / current_rank)
            print('Statistically significant count: {}/{} {}% ({} {} and {} {})'.format(
                len(statistically_significant[-1]),
                current_rank,
                percentage_of_significant_atoms,
                alabel_count,
                a_label_name,
                blabel_count, b_label_name))

            rank_replica_decompositions = parafac_decompositions[current_rank][0].factors.factors

            combined_atoms_left.append([[0 for _ in selected_frequencies] for _ in selected_channels])
            combined_atoms_right.append([[0 for _ in selected_frequencies] for _ in selected_channels])
            combined_atoms_both.append([[0 for _ in selected_frequencies] for _ in selected_channels])

            for statistically_significant_decomposition in statistically_significant[-1]:
                if statistically_significant_decomposition['rank'] == current_rank and \
                        statistically_significant_decomposition[
                            'replica'] == 0:
                    if statistically_significant_decomposition['relation'] == 'greater':
                        combine_atom(rank_replica_decompositions, statistically_significant_decomposition['atom'],
                                     combined_atoms_left[-1])
                    else:
                        combine_atom(rank_replica_decompositions, statistically_significant_decomposition['atom'],
                                     combined_atoms_right[-1])
                    combine_atom(rank_replica_decompositions, statistically_significant_decomposition['atom'],
                                 combined_atoms_both[-1])

            # visualize_combined_atoms_as_heatmap(combined_atoms_left[-1], combined_atoms_right[-1], selected_channels,
            #                                     selected_frequencies, a_label_name, b_label_name)

        similarities_left = []
        similarities_right = []
        similarities_both = []
        for pair in list(itertools.combinations(range(replicas), 2)):
            p1, p2 = pair
            l1 = np.array(combined_atoms_left[p1])
            l2 = np.array(combined_atoms_left[p2])
            r1 = np.array(combined_atoms_right[p1])
            r2 = np.array(combined_atoms_right[p2])
            b1 = np.array(combined_atoms_both[p1])
            b2 = np.array(combined_atoms_both[p2])
            similarities_left.append((l1 * l2).sum(axis=1) / np.linalg.norm(l1, axis=1) / np.linalg.norm(l2, axis=1))
            similarities_right.append((r1 * r2).sum(axis=1) / np.linalg.norm(r1, axis=1) / np.linalg.norm(r2, axis=1))
            similarities_both.append((b1 * b2).sum(axis=1) / np.linalg.norm(b1, axis=1) / np.linalg.norm(b2, axis=1))
        sim_l.append(np.mean(similarities_left))
        sim_r.append(np.mean(similarities_right))
        sim_b.append(np.mean(similarities_both))
        print('Averaged similarity left: {}'.format(np.mean(similarities_left)))
        print('Averaged similarity right: {}'.format(np.mean(similarities_right)))
        print('Averaged similarity combined: {}'.format(np.mean(similarities_left)))

        last_rank = current_rank
    sns.lineplot(x=ranks, y=sim_b, label='Similarities')
    sns.lineplot(data=pd.DataFrame(coverage), x='rank', y='percentage', label='Coverage')
    plt.show()

    if parafac_decompositions and statistically_significant:
        data_file = {
            'decompositions': parafac_decompositions,
            'statistically_significant_decompositions': statistically_significant,
            'metadata': {
                'max_rank': last_rank,
                'replica_count': 1,
                'selected_channels': selected_channels,
                'a_label': a_label_name,
                'b_label': b_label_name,
                'highpass_cutoff': band[0],
                'lowpass_cutoff': band[1]
            }
        }
        heatmap_data = {
            'heatmap': combined_atoms_both,
            'frequencies': selected_frequencies,
            'channels': selected_channels
        }
        t = time.localtime()
        timestamp = time.strftime('%Y-%m-%dT%H-%M-%S', t)
        with open('parafac_analysis/results/{}.npy'.format(timestamp), 'wb') as out_file:
            np.save(out_file, data_file, allow_pickle=True)
        reconstructed = recostruct_from_atoms(parafac_decompositions[last_rank][0].factors.factors, last_rank)
        with open('parafac_analysis/reconstructions/{}.npy'.format(timestamp), 'wb') as out_file:
            np.save(out_file, reconstructed, allow_pickle=True)
        with open('parafac_analysis/significant_heatmaps/{}.npy'.format(timestamp), 'wb') as out_file:
            np.save(out_file, heatmap_data, allow_pickle=True)


def recostruct_from_atoms(atoms, last_rank):
    reconstructed = None
    for atom_index in range(last_rank):
        chn = atoms[1][:, atom_index]
        frq = atoms[2][:, atom_index]
        smp = atoms[0][:, atom_index]
        chn = chn.reshape((len(chn), 1))
        frq = frq.reshape((1, len(frq)))
        smp = smp.reshape((len(smp), 1, 1))
        res = np.kron(np.kron(chn, frq), smp)

        if reconstructed is None:
            reconstructed = res
        else:
            reconstructed = np.add(reconstructed, res)
    return reconstructed


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


def decomposition(subject, selected_channels, selected_frequency_band, starting_rank, end_rank, step, replicas):
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

    process(subject, selected_frequency_band, selected_channels, selected_labels, starting_rank, end_rank, step,
            replicas)


def main():
    subject_path = fd.askopenfilename(filetypes=[("European Data Format files", "*.edf")])
    subject = Subject(subject_path)
    print("Fill PARAFAC decomposition parameters")
    starting_rank = int(input('Starting rank: '))
    end_rank = int(input('End rank: '))
    step = int(input('Step: '))
    replicas = input('Replica count (default: 5): ')
    if replicas != '':
        replicas = int(replicas)
    else:
        replicas = 5

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

    decomposition(subject, selected_channels, selected_frequency_band, starting_rank, end_rank, step, replicas)


main()
