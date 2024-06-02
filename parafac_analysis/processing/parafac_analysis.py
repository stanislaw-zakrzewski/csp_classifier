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


def process(subject, band, selected_channels, label_names, max_rank, replicas, verbose='DEBUG'):
    tmin, tmax = .0, 2.
    frequencies = 50

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
    epochs_data = np.abs(yf[:, :, 0:frequencies])

    return perform_parafac_decomposition(epochs_data, labels, selected_channels, label_names, max_rank, replicas)


def calculate_score_for_decomposition(rank_key, max_rank, pvalues, count):
    return rank_key / max_rank * min(pvalues) / count


def find_best_decomposition(statistically_significant_atoms, max_rank):
    structured_decompositions = {}
    best_decomposition = {'rank': -1, 'replica': -1, 'score': 1}

    for atom_data in statistically_significant_atoms:
        rank = atom_data['rank']
        replica = atom_data['replica']
        atom_index = atom_data['atom']
        pvalue = atom_data['pvalue']

        if rank not in structured_decompositions:
            structured_decompositions[rank] = {}
        if replica not in structured_decompositions[rank]:
            structured_decompositions[rank][replica] = {'count': 0, 'atoms': [], 'pvalues': []}
        decomposition = structured_decompositions[rank][replica]
        decomposition['count'] += 1
        decomposition['atoms'].append(atom_index)
        decomposition['pvalues'].append(pvalue)

    for rank_key in structured_decompositions:
        for replica_key in structured_decompositions[rank_key]:
            decomposition = structured_decompositions[rank_key][replica_key]
            count = decomposition['count']
            pvalues = decomposition['pvalues']
            score = calculate_score_for_decomposition(rank_key, max_rank, pvalues, count)
            if score < best_decomposition['score']:
                best_decomposition = {'rank': rank_key, 'replica': replica_key, 'score': score}
    return best_decomposition


def decompose(x, ranks, replica_count):
    ensemble = tt.Ensemble(fit_method="ncp_hals")
    ensemble.fit(x, ranks=ranks, replicates=replica_count)
    return ensemble.results


def perform_parafac_decomposition(x, y, selected_channels, label_names, max_rank, replica_count):
    ranks = range(1, max_rank + 1)
    label_keys = list(label_names.keys())
    label_keys.sort()
    a_label = label_keys[0]
    a_label_name = label_names[a_label]
    b_label = label_keys[1]
    b_label_name = label_names[b_label]

    # Perform PARAFAC decomposition
    parafac_decompositions = decompose(x, ranks, replica_count)

    statistically_significant = []
    for rank in ranks:
        for replica in range(replica_count):
            trial_factors = parafac_decompositions[rank][replica].factors[0]
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
                        {'rank': rank, 'replica': replica, 'atom': atom_index, 'relation': 'greater',
                         'pvalue': pvalue_greater, 'weight': weight})
                if pvalue_less <= .05:
                    weight = np.average(np.abs(trial_factors[:, atom_index]))
                    statistically_significant.append(
                        {'rank': rank, 'replica': replica, 'atom': atom_index, 'relation': 'less',
                         'pvalue': pvalue_less, 'weight': weight})

    data_file = {
        'decompositions': parafac_decompositions,
        'statistically_significant_decompositions': statistically_significant,
        'metadata': {
            'max_rank': max_rank,
            'replica_count': replica_count,
            'selected_channels': selected_channels,
            'a_label': a_label_name,
            'b_label': b_label_name
        }
    }
    t = time.localtime()
    timestamp = time.strftime('%Y-%m-%dT%H-%M-%S', t)
    with open('parafac_analysis/results/{}.npy'.format(timestamp), 'wb') as out_file:
        np.save(out_file, data_file, allow_pickle=True)

    return

    visualize_all_decompositions(statistically_significant, max_rank, replica_count)

    sorted_statistically_significant = sorted(statistically_significant, key=lambda stat_sign: stat_sign['pvalue'])

    best_decomposition = find_best_decomposition(statistically_significant, max_rank)

    visualize_best_decomposition(parafac_decompositions, sorted_statistically_significant, best_decomposition,
                                 selected_channels, a_label_name, b_label_name)


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


def decomposition(subject_path, selected_channels, selected_frequency_band, max_rank, replicas):
    subject = Subject(subject_path)
    label_names = {}
    for label_name in subject.id_dict:
        label_names[subject.id_dict[label_name]] = label_name
    selected_channels = adapt_selected_channels(subject, selected_channels)

    process(subject, selected_frequency_band, selected_channels, label_names, max_rank, replicas)


def main():
    subject_path = fd.askopenfilename(filetypes=[("European Data Format files", "*.edf")])
    print("Fill PARAFAC decomposition parameters")
    max_rank = int(input('Max rank: '))
    replicas = int(input('Replicas: '))
    selected_frequency_band = input('Selected frequency band (default: "2,24"): ')
    if selected_frequency_band:
        selected_frequency_band = selected_frequency_band.split(',')
        selected_frequency_band = (int(selected_frequency_band[0]), int(selected_frequency_band[1]))
    else:
        selected_frequency_band = (2, 24)
    selected_channels = input('Selected channels (default: "C5,C3,C1,Cz,C2,C4,C6"): ')
    if selected_channels:
        selected_channels.split(',')
    else:
        selected_channels = ['C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6']

    decomposition(subject_path, selected_channels, selected_frequency_band, max_rank, replicas)


main()
