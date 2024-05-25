import mne
import numpy as np
import matplotlib
from mne import Epochs, pick_types
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import ShuffleSplit
from sklearn.neural_network import MLPClassifier

from numpy.fft import fft, ifft
from matplotlib import pyplot as plt

from classifiers.morlet import cwt_morlet
from data_classes.subject import Subject
from tensorly.decomposition import parafac, constrained_parafac
from tensorly import unfold, cp_to_tensor
from scipy.fft import fft, fftfreq, rfft, rfftfreq
from sklearn.decomposition import TruncatedSVD
from scipy.signal import morlet
import tlviz
import tensortools as tt
from scipy import stats


def process(subject, bands, selected_channels, n_splits=10, reg=None, verbose='DEBUG', score_window_flag=False):
    tmin, tmax = 0., 2.
    frequencies = 50

    raw_signals = []
    for i in range(len(bands)):
        raw_signals.append(subject.get_raw_copy())

    if len(selected_channels) > 0:
        for raw_signal in raw_signals:
            for channel in subject.electrode_names:
                if channel not in selected_channels:
                    raw_signal.drop_channels([channel])

    filtered_raw_signals = []
    epochs = []
    epochs_train = []
    epochs_data = []
    epochs_data_train = []

    # Apply band-pass filter
    for index, band in enumerate(bands):
        filtered_raw_signals.append(
            raw_signals[index].filter(band[0], band[1], l_trans_bandwidth=2, h_trans_bandwidth=2, filter_length=1024,
                                      fir_design='firwin',
                                      skip_by_annotation='edge', verbose=verbose))

    picks = pick_types(filtered_raw_signals[0].info, meg=False, eeg=True, stim=False, eog=False,
                       exclude='bads')

    for index, band in enumerate(bands):
        epochs.append(
            Epochs(filtered_raw_signals[index], subject.events, subject.id_dict, tmin, tmax, proj=True, picks=picks,
                   baseline=None, preload=True, verbose=verbose))
        epochs_train.append(epochs[index].copy().crop(tmin=tmin, tmax=tmax))

        epochs_data.append(epochs[index].get_data())
        epochs_data_train.append(epochs_train[index].get_data())
    labels = np.array(epochs[0].events[:, -1])

    yf_train = rfft(epochs_data_train[0])
    epochs_data_train[0] = np.abs(yf_train[:, :, 0:frequencies])
    # xf = rfftfreq(epochs_data_train[0].shape[-1], 1 / 512)
    # plt.plot(xf[0:frequencies], np.abs(yf[1,0,0:frequencies]), marker="o")
    # plt.show()
    # epochs_data_train[0] = time_frequency_analysis(epochs_data_train[0])
    # epochs_data_trainpochs_data[0] = time_frequency_analysis(epochs_data[0])

    yf = rfft(epochs_data[0])
    epochs_data[0] = np.abs(yf[:, :, 0:frequencies])

    cv = ShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=42)
    cv_split = cv.split(epochs_data_train[0])
    # Assemble a classifier
    # classifier = MLPClassifier(hidden_layer_sizes=(10), random_state=1, n_iter_no_change=100,
    #                             learning_rate_init=0.01, max_iter=10000, )  # Originally: LinearDiscriminantAnalysis()
    classifier = MLPClassifier(hidden_layer_sizes=(128, 32, 8), random_state=1, n_iter_no_change=100,
                               learning_rate_init=0.01, max_iter=10000, )  # Originally: LinearDiscriminantAnalysis()
    # classifier = LinearDiscriminantAnalysis()
    # classifier = RandomForestClassifier(max_depth=20, n_estimators=10, max_features=10)
    mne.set_log_level('warning')
    do_parafac(epochs_data_train[0], labels, selected_channels)
    return

    # Initialize the TruncatedSVD model
    svd = TruncatedSVD(n_components=10)

    sfreq = raw_signals[0].info['sfreq']
    w_length = int(sfreq)  # running classifier: window length
    w_step = int(sfreq * 0.1)  # running classifier: window step size
    w_start = np.arange(0, epochs_data[0].shape[2] - w_length, w_step)

    scores_windows = []
    all_predictions = []
    all_correct = []
    for train_idx, test_idx in cv_split:
        y_train, y_test = labels[train_idx], labels[test_idx]

        x_train_csp = []
        x_test_csp = []
        for edt in epochs_data_train:
            if len(x_train_csp) > 0:
                x_train_csp = np.concatenate((x_train_csp, get_atoms(edt[train_idx])), axis=1)
                x_test_csp = np.concatenate((x_test_csp, get_atoms(edt[test_idx])), axis=1)
            else:
                x_train_csp = get_atoms(edt[train_idx], raw_signals[0].ch_names)
                x_test_csp = get_atoms(edt[test_idx])

        # Fit the model to the data
        svd.fit(x_train_csp, y_train)

        # Print the factors
        # print("U:", svd.components_)
        # print("S:", svd.singular_values_)
        x_train_csp = svd.transform(x_train_csp)
        x_test_csp = svd.transform(x_test_csp)
        classifier.fit(x_train_csp, y_train)

        # plt.plot(classifier.loss_curve_)
        # plt.title("Loss Curve", fontsize=14)
        # plt.xlabel('Iterations')
        # plt.ylabel('Cost')
        # plt.show()

        predictions = classifier.predict(x_test_csp)
        all_predictions.append(predictions)
        all_correct.append(y_test)

        # running classifier: test classifier on sliding window
        if score_window_flag:
            score_this_window = []
            for n in w_start:
                x_test_csp = []
                for edt in epochs_data:
                    if len(x_test_csp) > 0:
                        x_test_csp = np.concatenate(
                            (x_test_csp, get_atoms(edt[test_idx][:, :, n:(n + w_length)])),
                            axis=1)
                    else:
                        x_test_csp = get_atoms(edt[test_idx][:, :, n:(n + w_length)])
                score_this_window.append(classifier.score(x_test_csp, y_test))
            scores_windows.append(score_this_window)
    w_times = []
    if score_window_flag:
        w_times = (w_start + w_length / 2.) / sfreq + epochs[0].tmin
    return w_times, scores_windows, svd, epochs[0].info, all_predictions, all_correct, classifier, subject.info


def time_frequency_analysis(data):
    train_data = []
    for X in data:
        train_data.append(cwt_morlet(X, 512, freqs=np.arange(5, 30, 1), use_fft=True, n_cycles=5))
    return np.asarray(train_data)


def do_parafac(x, y, selected_channels):
    max_rank = 9
    ranks = range(1, max_rank + 1)
    replicas = 2
    a_label = 1
    b_label = 2
    ensemble = tt.Ensemble(fit_method="ncp_hals")
    ensemble.fit(x, ranks=ranks, replicates=replicas)
    parafac_decompositions = {}
    parafac_results = {}
    statistically_significant = []
    for rank in ranks:
        parafac_decompositions[rank] = {}
        for replica in range(replicas):
            # print('RANK: {}'.format(rank))
            # weights, factors = constrained_parafac(x, rank=rank)
            trial_factors = ensemble.results[rank][replica].factors[0]
            # parafac_decompositions[rank][replica] = {
            #     # 'weights': weights,
            #     'factors': factors
            # }
            label_loc = {}
            for label in set(y):
                label_loc[label] = np.where(y == label)[0]
            max_difference = 0
            max_difference_index = -1
            for atom_index in range(len(trial_factors[0])):
                label_values = {}
                labelll = {}
                # print('\tATOM {}'.format(atom_index + 1))
                for label_name in label_loc.keys():
                    aa = list(label_loc[label_name])
                    label_atoms = trial_factors[aa]
                    label_atoms = label_atoms[:, atom_index]
                    labelll[label_name] = label_atoms
                    label_values[label_name] = np.sum(label_atoms) / len(label_loc[label_name])
                    # print('\t\t{}: {}'.format(label_name, label_values[label_name]))
                new_max_difference = np.abs(list(label_values.values())[0] - list(label_values.values())[1])
                if max_difference < new_max_difference:
                    max_difference = new_max_difference
                    max_difference_index = atom_index
                normality1 = stats.normaltest(labelll[a_label]).pvalue
                normality2 = stats.normaltest(labelll[b_label]).pvalue
                pvalue_less = stats.ttest_ind(labelll[a_label], labelll[b_label], alternative='less').pvalue
                pvalue_greater = stats.ttest_ind(labelll[a_label], labelll[b_label], alternative='greater').pvalue
                if pvalue_greater <= .05:
                    statistically_significant.append(
                        {'rank': rank, 'replica': replica, 'atom': atom_index, 'type': 'greater',
                         'pvalue': pvalue_greater, 'averages': label_values})
                if pvalue_less <= .05:
                    statistically_significant.append(
                        {'rank': rank, 'replica': replica, 'atom': atom_index, 'type': 'less', 'pvalue': pvalue_less,
                         'averages': label_values})
            parafac_results[rank] = {
                'difference': max_difference,
                'idx': max_difference_index
            }
    differences = []
    for rank in parafac_results.keys():
        differences.append(parafac_results[rank]['difference'])
    statistically_significant_per_rank = np.zeros(max_rank)

    splits = split_into_groups(statistically_significant, max_rank)
    colors = get_colors(len(splits))
    labels = get_labels(len(splits))
    bottom = np.zeros(max_rank)

    for split_index, split in enumerate(splits):
        x = range(1, len(split) + 1)
        plt.bar(x, split, label=labels[split_index], color=colors[split_index], bottom=bottom)
        bottom = bottom + split
    plt.legend()
    plt.show()
    sorted_statistically_significant = sorted(statistically_significant, key=lambda stat_sign: stat_sign['pvalue'])
    for sss in sorted_statistically_significant:
        rank = sss['rank']
        replica = sss['replica']
        atom_index = sss['atom']
        factors = ensemble.results[rank][replica].factors

        fig, (ax1, ax2) = plt.subplots(2)
        fig.suptitle('Atom #{} (rank: {}, replica: {})'.format(sss['atom'], rank, replica))
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




    # s1 = np.zeros(max_rank)
    # s2 = np.zeros(max_rank)
    # s3 = np.zeros(max_rank)
    # s4 = np.zeros(max_rank)
    # s5 = np.zeros(max_rank)
    #
    # for stat_res in statistically_significant:
    #     print(stat_res)
    #     rank_index = stat_res['rank'] - 1
    #     pvalue = stat_res['pvalue']
    #     if pvalue > .01:
    #         s1[rank_index] += 1
    #     elif pvalue > .001:
    #         s2[rank_index] += 1
    #     elif pvalue > .0001:
    #         s3[rank_index] += 1
    #     elif pvalue > .00001:
    #         s4[rank_index] += 1
    #     elif pvalue > .000001:
    #         s5[rank_index] += 1
    #
    #     statistically_significant_per_rank[stat_res['rank'] - 1] += 1
    # x = range(1, len(statistically_significant_per_rank) + 1)
    # bottom = np.zeros(max_rank)
    # cmap = matplotlib.colormaps.get_cmap('plasma')
    # plt.bar(x, s1, label='pvalue > .01', color=cmap(0))
    # bottom = bottom + s1
    # plt.bar(x, s2, bottom=bottom, label='pvalue > .001', color=cmap(0.25))
    # bottom = bottom + s2
    # plt.bar(x, s3, bottom=bottom, label='pvalue > .0001', color=cmap(0.5))
    # bottom = bottom + s3
    # plt.bar(x, s4, bottom=bottom, label='pvalue > .00001', color=cmap(0.75))
    # bottom = bottom + s4
    # plt.bar(x, s5, bottom=bottom, label='pvalue > .000001', color=cmap(.99))
    # plt.legend()
    # plt.show()


def split_into_groups(to_split, max_rank, after_splitting=[], cutoff=.01, step=10):
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
        labels.append('pvalue > {}'.format(1/current_cutoff))
        current_cutoff *= step
    return labels


def get_atoms(x_train, ch_names=[]):
    # for x_1 in x_train:
    #     for x_2 in x_1:
    freq = np.fft.rfftfreq(500, d=1. / 250)[0:50]

    weights, factors = parafac(x_train, rank=5)
    print('oko')
    tlviz.visualisation.components_plot((weights, factors))
    plt.show()
    channels = factors[1]
    frequencies_list = factors[2]
    plot_data = []
    for channel_index, channel in enumerate(channels):
        plot_data.append([])
        for index in range(len(frequencies_list[0])):
            plot_data[-1].append(np.array([i[index] for i in frequencies_list]) * channel[index])

    fig, ax = plt.subplots(len(plot_data), 1)
    fig.tight_layout(h_pad=3)
    for row_index, row in enumerate(plot_data):
        for component_index, component in enumerate(row):
            ax[row_index].plot(freq, component, label=component_index + 1)
        ax[row_index].set_xlabel("Hz")
        ax[row_index].set_title(ch_names[row_index])
        ax[row_index].legend()
    plt.show()
    print('oko')
    # # return x_train.reshape((x_train.shape[0], np.prod(x_train.shape[1:])))
    # res_p = parafac(x_train, rank=3)
    # res = cp_to_tensor(res_p)
    # #
    # tlviz.visualisation.core_element_heatmap(res_p, x_train)
    # plt.show()
    # # for i in range(len(res[0][0])):
    # #     a = []
    # #     for o in res:
    # #         a.append(o[0][i])
    # #     u = rfft(a)
    # #     u = np.abs(u)
    # #     plt.subplot(4, 4, i+1)
    # #     plt.plot(range(len(u[2:])), u[2:])
    # #
    # #
    # # plt.show()
    # # input()
    # return res.reshape(res.shape[0], np.prod(res.shape[1:]))


subject = Subject('preprocessed_subjects/s14.edf')
# subject = Subject('data/data_s/2023-02-23T11-10-20_real_text.edf')
process(subject, [(2, 24)], ['C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6'])
