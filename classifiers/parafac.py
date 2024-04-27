import mne
import numpy as np
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
from tensorly.decomposition import parafac
from tensorly import unfold, cp_to_tensor
from scipy.fft import fft, fftfreq, rfft, rfftfreq
from sklearn.decomposition import TruncatedSVD
from scipy.signal import morlet
import tlviz


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


def get_atoms(x_train, ch_names=[]):
    # for x_1 in x_train:
    #     for x_2 in x_1:
    freq = np.fft.rfftfreq(500, d=1. / 250)[0:50]

    weights, factors = parafac(x_train, rank=5)
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
