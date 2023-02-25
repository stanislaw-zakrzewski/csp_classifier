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
from config import channels2
from data_classes.subject import Subject
from tensorly.decomposition import parafac
from tensorly import unfold, cp_to_tensor
from scipy.fft import fft, fftfreq, rfft, rfftfreq
from sklearn.decomposition import TruncatedSVD
from scipy.signal import morlet


def process(subject, bands, selected_channels, n_splits=10, reg=None, verbose='DEBUG'):
    tmin, tmax = 1., 3.
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
            raw_signals[index].filter(band[0], band[1], l_trans_bandwidth=2, h_trans_bandwidth=2, filter_length=500,
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
    classifier = LinearDiscriminantAnalysis()
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
                x_train_csp = get_atoms(edt[train_idx])
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

    w_times = (w_start + w_length / 2.) / sfreq + epochs[0].tmin
    return w_times, scores_windows, svd, epochs[0].info, all_predictions, all_correct, classifier, subject.info


def time_frequency_analysis(data):
    train_data = []
    for X in data:
        train_data.append(cwt_morlet(X, 512, freqs=np.arange(5, 30, 1), use_fft=True, n_cycles=5))
    return np.asarray(train_data)


def get_atoms(x_train):
    # return x_train.reshape((x_train.shape[0], np.prod(x_train.shape[1:])))
    res = parafac(x_train, rank=10)
    res = cp_to_tensor(res)
    return res.reshape(res.shape[0], np.prod(res.shape[1:]))