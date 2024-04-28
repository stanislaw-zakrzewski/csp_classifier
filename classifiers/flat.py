import mne
import numpy as np
from mne import Epochs, pick_types, concatenate_epochs
from src.feature_extractors.CSP import CSP
from src.classifiers.LDA import LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit
from sklearn.neural_network import MLPClassifier

from config.config import Configurations
from src.preprocessing.bandpass_filter import bandpass_filter
from src.preprocessing.channel_selector import select_channels


def process(subject, bands, selected_channels, n_splits=10, reg=None, verbose='DEBUG', score_window_flag=False):
    tmin, tmax = .0, subject.sub_event_length_sec

    # raw_signals = []
    raw_signal = subject.get_raw_copy()
    # for i in range(len(bands)):
    #     raw_signals.append(subject.get_raw_copy())
    raw_signal = bandpass_filter(raw_signal, 3, 20)




    # Apply band-pass filter
    # for index, band in enumerate(bands):
    #     filtered_raw_signals.append(
    #         bandpass_filter(raw_signals[index], band[0], band[1]))

    picks = pick_types(raw_signal.info, meg=False, eeg=True, stim=False, eog=False,
                       exclude='bads')

    epochs = Epochs(raw_signal, subject.events, subject.id_dict, tmin, tmax, proj=True, picks=picks,
                    baseline=None, preload=True, verbose=verbose)
    epochs_train = epochs.copy()  # .crop(tmin=tmin, tmax=tmax)

    epochs_data = epochs.get_data(copy=False)
    epochs_data_train = epochs_train.get_data(copy=False)
    labels = np.array(epochs.events[:, -1])

    cv = ShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=42)
    cv_split = cv.split(epochs_data_train)

    # Assemble a classifier
    classifier = LDA()
    csp_n_components = 32 if len(selected_channels) == 0 else min(len(selected_channels), 32)
    mne.set_log_level('warning')
    csp = CSP(n_components=2, reg=reg)

    sfreq = raw_signal.info['sfreq']
    w_length = int(sfreq)  # running classifier: window length
    w_step = int(sfreq * 0.1)  # running classifier: window step size
    w_start = np.arange(0, epochs_data.shape[2] - w_length, w_step)

    scores_windows = []
    all_predictions = []
    all_correct = []

    for train_idx, test_idx in cv_split:
        y_train, y_test = labels[train_idx], labels[test_idx]

        x_train_csp = []
        x_test_csp = []
        # for edt in epochs_data_train:
        #     if len(x_train_csp) > 0:
        #         x_train_csp = np.concatenate((x_train_csp, csp.fit_transform(edt[train_idx], y_train)),
        #                                      axis=1)
        #         x_test_csp = np.concatenate((x_test_csp, csp.transform(edt[test_idx])), axis=1)
        #     else:
        x_train_csp = csp.fit_transform(epochs_data_train[train_idx], y_train)
        x_test_csp = csp.transform(epochs_data_train[test_idx])

        classifier.fit(x_train_csp, y_train)

        predictions = classifier.predict(x_test_csp)
        predictions_proba = classifier.predict_proba(x_test_csp)
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
                            (x_test_csp, csp.transform(edt[test_idx][:, :, n:(n + w_length)])),
                            axis=1)
                    else:
                        x_test_csp = csp.transform(edt[test_idx][:, :, n:(n + w_length)])
                score_this_window.append(classifier.score(x_test_csp, y_test))
            scores_windows.append(score_this_window)

    w_times = []
    if score_window_flag:
        w_times = (w_start + w_length / 2.) / sfreq + epochs[0].tmin
    return w_times, scores_windows, csp, epochs[0].info, all_predictions, all_correct, classifier, subject.info
