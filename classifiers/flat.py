import mne
import numpy as np
from mne import Epochs, pick_types

from sklearn.model_selection import ShuffleSplit
from sklearn.neural_network import MLPClassifier

from config.config import Configurations
from src.classifiers.LDA import LDA
from src.classifiers.MLP import MLP
from src.feature_extractors.CSP import CSP
from src.preprocessing.bandpass_filter import bandpass_filter
from src.preprocessing.channel_selector import select_channels


def process(subject, bands, selected_channels, n_splits=10, reg=None, verbose='DEBUG', score_window_flag=False):
    tmin, tmax = .0, subject.sub_event_length_sec

    raw_signal = subject.get_raw_copy()

    select_channels(raw_signal, selected_channels)

    filtered_raw_signal = bandpass_filter(raw_signal, bands[0][0], bands[0][1])

    picks = pick_types(filtered_raw_signal.info, meg=False, eeg=True, stim=False, eog=False,
                       exclude='bads')

    epochs = Epochs(filtered_raw_signal, subject.events, subject.id_dict, tmin, tmax, proj=True, picks=picks,
                    baseline=None, preload=True, verbose=verbose)
    epochs_train = epochs.copy()  # .crop(tmin=tmin, tmax=tmax))

    epochs_data_train = epochs_train.get_data(copy=False)
    labels = np.array(epochs.events[:, -1])

    cv = ShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=42)
    cv_split = cv.split(epochs_data_train)



    all_predictions = []
    all_correct = []

    for train_idx, test_idx in cv_split:
        classifier = LDA()
        csp = CSP(selected_channels=selected_channels, reg=reg)
        y_train, y_test = labels[train_idx], labels[test_idx]

        x_train_csp = csp.fit_transform(epochs_data_train[train_idx], y_train)
        x_test_csp = csp.transform(epochs_data_train[test_idx])

        classifier.fit(x_train_csp, y_train)

        predictions = classifier.predict(x_test_csp)
        all_predictions.append(predictions)
        all_correct.append(y_test)

    return csp, all_predictions, all_correct, classifier, subject.info
