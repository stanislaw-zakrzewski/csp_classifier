from config.config import Configurations
import numpy as np
from src.preprocessing.bandpass_filter import bandpass_filter
from mne import Epochs, pick_types
from src.feature_extractors.CSP import CSP
from src.classifiers.LDA import LDA


class MultiCSP:
    def __init__(self):
        self.configurations = Configurations()
        self.selected_channels = self.configurations.read('general.selected_electrodes')
        self.bands = [(5, 10), (10, 15), (15, 20), (20, 25)]
        self.csp_filter_bank = []
        for _ in self.bands:
            self.csp_filter_bank.append(CSP(selected_channels=self.selected_channels, reg=None))

    def fit(self, raw_signal, train_idx, test_idx):

        picks = pick_types(raw_signal.info, meg=False, eeg=True, stim=False, eog=False,
                           exclude='bads')

        x_train_csp_data = []
        x_test_csp_data = []
        for index, csp in enumerate(self.csp_filter_bank):
            band = self.bands[index]

            filtered_raw_signal = bandpass_filter(raw_signal, band[0], band[1])

            epochs = Epochs(filtered_raw_signal, subject.events, subject.id_dict, tmin, tmax, proj=True, picks=picks,
                            baseline=None, preload=True, verbose=verbose)
            epochs_train = epochs.copy()  # .crop(tmin=tmin, tmax=tmax))

            epochs_data_train = epochs_train.get_data(copy=False)
            labels = np.array(epochs.events[:, -1])

            y_train, y_test = labels[train_idx], labels[test_idx]

            x_train_csp = csp.fit_transform(epochs_data_train[train_idx], y_train)
            x_test_csp = csp.transform(epochs_data_train[test_idx])

        classifier = LDA()
        classifier.fit(x_train_csp, y_train)

        predictions = classifier.predict(x_test_csp)

        return predictions, y_test

    def predict(self, x):
        pass

    def predict_proba(self, x):
        pass
