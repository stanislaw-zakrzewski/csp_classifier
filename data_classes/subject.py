import numpy as np
from mne import create_info
from mne.io import concatenate_raws, RawArray

from config import cache_preprocessed, preprocess_data, electrode_names, sampling_frequency, signal_configurations
from preprocessing.preprocess_subject import preprocess_subject


class Subject:
    def __init__(self, randomness=0):
        signals = []
        for signal_configuration in signal_configurations:
            signals.append({
                'label': signal_configuration['label'],
                'id': signal_configuration['id'],
                'data': np.load('data/{}.npy'.format(signal_configuration['label']), allow_pickle=True)})

        if cache_preprocessed:
            try:
                self.load_preprocessed_signals(signals)
            except FileNotFoundError:
                self.generate_seamless_signals(signals)

                if preprocess_data:
                    preprocess_subject(signals)

                self.save_preprocessed_signals(signals)
        else:
            self.generate_seamless_signals(signals)
            if preprocess_data:
                preprocess_subject(signals)

        self.electrode_names = electrode_names
        self.events = self.generate_events(signals, randomness)

        mne_info = create_info(self.electrode_names, sampling_frequency, 'eeg')

        raws = list(map(lambda signal: RawArray(signal['seamless'], mne_info), signals))

        # TODO setup montage locations for raws

        self.raw = concatenate_raws(raws)

    def get_raw_copy(self):
        return self.raw.copy()

    @staticmethod
    def load_preprocessed_signals(signals):
        for signal in signals:
            signal['seamless'] = np.load('preprocessed_data_npy/{}.npy'.format(signal['label']))

    @staticmethod
    def save_preprocessed_signals(signals):
        for signal in signals:
            np.save('preprocessed_data_npy/{}.npy'.format(signal['label']), signal['seamless'])

    @staticmethod
    def generate_seamless_signals(signals):
        for signal in signals:
            trials = signal['data']
            trials_seamless = list(trials[0])
            for i in range(1, len(trials)):
                for j in range(len(trials_seamless)):
                    trials_seamless[j] = np.concatenate((trials_seamless[j], list(trials[i][j])))
            signal['seamless'] = np.array(trials_seamless)

    @staticmethod
    def get_seamless(trials):
        trials_seamless = list(trials[0])
        for i in range(1, len(trials)):
            for j in range(len(trials_seamless)):
                trials_seamless[j] = np.concatenate((trials_seamless[j], list(trials[i][j])))
        return np.array(trials_seamless)

    @staticmethod
    def random_trial(source, randomness):
        if np.random.rand() < randomness:
            return np.random.randint(0, 2)
        return source

    def generate_events(self, signals, randomness):
        data = []
        position = 0
        for signal in signals:
            for trial in signal['data']:
                if len(trial[0]) >= 750:
                    data.append([position + 250, 0, self.random_trial(signal['id'], randomness)])
                if len(trial[0]) >= 1250:
                    data.append([position + 750, 0, self.random_trial(signal['id'], randomness)])
                position += len(trial[0])
            position = len(signal['seamless'][0])
        return data
