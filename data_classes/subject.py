import mne
from mne import create_info, events_from_annotations, read_annotations
from mne.io import read_raw_edf
import numpy as np

from config.config import Configurations


class Subject:
    """Subject class used to contain data for single subject

    Parameters
    ----------
    subject_edf_path : string
        Path to EDF file that contains EEG recording for given subject.
    """

    def __init__(self, subject_edf_path):
        self.raw = read_raw_edf(subject_edf_path, preload=True, verbose='ERROR')
        annotations = read_annotations(subject_edf_path)
        initial_events, self.id_dict = events_from_annotations(self.raw, verbose='ERROR')
        self.configurations = Configurations()
        self.sampling_frequency = self.configurations.read('general.sampling_rate')
        self.sub_event_length_sec = self.configurations.read('general.sub_event_length_sec')

        self.electrode_names = self.configurations.read('general.all_electrodes')

        events = []
        self.event_len = int(self.sub_event_length_sec * self.sampling_frequency)
        self.offset_samples = int(self.configurations.read('general.offset_seconds') * self.sampling_frequency)
        for index, initial_event in enumerate(initial_events):
            annotation_duration = int(annotations.duration[index] * self.sampling_frequency)
            current_event_start = self.offset_samples
            while current_event_start < annotation_duration + self.event_len:
                new_event = [initial_event[0] + current_event_start, 0, initial_event[2]]
                events.append(new_event)
                current_event_start += self.event_len
        self.events = np.array(events)

        mne_info = create_info(self.electrode_names, self.sampling_frequency, 'eeg')
        self.info = mne_info

    def get_raw_copy(self):
        """Return a copy of a raw data for subject."""
        return self.raw.copy()
