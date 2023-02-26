from mne import create_info, events_from_annotations
from mne.io import read_raw_edf

from config import electrode_names, sampling_frequency, event_beginning_offset_in_seconds, \
    event_length_in_seconds, trial_length_in_seconds


class Subject:
    """Subject class used to contain data for single subject

    Parameters
    ----------
    subject_edf_path : string
        Path to EDF file that contains EEG recording for given subject.
    """
    def __init__(self, subject_edf_path):
        self.raw = read_raw_edf(subject_edf_path, preload=True, verbose='ERROR')
        raw_events, self.id_dict = events_from_annotations(self.raw, verbose='ERROR')
        self.events = []
        for raw_event in raw_events:
            event_start = event_beginning_offset_in_seconds * sampling_frequency
            event_offset = event_length_in_seconds * sampling_frequency
            event_end = trial_length_in_seconds * sampling_frequency
            while event_start + event_offset <= event_end:
                self.events.append([raw_event[0] + event_start, raw_event[1], raw_event[2]])
                event_start += event_offset

        self.electrode_names = electrode_names

        mne_info = create_info(self.electrode_names, sampling_frequency, 'eeg')
        self.info = mne_info

    def get_raw_copy(self):
        """Return a copy of a raw data for subject."""
        return self.raw.copy()
