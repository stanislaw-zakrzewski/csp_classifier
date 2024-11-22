from config.config import Configurations


def bandpass_filter(signal, l_frequency, h_frequency, sampling_frequency=None):
    """Band pass filtering for MNE's raw signal.

    Parameters
    ----------
    signal : mne.io.Raw
        EEG signal in MNE's Raw format.
    l_frequency : float
        Highpass frequency for bandpass filter.
    h_frequency : float
        Lowpass frequency for bandpass filter.
    sampling_frequency : int
        Sampling frequency, leaving empty defaults to configuration
    """
    configurations = Configurations()
    if sampling_frequency is None:
        sampling_frequency = configurations.read('general.sampling_rate')
    verbose = configurations.read('general.verbose')
    l_trans_bandwidth = min(2, l_frequency)

    return signal.filter(l_frequency, h_frequency, l_trans_bandwidth=l_trans_bandwidth, h_trans_bandwidth=2,
                         filter_length=sampling_frequency * 4,
                         fir_design='firwin',
                         skip_by_annotation='edge', verbose=verbose)
