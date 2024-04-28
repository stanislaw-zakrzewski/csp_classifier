from config.config import Configurations


def bandpass_filter(signal, l_frequency, h_frequency):
    """Band pass filtering for MNE's raw signal.

    Parameters
    ----------
    signal : mne.io.Raw
        EEG signal in MNE's Raw format.
    l_frequency : float
        Highpass frequency for bandpass filter.
    h_frequency : float
        Lowpass frequency for bandpass filter.
    """
    configurations = Configurations()
    sampling_frequency = configurations.read('general.sampling_rate')
    verbose = configurations.read('general.verbose')

    return signal.filter(l_frequency, h_frequency, l_trans_bandwidth=2, h_trans_bandwidth=2,
                         filter_length=sampling_frequency * 2,
                         fir_design='firwin',
                         skip_by_annotation='edge', verbose=verbose)
