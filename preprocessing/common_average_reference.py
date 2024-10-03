import numpy as np


def car(raw_signal_data):
    averaged_signal_over_electrodes = np.sum(raw_signal_data, axis=0) / raw_signal_data.shape[0]
    for channel in raw_signal_data:
        channel -= averaged_signal_over_electrodes
    return raw_signal_data


def common_average_reference(raw_signal):
    raw_signal.apply_function(car, channel_wise=False)
