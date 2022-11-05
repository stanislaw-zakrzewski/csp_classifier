import random

from matplotlib import pyplot as plt
import numpy as np
from pyedflib import highlevel

from config import batches_per_second, trial_count, signal_configurations, trial_timeout_in_seconds, \
    trial_length_random_addition_in_seconds, trial_length_in_seconds, trial_timeout_random_addition_in_seconds, \
    electrode_names, sampling_frequency

import pygds

signal = []
trial_order = []
current_label = -1
current_trial_remaining_length = 0
current_length_in_seconds = 0


def collect_data():
    global current_trial_remaining_length
    global current_label
    global trial_order
    global signal
    global current_length_in_seconds
    global annotations

    print("Initializing")
    d = pygds.GDS()
    pygds.configure_demo(d)
    d.SetConfiguration()

    print("Acquisition started")
    annotations = []
    current_label = -1
    current_trial_remaining_length = trial_timeout_in_seconds * batches_per_second

    for _ in range(32):
        signal.append([])

    dict = {-1: '---+---'}
    for signal_configuration in signal_configurations:
        dict[signal_configuration['id']] = signal_configuration['label']

    for _ in range(trial_count):
        for signal_configuration in signal_configurations:
            trial_order.append(signal_configuration['id'])
    random.shuffle(trial_order)

    def processCallback(samples):
        try:
            global current_trial_remaining_length
            global current_label
            global trial_order
            global signal
            global current_length_in_seconds
            global annotations

            for channel in range(32):
                signal[channel] = np.concatenate((signal[channel], list(samples[:, channel])))
            current_trial_remaining_length -= 1
            current_length_in_seconds += 1 / batches_per_second

            if current_trial_remaining_length == 0:
                if current_label == -1:
                    current_label = trial_order.pop(0)
                    current_trial_remaining_length = \
                        np.random.randint(
                            trial_length_random_addition_in_seconds * batches_per_second + 1) + trial_length_in_seconds * batches_per_second
                    annotations.append(
                        [current_length_in_seconds, current_trial_remaining_length / 2, dict[current_label]])
                else:
                    if len(trial_order) == 0:
                        return False
                    current_label = -1
                    current_trial_remaining_length = \
                        np.random.randint(
                            trial_timeout_random_addition_in_seconds * batches_per_second + 1) + trial_timeout_in_seconds * batches_per_second

            print(dict[current_label], current_length_in_seconds)
            return True
        except Exception as e:
            print('ERROR:', e)

    print(dict[current_label], 0)

    # loop = True
    # while loop:
    #     data = []
    #     for _ in range(250):
    #         data.append([])
    #         for i in range(32):
    #             data[-1].append(i)
    #     loop = processCallback(np.array(data))
    # print(annotations)
    
    d.GetData(d.SamplingRate // batches_per_second, processCallback)
    d.Close()
    del d

    filename = 'data/edf_file.edf'

    sig_headers = highlevel.make_signal_headers(electrode_names, sample_rate=sampling_frequency)

    header = highlevel.make_header(patientname='patient_x', gender='Male')
    header.update({'annotations': annotations})

    highlevel.write_edf(filename, signal, sig_headers, header)

    signal, signalheaders, header = highlevel.read_edf(filename)
    annot = header['annotations']
    print(annot)

    print('DATA COLLECTION FINISHED')


collect_data()
