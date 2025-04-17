

import pygds
from datetime import datetime
import numpy as np
import time
from scipy.fft import rfft
import traceback

from config.config import Configurations

signal = []
trial_order = []
current_label = -1
current_trial_remaining_length = 0
current_length_in_seconds = 0
annotations = []
last = 0

channel_dictionary = {
    0: 'FP1',
    1: 'FP2',
    2: 'FZ',
    3: 'FT7',
    4: 'FC5',
    5: 'FC3',
    6: 'FC1',
    7: 'FCZ',
    8: 'FC2',
    9: 'FC4',
    10: 'FC6',
    11: 'FT8',
    12: 'T7',
    13: 'C5',
    14: 'C3',
    15: 'C1',
    16: 'CZ',
    17: 'C2',
    18: 'C4',
    19: 'C6',
    20: 'T8',
    21: 'TP7',
    22: 'CP5',
    23: 'CP3',
    24: 'CP1',
    25: 'CPZ',
    26: 'CP2',
    27: 'CP4',
    28: 'CP6',
    29: 'TP8',
    30: 'PZ',
    31: 'POZ',
}



class GtecNautilusProInterface:
    def __init__(self):
        self.configurations = Configurations()
        self.indicator_frequency_min = self.configurations.read("collect_data.indicator_frequency_min")
        self.indicator_frequency_max = self.configurations.read("collect_data.indicator_frequency_max")
        self.indicator_channel = self.configurations.read("collect_data.indicator_channel")
        self.sampling_rate = self.configurations.read("general.sampling_rate")
        self.selected_electrodes = self.configurations.read("general.selected_electrodes")

    def run_acquisition(self, prompt_viewer, current_queue, update_experiment_timeline_plot, progressbar_value):
        global current_trial_remaining_length
        global current_label
        global trial_order
        global signal
        global current_length_in_seconds
        global annotations
        global last

        d = pygds.GDS()
        a = d.GetBandpassFilters()
        pygds.configure_demo(d)
        supported_sensitivities = d.GetSupportedSensitivities()
        sensitivity_id = 0  # [[2250000.0, 1125000.0, 750000.0, 562500.0, 375000.0, 187500.0]]
        channels_in_order = []
        for ch_idx, ch in enumerate(d.Channels):
            ch.Sensitivity = supported_sensitivities[0][sensitivity_id]
            if ch_idx not in channel_dictionary or channel_dictionary[ch_idx] not in self.selected_electrodes:
                # Operacje na nieaktywnych kanalach
                ch.Enabled = 0
            if ch_idx in channel_dictionary and channel_dictionary[ch_idx] in self.selected_electrodes:
                channels_in_order.append(channel_dictionary[ch_idx])
                # Operacje na aktywnych kanalach
                ch.Enabled = 1
            ch.BandpassFilterIndex = 17  # 2-30Hz bandpass
        d.SetConfiguration()

        batches_per_second = 2

        signal = []
        for _ in range(len(channels_in_order)):
            signal.append([])

        def processCallback(samples):
            try:
                global current_trial_remaining_length
                global current_label
                global trial_order
                global signal
                global current_length_in_seconds
                global annotations
                global last
                dt = datetime.now()
                last = dt

                for channel in range(len(channels_in_order)):
                    signal[channel] = np.concatenate((signal[channel], list(samples[:, channel])))

                # Podglad aktywnosci kanalow:
                np.set_printoptions(suppress=True, linewidth=10000, precision=2)
                # selected_channel = signal[self.indicator_channel][-self.sampling_rate:]
                # #selected_channel2 = signal[18][-self.sampling_rate:]
                # frequency_values = rfft(selected_channel)
                # #frequency_values2 = rfft(selected_channel2)
                # frequency_values = np.abs(frequency_values)
                # #frequency_values2 = np.abs(frequency_values2)
                # indicator_value = min(frequency_values[self.indicator_frequency_min:self.indicator_frequency_max])
                # #indicator_value2 = min(frequency_values2[self.indicator_frequency_min:self.indicator_frequency_max])
                # progressbar_value.set(indicator_value/100)
                # #progressbar_value.set((indicator_value + indicator_value2)/200)
                # print(np.std(samples[:, [5, 15, 14, 13, 23, 9, 17, 18, 19, 27, 16]],
                #              axis=0))  # FC3, C1, C3, C5, CP3, FC4, C2, C4, C6, CP4, CZ

                if current_queue is None or len(current_queue) == 0:
                    return False
                item = current_queue[0]
                if not prompt_viewer.closed:
                    prompt_viewer.change_prompt(item[0])
                    time.sleep(.5)
                    item[1] -= .5
                    if item[1] < .5 and current_queue is not None:
                        current_queue.pop(0)
                    update_experiment_timeline_plot(.5)
                else:
                    return False

                return True
            except Exception:
                print(traceback.format_exc())

        start_date = datetime.now()
        d.GetData(d.SamplingRate // batches_per_second, processCallback)
        d.Close()

        del d

        return signal, start_date, channels_in_order
