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

channel_index_to_name = {
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
    31: 'POZ'
}

existing_channels = {
}


class GtecNautilusProInterface:
    def __init__(self):
        self.configurations = Configurations()
        self.indicator_frequency_min = self.configurations.read("collect_data.indicator_frequency_min")
        self.indicator_frequency_max = self.configurations.read("collect_data.indicator_frequency_max")
        self.indicator_channel = self.configurations.read("collect_data.indicator_channel")
        self.sampling_rate = self.configurations.read("general.sampling_rate")
        self.selected_channels = self.configurations.read('general.selected_electrodes')

    def run_acquisition(self, prompt_viewer, current_queue, update_experiment_timeline_plot, progressbar_value, batches_per_second, real_time_processor=None):
        global current_trial_remaining_length
        global current_label
        global trial_order
        global signal
        global current_length_in_seconds
        global annotations
        global last
        global existing_channels

        d = pygds.GDS()
        pygds.configure_demo(d)
        supported_sensitivities = d.GetSupportedSensitivities()
        sensitivity_id = 0  # [[2250000.0, 1125000.0, 750000.0, 562500.0, 375000.0, 187500.0]]
        valid_channel_count = 0
        for ch_idx, ch in enumerate(d.Channels):
            ch.Sensitivity = supported_sensitivities[0][sensitivity_id]
            ch.BandpassFilterIndex = 16  # 2-30Hz bandpass
            if ch_idx in channel_index_to_name and channel_index_to_name[ch_idx] in self.selected_channels:
                ch.Enabled = 1
                existing_channels[ch_idx] = valid_channel_count
                valid_channel_count += 1
            else:
                ch.Enabled = 0
                existing_channels[ch_idx] = -1
        d.SetConfiguration()

        signal = []
        for _ in range(32):
            signal.append([])
            # signal.append(np.zeros(125))

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


                for channel in range(32):
                    signal_index = existing_channels[channel]
                    if signal_index != -1:
                        signal[channel] = np.concatenate((signal[channel], list(samples[:, signal_index])))
                    else:
                        signal[channel] = np.concatenate((signal[channel], list(np.zeros(len(samples[:, 0])))))

                # Podglad aktywnosci kanalow:
                np.set_printoptions(suppress=True, linewidth=10000, precision=2)
                # selected_channel = signal[self.indicator_channel][-self.sampling_rate:]
                # frequency_values = rfft(selected_channel)
                # frequency_values = np.abs(frequency_values)
                # indicator_value = min(frequency_values[self.indicator_frequency_min:self.indicator_frequency_max])
                # progressbar_value.set(indicator_value/100)
                # print(np.std(samples[:, [5, 15, 14, 13, 23, 9, 17, 18, 19, 27, 16]],
                #              axis=0))  # FC3, C1, C3, C5, CP3, FC4, C2, C4, C6, CP4, CZ

                if current_queue is None or len(current_queue) == 0:
                    return False
                item = current_queue[0]
                
                if real_time_processor is not None:
                    prompt_text = real_time_processor(signal)
                    if prompt_text is not None:
                        item[0] = prompt_text
                        
                print('GTEC', datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3], len(signal[0]), current_queue)
                if not prompt_viewer.closed:
                    prompt_viewer.change_prompt(item[0])
                    # time.sleep(.5) # TUTAJ JEST PROBLEM
                    item[1] -= .1#1.0/batches_per_second
                    item[1] = round(item[1], 1)
                    if item[1] < .1 and current_queue is not None:
                        current_queue.pop(0)
                    update_experiment_timeline_plot(.1)
                else:
                    return False

                return True
            except Exception as e:
                print('ERROR:', e)
                print(traceback.format_exc())

        start_date = datetime.now()
        d.GetData(d.SamplingRate // batches_per_second, processCallback)
        d.Close()
        for i in range(len(signal)):
            print(i)
            # signal[i][2500:2625] = 999
            # signal[i] = signal[i][:-125]

        del d

        return signal, start_date
