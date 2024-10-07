import pygds
from datetime import datetime
import numpy as np
import time
from scipy.fft import rfft

from config.config import Configurations

signal = []
trial_order = []
current_label = -1
current_trial_remaining_length = 0
current_length_in_seconds = 0
annotations = []
last = 0


class GtecNautilusProInterface:
    def __init__(self):
        self.configurations = Configurations()
        self.indicator_frequency_min = self.configurations.read("collect_data.indicator_frequency_min")
        self.indicator_frequency_max = self.configurations.read("collect_data.indicator_frequency_max")
        self.indicator_channel = self.configurations.read("collect_data.indicator_channel")
        self.sampling_rate = self.configurations.read("general.sampling_rate")

    def run_acquisition(self, prompt_viewer, current_queue, update_experiment_timeline_plot, progressbar_value):
        global current_trial_remaining_length
        global current_label
        global trial_order
        global signal
        global current_length_in_seconds
        global annotations
        global last

        d = pygds.GDS()
        pygds.configure_demo(d)
        supported_sensitivities = d.GetSupportedSensitivities()
        sensitivity_id = 0  # [[2250000.0, 1125000.0, 750000.0, 562500.0, 375000.0, 187500.0]]
        for ch in d.Channels:
            ch.Sensitivity = supported_sensitivities[0][sensitivity_id]
            ch.BandpassFilterIndex = 16  # 2-30Hz bandpass
        d.SetConfiguration()

        batches_per_second = 2

        signal = []
        for _ in range(32):
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

                for channel in range(32):
                    signal[channel] = np.concatenate((signal[channel], list(samples[:, channel])))

                # Podglad aktywnosci kanalow:
                np.set_printoptions(suppress=True, linewidth=10000, precision=2)
                selected_channel = signal[self.indicator_channel][-self.sampling_rate:]
                frequency_values = rfft(selected_channel)
                frequency_values = np.abs(frequency_values)
                indicator_value = min(frequency_values[self.indicator_frequency_min:self.indicator_frequency_max])
                progressbar_value.set(indicator_value/100)
                print(np.std(samples[:, [5, 15, 14, 13, 23, 9, 17, 18, 19, 27, 16]],
                             axis=0))  # FC3, C1, C3, C5, CP3, FC4, C2, C4, C6, CP4, CZ

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
            except Exception as e:
                print('ERROR:', e)

        start_date = datetime.now()
        d.GetData(d.SamplingRate // batches_per_second, processCallback)
        d.Close()

        del d

        return signal, start_date
