import pygds
from datetime import datetime
import numpy as np
import time
from pyedflib import highlevel

from config.config import Configurations


class GtecNautilusProInterface:
    def __init__(self):
        self.configurations = Configurations()

    def run_acquisition(self):
        global current_trial_remaining_length
        global current_label
        global trial_order
        global signal
        global current_length_in_seconds
        global annotations
        global last
        global commands

        d = pygds.GDS()
        pygds.configure_demo(d)
        supported_sensitivities = d.GetSupportedSensitivities()
        sensitivity_id = 0  # [[2250000.0, 1125000.0, 750000.0, 562500.0, 375000.0, 187500.0]]
        for ch in d.Channels:
            ch.Sensitivity = supported_sensitivities[0][sensitivity_id]
            ch.BandpassFilterIndex = 16  # 2-30Hz bandpass
        d.SetConfiguration()

        batches_per_second = 2
        trial_length_random_addition_in_seconds = 0
        instructions_dict = {-1: 'pause', 0: 'rest', 1: 'movement'}
        electrode_names = self.configurations.read('all.general.all_electrodes')
        sampling_frequency = self.configurations.read('all.general.sampling_rate')

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
                global commands
                dt = datetime.now()
                last = dt

                for channel in range(32):
                    signal[channel] = np.concatenate((signal[channel], list(samples[:, channel])))

                # Podglad aktywnosci kanalow:
                np.set_printoptions(suppress=True, linewidth=10000, precision=2)
                # print(np.std(samples, axis=0)) # wszystkie kanały
                # print(np.std(samples[:, [32, 33, 34]], axis=0)) # akcelerometry - dla kontroli ;-)
                print(np.std(samples[:, [5, 15, 14, 13, 23, 9, 17, 18, 19, 27, 16]],
                             axis=0))  # FC3, C1, C3, C5, CP3, FC4, C2, C4, C6, CP4, CZ

                if self.current_queue is None or len(self.current_queue) == 0:
                    return False
                item = self.current_queue[0]
                if not self.prompt_viewer.closed:
                    self.prompt_viewer.change_prompt(item[0])
                    time.sleep(.5)
                    item[1] -= .5
                    if item[1] < .5 and self.current_queue is not None:
                        self.current_queue.pop(0)
                    self.update_experiment_timeline_plot()
                else:
                    return False

                # current_trial_remaining_length -= 1
                # current_length_in_seconds += 1 / batches_per_second
                #
                # if current_trial_remaining_length == 0:
                #     if current_label == -1:
                #         current_label = trial_order.pop(0)
                #
                #         current_trial_remaining_length = \
                #             np.random.randint(
                #                 trial_length_random_addition_in_seconds * batches_per_second + 1) + trial_length_in_seconds * batches_per_second
                #         annotations.append(
                #             [current_length_in_seconds, current_trial_remaining_length / 2,
                #              instructions_dict[current_label]])
                #     else:
                #         if len(trial_order) == 0:
                #             return False
                #         current_label = -1
                #         current_trial_remaining_length = \
                #             np.random.randint(
                #                 trial_timeout_random_addition_in_seconds * batches_per_second + 1) + trial_timeout_in_seconds * batches_per_second
                #
                # commands.perform_command(instructions_dict[current_label])

                return True
            except Exception as e:
                print('ERROR:', e)

        # while self.current_queue is not None and len(self.current_queue) > 0:
        #     item = self.current_queue[0]
        #     if not self.prompt_viewer.closed:
        #         self.prompt_viewer.change_prompt(item[0])
        #         time.sleep(.5)
        #         item[1] -= .5
        #         if item[1] < .5 and self.current_queue is not None:
        #             self.current_queue.pop(0)
        #         self.update_experiment_timeline_plot()
        #     else:
        #         break
        last = datetime.now()
        all = datetime.now()
        start_date = datetime.now()
        d.GetData(d.SamplingRate // batches_per_second, processCallback)
        d.Close()

        del d
        a = signal  # SYGNAŁ
        # plt.plot(a[16], label = "CZ")
        # plt.plot(a[5], label = "FC3")
        # plt.plot(a[13], label = "C5")
        # plt.plot(a[14], label = "C3")
        # plt.plot(a[15], label = "C1")
        # plt.plot(a[23], label = "CP3")
        # plt.plot(a[9], label = "FC4")
        # plt.plot(a[17], label="C2")
        # plt.plot(a[18], label="C4")
        # plt.plot(a[19], label="C6")
        # plt.plot(a[27], label="CP4")
        # plt.legend()
        # plt.show()
        t = time.localtime()
        timestamp = time.strftime('%Y-%m-%dT%H-%M-%S', t)
        filename = 'data/{}.edf'.format(timestamp)

        sig_headers = highlevel.make_signal_headers(electrode_names, sample_rate=sampling_frequency,
                                                    physical_max=1000.0,
                                                    physical_min=-1000.0)

        annotations = []
        len_for_annot = 0
        for index, queue_element in enumerate(self.queue):
            if queue_element[0] != 'break':
                annotations.append([len_for_annot, queue_element[1], queue_element[0]])
            len_for_annot += queue_element[1]

        header = highlevel.make_header(patientname=self.patient_name_value.get(), gender=self.gender_value.get(),
                                       startdate=start_date)
        header.update({'annotations': annotations})

        if not self.prompt_viewer.closed:
            self.prompt_viewer.change_prompt('end')
        else:
            self.prompt_viewer.destroy()
        print(sig_headers)
        highlevel.write_edf(filename, signal, sig_headers, header)
