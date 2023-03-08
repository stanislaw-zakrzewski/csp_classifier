import numpy as np
import pygds
import random
import time

from datetime import datetime
from pyedflib import highlevel

from commands.get_commands import get_commands
from config_old import batches_per_second, trial_count, signal_configurations, trial_timeout_in_seconds, \
    trial_length_random_addition_in_seconds, trial_length_in_seconds, trial_timeout_random_addition_in_seconds, \
    electrode_names, sampling_frequency

commands = get_commands()
signal = []
trial_order = []
current_label = -1
current_trial_remaining_length = 0
current_length_in_seconds = 0
annotations = []
last = 0

#commands.perform_command('end')
#import sys
#sys.exit(0)

def collect_data():
    global current_trial_remaining_length
    global current_label
    global trial_order
    global signal
    global current_length_in_seconds
    global annotations
    global last
    global commands

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

    instructions_dict = {-1: 'pause'}
    for signal_configuration in signal_configurations:
        instructions_dict[signal_configuration['id']] = signal_configuration['label']

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
            global last
            global commands
            dt = datetime.now()
            last = dt

            for channel in range(32):
                signal[channel] = np.concatenate((signal[channel], list(samples[:, channel])))

            # Podglad aktywnosci kanalow:
            np.set_printoptions(suppress=True, linewidth=10000, precision=2)
            # print(np.std(samples, axis=0)) # wszystkie kanały
            #print(np.std(samples[:, [32, 33, 34]], axis=0)) # akcelerometry - dla kontroli ;-)
            # print(np.std(samples[:, [5, 15, 14, 13, 23, 9, 17, 18, 19, 27, 16]], axis=0)) # FC3, C1, C3, C5, CP3, FC4, C2, C4, C6, CP4, CZ

            current_trial_remaining_length -= 1
            current_length_in_seconds += 1 / batches_per_second

            if current_trial_remaining_length == 0:
                if current_label == -1:
                    current_label = trial_order.pop(0)

                    current_trial_remaining_length = \
                        np.random.randint(
                            trial_length_random_addition_in_seconds * batches_per_second + 1) + trial_length_in_seconds * batches_per_second
                    annotations.append(
                        [current_length_in_seconds, current_trial_remaining_length / 2,
                         instructions_dict[current_label]])
                else:
                    if len(trial_order) == 0:
                        return False
                    current_label = -1
                    current_trial_remaining_length = \
                        np.random.randint(
                            trial_timeout_random_addition_in_seconds * batches_per_second + 1) + trial_timeout_in_seconds * batches_per_second

            commands.perform_command(instructions_dict[current_label])

            return True
        except Exception as e:
            print('ERROR:', e)

    last = datetime.now()
    all = datetime.now()
    start_date = datetime.now()
    d.GetData(d.SamplingRate // batches_per_second, processCallback)
    d.Close()

    all = datetime.now() - all
    print(current_length_in_seconds, all)
    del d

    t = time.localtime()
    timestamp = time.strftime('%Y-%m-%dT%H-%M-%S', t)
    filename = 'data/{}.edf'.format(timestamp)

    sig_headers = highlevel.make_signal_headers(electrode_names, sample_rate=sampling_frequency, physical_max=2000000,
                                                physical_min=-2000000)

    header = highlevel.make_header(patientname='patient_x', gender='Male', startdate=start_date)
    header.update({'annotations': annotations})

    # commands = get_commands()
    commands.perform_command('end')
    # time.sleep(4)
    # input("Press any key")

    highlevel.write_edf(filename, signal, sig_headers, header)

    signal, signalheaders, header = highlevel.read_edf(filename)
    annot = header['annotations']
    print(annot)

time.sleep(30)
collect_data()
