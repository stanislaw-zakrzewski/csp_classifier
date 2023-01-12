""" GLOBAL DATA"""
preprocessed_data_path = 'preprocessed_data/subjects'
use_common_average_reference = False  # this is not working at this time
electrode_names = ['FP1', 'FP2', 'FZ', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ',
                   'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1',
                   'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3',
                   'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'PZ', 'POZ']
sampling_frequency = 250
valid_channel_threshold = 1e-15

''' DATA ACQUISITION '''
batches_per_second = 2
trial_count = 4
trial_timeout_in_seconds = 3
trial_timeout_random_addition_in_seconds = 1
trial_length_in_seconds = 10
trial_length_random_addition_in_seconds = 1
signal_configurations = [
    {'label': 'movement', 'id': 0},
    {'label': 'rest', 'id': 1}
]

''' DATA ANALYSIS '''
subject_to_analyze = 'data/2022-11-05T15-11-01.edf'
# We don't want (at least yet) to have P300 in our data
event_beginning_offset_in_seconds = 1
event_length_in_seconds = 4
experiment_frequency_range = (2, 20)

channels1 = ['C3', 'C4']
channels2 = ['C1', 'C2', 'C5', 'C3', 'C4', 'C6', 'FC3', 'CP3', 'FC4', 'CP4', 'CZ']
channels3 = []  # All channels
configurations = [
    {'channels': channels2, 'band_width': 1},
    {'channels': channels2, 'band_width': 2},
    {'channels': channels2, 'band_width': 3},
    {'channels': channels2, 'band_width': 4},
    {'channels': channels2, 'band_width': 5},
    {'channels': channels2, 'band_width': 6},
    {'channels': channels2, 'band_width': 7},
]

''' REAL TIME '''
real_time_train_data = 'data/2023-01-12T12-03-50.edf'
bandpass_filter_start_frequency = 14
bandpass_filter_end_frequency = 20

''' VISUALIZATION '''
accuracy_over_bands_show_standard_deviation = False
subject_to_visualize = 'data/2022-11-05T15-11-01.edf'

'''VR'''

# EEGLAB:
# EEGlaptop: 192.168.50.100
# Gogle VR 1: 192.168.50.101
# Gogle VR 2: 192.168.50.102

# ipaddress = '172.20.10.4'
# port = 25002
# ipaddress = '172.20.10.4'
ipaddress = '192.168.150.6'
port = 25002

# send_to_vr = True
send_to_vr = False
