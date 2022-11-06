eeg_data_path = 'C:/Users/stz/Documents/Data/eeg_data'
preprocessed_data_path = 'preprocessed_data/subjects'
preprocess_data = True
cache_preprocessed = False
use_common_average_reference = False
electrode_names = ['FP1', 'FP2', 'FZ', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ',
                   'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1',
                   'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3',
                   'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'PZ', 'POZ']
sampling_frequency = 250

''' DATA ACQUISITION '''
batches_per_second = 2
trial_count = 15
trial_timeout_in_seconds = 3
trial_timeout_random_addition_in_seconds = 1
trial_length_in_seconds = 5
trial_length_random_addition_in_seconds = 1
signal_configurations = [
    {'label': 'movement', 'id': 0},
    {'label': 'rest', 'id': 1}
]

''' DATA ANALYSIS '''
subject_to_analyze = 'data/2022-11-05T15-54-46.edf'
# We don't want (at least yet) to have P300 in our data
event_beginning_offset_in_seconds = 1
event_length_in_seconds = 2

channels1 = ['C3', 'C4']
channels2 = ['C5', 'C3', 'C1', 'C2', 'C4', 'C6', 'FC3', 'CP3', 'FC4', 'CP4', 'CZ']
channelsK = ['C5', 'C3', 'C4', 'C6', 'FC3', 'CP3', 'FC4', 'CP4', 'CZ']
channels3 = []
configurations = [
    {'channels': channelsK, 'band_width': 1, 'randomness': 0},
    # {'channels': channelsK, 'band_width': 2, 'randomness': 0},
    # {'channels': channelsK, 'band_width': 3, 'randomness': 0},
    # {'channels': channelsK, 'band_width': 4, 'randomness': 0},
    # {'channels': channelsK, 'band_width': 5, 'randomness': 0},
    # {'channels': channelsK, 'band_width': 6, 'randomness': 0},
    # {'channels': channelsK, 'band_width': 7, 'randomness': 0},
    # {'channels': channels3, 'band_width': 2, 'randomness': 0},
    # {'channels': channels3, 'band_width': 4, 'randomness': 0},
    # {'channels': channels2, 'band_width': 4, 'randomness': 0},
    # {'channels': channels2, 'band_width': 5, 'randomness': 0},
    # {'channels': channels2, 'band_width': 6, 'randomness': 0},
    # {'channels': channels2, 'band_width': 7, 'randomness': 0},
]

experiment_frequency_range = (2, 30)
