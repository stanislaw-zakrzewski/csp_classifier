eeg_data_path = 'C:/Users/stz/Documents/Data/eeg_data'
preprocessed_data_path = 'preprocessed_data/subjects'
preprocess_data = True
cache_preprocessed = False
use_common_average_reference = False
electrode_names = ['FP1', 'FP2', 'FZ', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ',
                   'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1',
                   'CZ', 'C2', 'C4', 'C6', ' T8', 'TP7', 'CP5', 'CP3',
                   'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'PZ', 'POZ']
sampling_frequency = 250

signal_configurations = [
    {'label': 'movement', 'id': 0},
    {'label': 'rest', 'id': 1}
]
