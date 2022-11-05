from demo.collect_data import collect_data
from demo.train_classifier import train_classifier
from config import configurations, experiment_frequency_range

movement_data_npy, rest_data_npy = collect_data()

band0 = 10
band1 = 14
csp, lda, mne_info = train_classifier(
    [experiment_frequency_range],
    configurations[0]['channels'],
    configurations[0]['randomness']
)