from data_classes.subject import Subject
from tkinter import filedialog as fd
from pyedflib import highlevel
from mne.filter import filter_data
from sklearn.cluster import KMeans
import numpy as np

significance_heatmap = 'parafac_analysis/significant_heatmaps/2024-10-09T20-07-32.npy'
# significance_heatmap = fd.askopenfilename(filetypes=[("Numpy files", "*.npy")])
heatmap_data = np.load(significance_heatmap, allow_pickle=True).item()
heatmap = np.array(heatmap_data['heatmap'])
frequencies = heatmap_data['frequencies']
channels = heatmap_data['channels']

kmeans = KMeans(2)
kmeans.fit(heatmap.flatten().reshape(-1, 1))

significant_cluster = 0
if kmeans.cluster_centers_[0] < kmeans.cluster_centers_[1]:
    significant_cluster = 1

significant_channels = []
significant_frequencies = []
frequency_filters = []
for i, ch in enumerate(heatmap):
    a = kmeans.predict(ch.reshape(-1, 1))
    if significant_cluster in a:
        significant_channels.append(channels[i])
gap = True
transformed_heatmap = np.transpose(heatmap)
for i, ch in enumerate(transformed_heatmap):
    a = kmeans.predict(ch.reshape(-1, 1))
    if significant_cluster in a:
        frequency = frequencies[i]
        if gap:
            frequency_filters.append([frequency,frequency])
        else:
            frequency_filters[-1][1] = frequency
        significant_frequencies.append(frequency)
        gap = False
    else:
        gap = True
subject_path = 'preprocessed_subjects/s43.edf'
# subject_path = fd.askopenfilename(filetypes=[("European Data Format files", "*.edf")])
subject = Subject(subject_path)


frequency_filters.append([20,24])
filtered_signals = []
for frequency_filter in frequency_filters:
    filtered_signals.append(filter_data(subject.signals, subject.sampling_frequency, frequency_filter[0]-.5, frequency_filter[1]+.5))
res_data = np.sum(filtered_signals, axis=0)
# print('filter')
# filtered_signal.filter(2, 24, l_trans_bandwidth=2, h_trans_bandwidth=2, filter_length=subject.sampling_frequency * 2,
#                        fir_design='firwin',
#                        skip_by_annotation='edge', )
# print('done')

highlevel.write_edf('tmp.edf', res_data, subject.signal_headers, subject.header)
