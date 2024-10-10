from data_classes.subject import Subject
from tkinter import filedialog as fd
from pyedflib import highlevel
from mne.filter import filter_data
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def visualize_significant_heatmap(heatmap, frequencies, channels):
    heatmap_plot_data = {'Channel': [], 'Frequency': [], 'Amplitude': []}
    for channel_index, heatmap_channel in enumerate(heatmap):
        channel = channels[channel_index]
        for frequency_index, heatmap_channel_frequency in enumerate(heatmap_channel):
            heatmap_plot_data['Channel'].append(channel)
            heatmap_plot_data['Frequency'].append(frequencies[frequency_index])
            heatmap_plot_data['Amplitude'].append(heatmap_channel_frequency)
    df = pd.DataFrame(heatmap_plot_data)
    df['Channel'] = pd.Categorical(df['Channel'], categories=channels)
    df = df.sort_values('Channel')
    fig, axes = plt.subplots(ncols=2,
                             gridspec_kw=dict(width_ratios=[len(channels), 0.5]))
    sns.heatmap(df.pivot(index='Frequency', columns='Channel', values='Amplitude'), ax=axes[0], cbar=False,
                vmin=min(df['Amplitude']))
    axes[0].set_title('Heatmap Of Significance')
    axes[0].invert_yaxis()
    fig.colorbar(axes[0].collections[0], cax=axes[1])
    plt.show()


def load_data():
    # Hard loaded heatmap
    # significance_heatmap = 'parafac_analysis/significant_heatmaps/2024-10-09T20-07-32.npy'
    significance_heatmap = fd.askopenfilename(filetypes=[("Numpy files", "*.npy")])
    heatmap_data = np.load(significance_heatmap, allow_pickle=True).item()
    heatmap = np.array(heatmap_data['heatmap'])
    frequencies = heatmap_data['frequencies']
    channels = heatmap_data['channels']

    # Hard loaded subject
    # subject_path = 'preprocessed_subjects/s43.edf'
    subject_path = fd.askopenfilename(filetypes=[("European Data Format files", "*.edf")])
    subject = Subject(subject_path)

    return heatmap, frequencies, channels, subject

def main():
    heatmap, frequencies, channels, subject = load_data()
    visualize_significant_heatmap(heatmap, frequencies, channels)

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

    filtered_signals = []
    for frequency_filter in frequency_filters:
        filtered_signals.append(filter_data(subject.signals, subject.sampling_frequency, frequency_filter[0]-.5, frequency_filter[1]+.5))
    res_data = np.sum(filtered_signals, axis=0)

    highlevel.write_edf('tmp.edf', res_data, subject.signal_headers, subject.header)

main()
