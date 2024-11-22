from data_classes.subject import Subject
from tkinter import filedialog as fd
from pyedflib import highlevel
from mne.filter import filter_data, notch_filter
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings


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


def load_data(heatmap_data=None, subject=None):
    # Hard loaded heatmap
    # significance_heatmap = 'parafac_analysis/significant_heatmaps/2024-10-09T20-07-32.npy'
    if heatmap_data is None:
        significance_heatmap = fd.askopenfilename(filetypes=[("Numpy files", "*.npy")])
        heatmap_data = np.load(significance_heatmap, allow_pickle=True).item()
    heatmap = np.array(heatmap_data['heatmap'])
    frequencies = heatmap_data['frequencies']
    channels = heatmap_data['channels']

    # Hard loaded subject
    # subject_path = 'preprocessed_subjects/s43.edf'
    if subject is None:
        subject_path = fd.askopenfilename(filetypes=[("European Data Format files", "*.edf")])
        subject = Subject(subject_path)

    return heatmap, frequencies, channels, subject


def main(heatmap_data=None, subject=None, inverse=False):
    heatmap, frequencies, channels, subject = load_data(heatmap_data, subject)
    if heatmap_data == None:
        visualize_significant_heatmap(heatmap, frequencies, channels)

    t_min = heatmap_data['t_min']
    t_max = heatmap_data['t_max']
    delta_t = t_max - t_min
    filter_length = int(delta_t * 4 * subject.sampling_frequency)
    trans_bandwidth = 2 / delta_t

    cluster_count = 2
    kmeans = KMeans(n_init=cluster_count)
    kmeans.fit(heatmap.flatten().reshape(-1, 1))

    significant_cluster = 0
    for cluster_id, cluster_center in enumerate(kmeans.cluster_centers_):
        if inverse:
            if kmeans.cluster_centers_[significant_cluster] > cluster_center:
                significant_cluster = cluster_id
        else:
            if kmeans.cluster_centers_[significant_cluster] < cluster_center:
                significant_cluster = cluster_id

    significant_channels = []
    significant_frequencies = []

    band_stop_filtered_signals = []
    for i, ch in enumerate(heatmap):
        cluster_belonging = kmeans.predict(ch.reshape(-1, 1))
        if significant_cluster in cluster_belonging:
            significant_channels.append(channels[i])

        frequency_filters = []
        gap = True
        for a_idx, a_val in enumerate(cluster_belonging):
            if significant_cluster == a_val:
                frequency = frequencies[a_idx]
                if gap:
                    frequency_filters.append([frequency, frequency])
                else:
                    frequency_filters[-1][1] = frequency
                significant_frequencies.append(frequency)
                gap = False
            else:
                gap = True
        if len(frequency_filters) == 0:
            band_stop_filtered_signals.append(np.zeros(len(subject.signals[i])))
        else:
            try:
                band_stop_filtered_signals.append(filter_data(subject.signals[i], subject.sampling_frequency,
                                                              max(1, frequency_filters[0][0]),
                                                              frequency_filters[-1][1],
                                                              filter_length=filter_length,
                                                              l_trans_bandwidth=trans_bandwidth,
                                                              h_trans_bandwidth=trans_bandwidth, verbose='CRITICAL'))
            except:
                band_stop_filtered_signals.append(notch_filter(subject.signals[i], subject.sampling_frequency,
                                                               (frequency_filters[0][0] + frequency_filters[-1][1]) / 2,
                                                               verbose='CRITICAL'))

            for ff_idx in range(len(frequency_filters) - 1):
                l_pass = frequency_filters[ff_idx][1]
                h_pass = frequency_filters[ff_idx + 1][0]
                if l_pass < h_pass:
                    try:
                        band_stop_filtered_signals[i] = filter_data(band_stop_filtered_signals[i],
                                                                    subject.sampling_frequency,
                                                                    frequency_filters[ff_idx + 1][0],
                                                                    frequency_filters[ff_idx][1],
                                                                    filter_length=filter_length,
                                                                    l_trans_bandwidth=trans_bandwidth,
                                                                    h_trans_bandwidth=trans_bandwidth,
                                                                    verbose='CRITICAL')
                    except:
                        band_stop_filtered_signals[i] = notch_filter(band_stop_filtered_signals[i],
                                                                     subject.sampling_frequency,
                                                                     (frequency_filters[0][0] + frequency_filters[-1][
                                                                         1]) / 2,
                                                                     verbose='CRITICAL')
    # gap = True
    # transformed_heatmap = np.transpose(heatmap)
    # for i, ch in enumerate(transformed_heatmap):
    #     a = kmeans.predict(ch.reshape(-1, 1))
    #     if significant_cluster in a:
    #         frequency = frequencies[i]
    #         if gap:
    #             frequency_filters.append([frequency, frequency])
    #         else:
    #             frequency_filters[-1][1] = frequency
    #         significant_frequencies.append(frequency)
    #         gap = False
    #     else:
    #         gap = True

    #
    # band_stop_filtered_signals = filter_data(subject.signals, subject.sampling_frequency,
    #                                          max(1, frequency_filters[0][0] - trans_bandwidth),
    #                                          frequency_filters[-1][1] + trans_bandwidth,
    #                                          filter_length=filter_length, l_trans_bandwidth=trans_bandwidth,
    #                                          h_trans_bandwidth=trans_bandwidth)
    #
    # for ff_idx in range(len(frequency_filters) - 1):
    #     l_pass = frequency_filters[ff_idx][1] + trans_bandwidth
    #     h_pass = frequency_filters[ff_idx + 1][0] - trans_bandwidth
    #     if l_pass < h_pass:
    #         band_stop_filtered_signals = filter_data(band_stop_filtered_signals, subject.sampling_frequency,
    #                                                  frequency_filters[ff_idx + 1][0], frequency_filters[ff_idx][1],
    #                                                  filter_length=filter_length, l_trans_bandwidth=trans_bandwidth,
    #                                                  h_trans_bandwidth=trans_bandwidth)
    with warnings.catch_warnings(action="ignore"):
        highlevel.write_edf('tmp.edf', band_stop_filtered_signals, subject.signal_headers, subject.header)
    if inverse:
        with warnings.catch_warnings(action="ignore"):
            highlevel.drop_channels('tmp.edf', edf_target='filtered_inverse.edf', to_keep=significant_channels)
    else:
        with warnings.catch_warnings(action="ignore"):
            highlevel.drop_channels('tmp.edf', edf_target='filtered.edf', to_keep=significant_channels)
    return len(significant_frequencies), len(frequencies) * len(channels), len(significant_channels), len(channels)

# subject = Subject('dataset_temp/s14.edf')
# heatmap = np.load('parafac_analysis/significant_heatmaps/s14.npy', allow_pickle=True).item()
# main(heatmap, subject)
