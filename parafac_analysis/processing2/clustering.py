import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import pandas as pd

from tkinter import filedialog as fd
from sklearn.cluster import DBSCAN, KMeans, HDBSCAN, k_means
from scipy.signal import argrelextrema, find_peaks


def visualize_best_decomposition(dataframe, filename, centroids, x_values, x_label):
    fig, _ = plt.subplots(2, 1)
    ax0 = plt.subplot(211)
    ax1 = ax0.twinx()
    ax2 = plt.subplot(212)
    # ax3 = ax2.twinx()

    fig.set_figwidth(20)
    fig.set_figheight(15)
    sns.lineplot(dataframe, x='min_cluster_size', y='noise', ax=ax0)

    sns.lineplot(data=dataframe, x='min_cluster_size', y='cluster_count', ax=ax1, color='red')
    ax1.set_ylim([0, None])
    ax0.xaxis.grid(True)
    # ax0.xaxis.set_major_locator(ticker.MultipleLocator(1))

    # sns.lineplot(dataframe, x='min_cluster_size', y='noise', ax=ax2)

    for i in centroids:
        # for i in statistically_significant_atoms:
        b = []
        prev = 0
        for j in i:
            b.append(j - prev)
            prev = j
        ax2.plot(x_values, b)
    # sns.lineplot(data=dataframe, x='min_cluster_size', y='cluster_count', ax=ax3, color='red')
    # ax3.set_ylim([0, None])
    ax2.xaxis.grid(True)
    ax2.set_xlabel(x_label)
    ax2.set_ylabel('amplitude')
    # ax2.xaxis.set_major_locator(ticker.MultipleLocator(1))

    fig.suptitle(f'Quality of PARAFAC decomposition for:\n{filename}', fontsize=16)
    plt.show()


def visualize_clustering(centroids_a_fq, centroids_a_ch, centroids_b_fq, centroids_b_ch, filename, ch_x, fq_x, a_label, b_label):
    fig, _ = plt.subplots(2, 2)
    ax0 = plt.subplot(221)
    ax1 = plt.subplot(222)
    ax2 = plt.subplot(223)
    ax3 = plt.subplot(224)

    fig.set_figwidth(30)
    fig.set_figheight(15)

    ax0.set_ylabel('amplitude')
    ax0.set_xlabel('channel')
    ax0.xaxis.grid(True)
    ax0.set_title(a_label)
    for i in centroids_a_ch:
        b = []
        prev = 0
        for j in i:
            b.append(j - prev)
            prev = j
        ax0.plot(ch_x, b)

    ax1.set_ylabel('amplitude')
    ax1.set_xlabel('channel')
    ax1.xaxis.grid(True)
    ax1.set_title(b_label)
    for i in centroids_b_ch:
        b = []
        prev = 0
        for j in i:
            b.append(j - prev)
            prev = j
        ax1.plot(ch_x, b)

    ax2.set_ylabel('amplitude')
    ax2.set_xlabel('frqequency')
    ax2.xaxis.grid(True)
    for i in centroids_a_fq:
        b = []
        prev = 0
        for j in i:
            b.append(j - prev)
            prev = j
        ax2.plot(fq_x, b)

    ax3.set_ylabel('amplitude')
    ax3.set_xlabel('frqequency')
    ax3.xaxis.grid(True)
    for i in centroids_b_fq:
        b = []
        prev = 0
        for j in i:
            b.append(j - prev)
            prev = j
        ax3.plot(fq_x, b)
    # sns.lineplot(data=dataframe, x='min_cluster_size', y='cluster_count', ax=ax3, color='red')
    # ax3.set_ylim([0, None])

    # ax2.xaxis.set_major_locator(ticker.MultipleLocator(1))

    fig.suptitle(f'Label related channels and frequencies clustering:\n{filename.split("/")[-1]}', fontsize=16)
    plt.show()


filename = fd.askopenfilename(filetypes=[("NumPy data files", "*.npy")])


# filename = 'parafac_analysis/results/2024-06-05T22-38-32_s43.npy'

def transform_to_additive(data_to_transform):
    transformed_data = []
    for i in data_to_transform:
        a = []
        prev = 0
        for j in i:
            prev += j
            a.append(prev)
        if prev == 0:
            transformed_data.append(a)
        else:
            transformed_data.append(a / prev)
    return transformed_data


def cluster_processed_data(processed_data):
    clustering_data = {'noise': [], 'cluster_count': [], 'min_cluster_size': []}

    noise_data = []
    cluster_counts = []
    noise_range = range(2, int(len(processed_data) / 2), 1)
    for i in noise_range:
        hdb = HDBSCAN(min_cluster_size=i, store_centers='both')
        hdb.fit(processed_data)
        clustering_data['noise'].append(np.count_nonzero(hdb.labels_ == -1) / len(processed_data) * 100)
        clustering_data['min_cluster_size'].append(i)
        clustering_data['cluster_count'].append(max(hdb.labels_) + 1)
        if max(hdb.labels_) + 1 == 0:
            break
    local_maxima_indexes = argrelextrema(np.array(clustering_data['noise']), np.less, order=2)
    return clustering_data, local_maxima_indexes[0]


def select_best_cluster(processed_data):
    clustering_data, local_maxima_indexes = cluster_processed_data(processed_data)

    print('min_cluster_size', np.array(clustering_data['min_cluster_size'])[local_maxima_indexes])
    print('cluster_count', np.array(clustering_data['cluster_count'])[local_maxima_indexes])
    print('noise', np.array(clustering_data['noise'])[local_maxima_indexes])

    return int(input('Select min_cluster_size:'))


with open(filename, 'rb') as out_file:
    data_file = np.load(out_file, allow_pickle=True)
    all_decompositions = data_file.item().get("decompositions")
    selected_channels = data_file.item().get("metadata")['selected_channels']
    a_label = data_file.item().get("metadata")['a_label']
    b_label = data_file.item().get("metadata")['b_label']
    statistically_significant_decompositions = data_file.item().get("statistically_significant_decompositions")
    selected_rank = 30

    rank_decompositions = all_decompositions[selected_rank]
    statistically_significant_a_frequencies_atoms = []
    statistically_significant_a_channels_atoms = []
    statistically_significant_b_frequencies_atoms = []
    statistically_significant_b_channels_atoms = []
    # for statistically_significant_decomposition in statistically_significant_decompositions:
    #     if statistically_significant_decomposition['rank'] == selected_rank:
    #         replica_key = statistically_significant_decomposition['replica']
    #         atom_key = statistically_significant_decomposition['atom']
    #         all_atoms = rank_decompositions[replica_key].factors.factors
    #         channel_factors = all_atoms[1][:, atom_key]
    #         frequency_factors = all_atoms[2][:, atom_key]
    #         # statistically_significant_atoms.append(channel_factors)
    #         statistically_significant_atoms.append(frequency_factors)
    #         # statistically_significant_atoms.append(np.concatenate((channel_factors, frequency_factors)))

    for statistically_significant_decomposition in statistically_significant_decompositions:
        relation = statistically_significant_decomposition['relation']

        rank_key = statistically_significant_decomposition['rank']
        replica_key = statistically_significant_decomposition['replica']
        atom_key = statistically_significant_decomposition['atom']
        rank_decompositions = all_decompositions[rank_key]
        all_atoms = rank_decompositions[replica_key].factors.factors
        channel_factors = all_atoms[1][:, atom_key]
        frequency_factors = all_atoms[2][6:24, atom_key]

        if relation == 'greater':
            statistically_significant_a_frequencies_atoms.append(frequency_factors)
            statistically_significant_a_channels_atoms.append(channel_factors)
        else:
            statistically_significant_b_frequencies_atoms.append(frequency_factors)
            statistically_significant_b_channels_atoms.append(channel_factors)

    processed_a_frequencies_data = transform_to_additive(statistically_significant_a_frequencies_atoms)
    processed_a_channels_data = transform_to_additive(statistically_significant_a_channels_atoms)
    processed_b_frequencies_data = transform_to_additive(statistically_significant_b_frequencies_atoms)
    processed_b_channels_data = transform_to_additive(statistically_significant_b_channels_atoms)

    # for i in processed_data:
    # # for i in statistically_significant_atoms:
    #     plt.plot(range(len(i)), i)
    # plt.show()

    # processed_data = []
    # max_val = np.max(statistically_significant_atoms) * .1
    # for i in statistically_significant_atoms:
    #     a = []
    #     for j in i:
    #         if j < max_val:
    #             a.append(0)
    #         else:
    #             a.append(j)
    #     processed_data.append(a)
    # for i in processed_data:
    # # for i in statistically_significant_atoms:
    #     plt.plot(range(len(i)), i)
    # plt.show()

    # # clustering = DBSCAN(eps=.1, min_samples=2).fit(statistically_significant_atoms)
    # clustering = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(statistically_significant_atoms)
    # print(clustering.labels_)
    # print(clustering.cluster_centers_)
    # fig, (ax1, ax2) = plt.subplots(2)
    # fig.suptitle(
    #     'PARAFAC decomposition of rank {}, replica: {})'.format(rank, replica))
    # fig.set_figwidth(25)
    # fig.set_figheight(10)

    # for i in statistically_significant_atoms:
    #     avg = np.average(i)
    #     i -= avg

    selected_min_cluster_size_a_fq = select_best_cluster(processed_a_frequencies_data)
    selected_min_cluster_size_a_ch = select_best_cluster(processed_a_channels_data)
    selected_min_cluster_size_b_fq = select_best_cluster(processed_b_frequencies_data)
    selected_min_cluster_size_b_ch = select_best_cluster(processed_b_channels_data)

    hdb_a_fq = HDBSCAN(min_cluster_size=selected_min_cluster_size_a_fq, store_centers='both')
    hdb_a_fq.fit(processed_a_frequencies_data)
    hdb_a_ch = HDBSCAN(min_cluster_size=selected_min_cluster_size_a_ch, store_centers='both')
    hdb_a_ch.fit(processed_a_channels_data)
    hdb_b_fq = HDBSCAN(min_cluster_size=selected_min_cluster_size_b_fq, store_centers='both')
    hdb_b_fq.fit(processed_b_frequencies_data)
    hdb_b_ch = HDBSCAN(min_cluster_size=selected_min_cluster_size_b_ch, store_centers='both')
    hdb_b_ch.fit(processed_b_channels_data)

    visualize_clustering(hdb_a_fq.centroids_, hdb_a_ch.centroids_, hdb_b_fq.centroids_, hdb_b_ch.centroids_, filename,
                         selected_channels, range(6, 24), a_label, b_label)

    # dataframe = pd.DataFrame(data=clustering_data)
    # visualize_best_decomposition(dataframe, filename, hdb.centroids_, selected_channels, 'channels')

    # hdb.fit(statistically_significant_atoms)
    # print(hdb.labels_)
    # print('Noisy samples: {0:.2f}%'.format(np.count_nonzero(hdb.labels_ == -1)/len(processed_data) * 100))
