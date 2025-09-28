import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm
from sklearn.cluster import KMeans


def visualize(title, data, selected_channels, frequencies):
    data = np.abs(data)
    data *= 1000000
    data += 1
    print(np.min(data), np.max(data))
    sns.set_theme(rc={'figure.figsize': (6, 4)})
    df_a_data = {'Channel': [], "Frequency": [], "Amplitude": []}
    df_b_data = {'Channel': [], "Frequency": [], "Amplitude": []}
    for channel_index, channel_name in enumerate(selected_channels):
        for frequency_index, frequency in enumerate(frequencies):
            df_a_data['Channel'].append(channel_name)
            df_a_data['Frequency'].append(round(frequency, 1))
            df_a_data['Amplitude'].append(data[channel_index][frequency_index])
            df_b_data['Channel'].append(channel_name)
            df_b_data['Frequency'].append(frequency)
            df_b_data['Amplitude'].append(data[channel_index][frequency_index])
    df_a = pd.DataFrame(df_a_data)
    df_a['Channel'] = pd.Categorical(df_a['Channel'], categories=selected_channels)
    df_a = df_a.sort_values('Channel')
    df_b = pd.DataFrame(df_b_data)
    df_b['Channel'] = pd.Categorical(df_b['Channel'], categories=selected_channels)
    df_b = df_b.sort_values('Channel')
    vmin = min(min(df_a['Amplitude']), min(df_b['Amplitude']))
    vmax = min(max(df_a['Amplitude']), max(df_b['Amplitude']))
    fig, axes = plt.subplots(ncols=2,
                             gridspec_kw=dict(width_ratios=[len(selected_channels), 0.5]))
    a = df_a.pivot(index='Frequency', columns='Channel', values='Amplitude')
    sns.heatmap(df_a.pivot(index='Frequency', columns='Channel', values='Amplitude'), ax=axes[0], cbar=False, vmin=vmin)#, vmax=6000)#, norm=LogNorm(vmin=1, vmax=6000))
    # axes[0].set_title('a')
    # axes[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    # axes[0].invert_yaxis()
    # g2 = sns.heatmap(df_b.pivot(index='Frequency', columns='Channel', values='Amplitude'), ax=axes[1], cbar=False,
    #                  vmax=vmax)
    # g2.set(ylabel=None)
    # axes[1].set_title(b_label_name)
    axes[0].invert_yaxis()
    fig.colorbar(axes[0].collections[0], cax=axes[1])
    fig.suptitle(title)
    plt.show()


selected_trial = 0
a = np.load('decomposition.npy', allow_pickle=True)
a = a.item()
frequencies = a['frequencies']
channels = a['channels']
decompositions = a['decompositions']
labels = a['labels']
label_names = a['label_names']
significant = a['significant']

print('Left / Right')
for i in range(4):
    left_hand = np.average(decompositions[0][:100,i])
    right_hand = np.average(decompositions[0][100:,i])
    print(f"{left_hand}/{right_hand}")

def combine_atom(decompositions, atom_index, atom_aggregator):
    channel_factors = decompositions[1][:, atom_index]
    frequency_factors = decompositions[2][:, atom_index]

    for channel_factor_index, channel_factor in enumerate(channel_factors):
        atom_aggregator[channel_factor_index] += frequency_factors * channel_factor

a0 = [[0 for _ in frequencies] for _ in channels]
a1 = [[0 for _ in frequencies] for _ in channels]
a2 = [[0 for _ in frequencies] for _ in channels]
a3 = [[0 for _ in frequencies] for _ in channels]
combine_atom(decompositions, 0, a0)
combine_atom(decompositions, 1, a1)
combine_atom(decompositions, 2, a2)
combine_atom(decompositions, 3, a3)
a0 = np.array(a0)
a1 = np.array(a1)
a2 = np.array(a2)
a3 = np.array(a3)
a0 *= decompositions[0][selected_trial][0]
# a1 *= decompositions[0][selected_trial][1]
# a2 *= decompositions[0][selected_trial][2]
# a3 *= decompositions[0][selected_trial][3]

atoms = [a0,a1,a2,a3]
print(atoms)



visualize(f'trial {selected_trial + 1}', a['data'][selected_trial], a['channels'], a['frequencies'])
visualize('trial atom 1', a0, a['channels'], a['frequencies'])
visualize('atom 2', a1, a['channels'], a['frequencies'])
visualize('atom 3', a2, a['channels'], a['frequencies'])
visualize('atom 4', a3, a['channels'], a['frequencies'])
visualize('significance heatmap', a1+a2+a0, a['channels'], a['frequencies'])
for s in significant[0]:
    print(s)

heatmap_data = a1+a2+a0
kmeans = KMeans(n_clusters=2, n_init='auto')
kmeans.fit((heatmap_data).flatten().reshape(-1, 1))
significant_cluster = 0
for cluster_id, cluster_center in enumerate(kmeans.cluster_centers_):
    if kmeans.cluster_centers_[significant_cluster] < cluster_center:
        significant_cluster = cluster_id

one_hot = []
all_count = 0
selected_count = 0
for channel_idx, channel_data in enumerate(heatmap_data):
    preds = kmeans.predict(channel_data.reshape(-1, 1))
    one_hot.append(preds)
one_hot = np.array(one_hot)
visualize('selected significant parts of the signal', one_hot, a['channels'], a['frequencies'])

print(f"Label: {label_names[labels[selected_trial]]}")
print(f"Atom 0: {decompositions[0][selected_trial][0]}")
print(f"Atom 1: {decompositions[0][selected_trial][1]}")
print(f"Atom 2: {decompositions[0][selected_trial][2]}")
print(f"Atom 3: {decompositions[0][selected_trial][3]}")


