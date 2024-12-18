import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from tkinter import filedialog as fd
import numpy as np
from matplotlib.ticker import FormatStrFormatter


def visualize_combined_atoms_as_heatmap(heatmap_path, combined_atoms_a, combined_atoms_b, selected_channels,
                                        frequencies,
                                        a_label_name, b_label_name):
    sns.set_theme(rc={'figure.figsize': (7, 4)})
    df_a_data = {'Channel': [], "Frequency": [], "Amplitude": []}
    df_b_data = {'Channel': [], "Frequency": [], "Amplitude": []}
    for channel_index, channel_name in enumerate(selected_channels):
        for frequency_index, frequency in enumerate(frequencies):
            df_a_data['Channel'].append(channel_name)
            df_a_data['Frequency'].append(int(frequency))
            df_a_data['Amplitude'].append(combined_atoms_a[channel_index][frequency_index])
            df_b_data['Channel'].append(channel_name)
            df_b_data['Frequency'].append(int(frequency))
            df_b_data['Amplitude'].append(combined_atoms_b[channel_index][frequency_index])
    df_a = pd.DataFrame(df_a_data)
    df_a['Channel'] = pd.Categorical(df_a['Channel'], categories=selected_channels)
    df_a = df_a.sort_values('Channel')
    df_b = pd.DataFrame(df_b_data)
    df_b['Channel'] = pd.Categorical(df_b['Channel'], categories=selected_channels)
    df_b = df_b.sort_values('Channel')
    vmin = min(min(df_a['Amplitude']), min(df_b['Amplitude']))
    vmax = min(max(df_a['Amplitude']), max(df_b['Amplitude']))
    fig, axes = plt.subplots(ncols=3,
                             gridspec_kw=dict(width_ratios=[len(selected_channels), len(selected_channels), 0.5]))
    a = df_a.pivot(index='Frequency', columns='Channel', values='Amplitude')
    b = df_b.pivot(index='Frequency', columns='Channel', values='Amplitude')
    sns.heatmap(a, ax=axes[0], cbar=False, vmin=vmin, vmax=vmax)
    sns.heatmap(b, ax=axes[1], cbar=False, vmin=vmin, vmax=vmax)
    axes[1].set(ylabel=None)
    axes[0].set_title(f'{a_label_name} imagery movement')
    axes[1].set_title(f'{b_label_name} imagery movement')
    axes[0].invert_yaxis()
    axes[1].invert_yaxis()
    fig.colorbar(axes[0].collections[0], cax=axes[2])
    # fig.suptitle(heatmap_path)
    plt.show()


# heatmap_path = fd.askopenfilename(filetypes=[("NumPy data files", "*.npy")])
heatmap_path = 'parafac_analysis/significant_heatmaps/s48.npy'
heatmap_data = np.load(heatmap_path, allow_pickle=True).item()

visualize_combined_atoms_as_heatmap(heatmap_path, heatmap_data['heatmap_a'], heatmap_data['heatmap_b'],
                                    heatmap_data['channels'],
                                    heatmap_data['frequencies'], heatmap_data['a_label_name'],
                                    heatmap_data['b_label_name'])
