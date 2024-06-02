import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import pandas as pd

from tkinter import filedialog as fd

filename = fd.askopenfilename(filetypes=[("NumPy data files", "*.npy")])

with open(filename, 'rb') as out_file:
    data_file = np.load(out_file, allow_pickle=True)
    dataframe_data = {'rank': [], 'replica': [], 'pvalue': [], 'weight': []}
    statistically_significant_decompositions = data_file.item().get("statistically_significant_decompositions")

    for statistically_significant_decomposition in statistically_significant_decompositions:
        dataframe_data['rank'].append(statistically_significant_decomposition['rank'])
        dataframe_data['replica'].append(statistically_significant_decomposition['replica'])
        dataframe_data['pvalue'].append(statistically_significant_decomposition['pvalue'])
        dataframe_data['weight'].append(statistically_significant_decomposition['weight'])
    dataframe = pd.DataFrame(data=dataframe_data)
    df2 = dataframe.groupby(['rank']).size().reset_index(name='statistically significant atoms')

    fig, _ = plt.subplots(2, 1)
    ax0 = plt.subplot(211)
    ax1 = ax0.twinx()
    ax2 = plt.subplot(212)
    ax3 = ax2.twinx()

    fig.set_figwidth(20)
    fig.set_figheight(15)
    sns.lineplot(dataframe, x='rank', y='pvalue', ax=ax0)
    ax0.set_yscale('log')
    ax1.set_yscale('linear')

    # ax12 = ax1.twinx()
    sns.lineplot(data=df2, x='rank', y='statistically significant atoms', ax=ax1, color='red')
    ax1.set_ylim([0, None])
    ax0.xaxis.grid(True)
    ax0.xaxis.set_major_locator(ticker.MultipleLocator(1))

    sns.lineplot(dataframe, x='rank', y='weight', ax=ax2)
    ax2.set_yscale('log')
    ax3.set_yscale('linear')

    # ax22 = ax2.twinx()
    sns.lineplot(data=df2, x='rank', y='statistically significant atoms', ax=ax3, color='red')
    ax3.set_ylim([0, None])
    ax2.xaxis.grid(True)
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(1))

    fig.suptitle(f'Quality of PARAFAC decomposition for:\n{filename}', fontsize=16)
    plt.show()
