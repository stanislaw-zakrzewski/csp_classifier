import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from config import accuracy_over_bands_show_standard_deviation

'''
    `accuracy_data` needs to be an dictionary with following keys:
     - frequency (exact middle of the band)
     - accuracy
     - configuration (with band start and width or end specified)
'''


def visualize_accuracy_over_bands(accuracy_data):
    if accuracy_over_bands_show_standard_deviation:
        errorbar = 'sd'
    else:
        errorbar = None

    ax = sns.lineplot(data=accuracy_data, x="frequency", y="accuracy", hue="configuration", errorbar=errorbar)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(.5))
    plt.grid()
    plt.show()
