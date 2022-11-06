import seaborn as sns
import matplotlib.pyplot as plt

'''
    `accuracy_data` needs to be an dictionary with following keys:
     - frequency (exact middle of the band)
     - accuracy
     - configuration (with band start and width or end specified)
'''


def visualize_accuracy_over_bands(accuracy_data):
    sns.lineplot(data=accuracy_data, x="frequency", y="accuracy", hue="configuration")
    plt.show()
