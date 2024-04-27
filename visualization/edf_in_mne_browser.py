import matplotlib.pyplot as plt
from mne.io import read_raw_edf


def visualize_edf_in_mne_browser(edf_path):
    """Visualize data contained in EDF file in mne browser.

    Parameters
    ----------
    edf_path : string
        Path to EDF file that is meant to be visualized.
    """
    raw = read_raw_edf(edf_path, preload=True)
    raw.plot(block=True, show_options=True)
    plt.show()
