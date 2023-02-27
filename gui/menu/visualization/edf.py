from tkinter import filedialog as fd

from visualization.edf_in_mne_browser import visualize_edf_in_mne_browser


def visualize_edf():
    filename = fd.askopenfilename(filetypes=[("European Data Format files", "*.edf")])
    if filename:
        visualize_edf_in_mne_browser(filename)
