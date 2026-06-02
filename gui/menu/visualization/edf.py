from PySide6.QtWidgets import QFileDialog
from visualization.edf_in_mne_browser import visualize_edf_in_mne_browser

def visualize_edf(parent=None):
    filename, _ = QFileDialog.getOpenFileName(
        parent, "Select EDF file", "", "European Data Format files (*.edf)"
    )
    if filename:
        visualize_edf_in_mne_browser(filename)
