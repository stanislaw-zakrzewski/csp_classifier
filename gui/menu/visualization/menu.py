from tkinter import *

from gui.menu.visualization.edf import visualize_edf


class VisualizationMenu(Menu):
    def __init__(self, root):
        Menu.__init__(self, root, tearoff=0)
        self.add_command(label="Visualize EDF file", command=visualize_edf)
