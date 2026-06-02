from PySide6.QtWidgets import QMenu
from PySide6.QtGui import QAction
from gui.menu.visualization.edf import visualize_edf

class VisualizationMenu(QMenu):
    def __init__(self, parent=None):
        super().__init__("Visualize", parent)
        self.visualize_edf_action = QAction("Visualize EDF file", self)
        self.visualize_edf_action.triggered.connect(lambda checked=False: visualize_edf(self.parent()))
        self.addAction(self.visualize_edf_action)
