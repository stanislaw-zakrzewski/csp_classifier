from PySide6.QtWidgets import QMenuBar, QMenu
from PySide6.QtGui import QAction
from gui.menu.visualization.edf import visualize_edf
from gui.menu.visualization.menu import VisualizationMenu
from gui.settings import Settings

class ApplicationMenu(QMenuBar):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        
        # File Menu
        self.file_menu = QMenu("File", self)
        
        self.new_action = QAction("New", self)
        self.file_menu.addAction(self.new_action)
        
        self.open_action = QAction("Open", self)
        self.file_menu.addAction(self.open_action)
        
        self.save_action = QAction("Save", self)
        self.file_menu.addAction(self.save_action)
        
        self.file_menu.addSeparator()
        
        self.exit_action = QAction("Exit", self)
        self.exit_action.triggered.connect(self.exit_app)
        self.file_menu.addAction(self.exit_action)
        
        self.addMenu(self.file_menu)
        
        # Visualize Menu
        self.visualization_menu = VisualizationMenu(self)
        self.addMenu(self.visualization_menu)
        
        # Settings Action directly in MenuBar
        self.settings_action = QAction("Settings", self)
        self.settings_action.triggered.connect(self.open_settings)
        self.addAction(self.settings_action)
        
    def exit_app(self):
        if self.parent_window:
            self.parent_window.close()
            
    def open_settings(self):
        dialog = Settings(self.parent_window)
        dialog.exec()

def create_menu(parent):
    return ApplicationMenu(parent)