import sys
from pathlib import Path

# Add project root directory to sys.path to support running directly as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PySide6.QtWidgets import QApplication, QMainWindow, QStackedWidget

from gui.menu.menu import ApplicationMenu
from gui.pages.analyze_data import AnalyzeData
from gui.pages.browse_recordings import BrowseRecordings
from gui.pages.collect_data import CollectData
from gui.pages.erds_analysis import ERDSAnalysis
from gui.pages.filter_browser import FilterBrowser
from gui.pages.prompt_viewer import PromptViewer
from gui.pages.start_page import StartPage
from gui.pages.test_electrodes import TestElectrodes
from gui.pages.real_time import RealTime
from gui.pages.adaptive_real_time import AdaptiveRealTime

class App(QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setWindowTitle("Kombajn EEG")
        self.showMaximized()
        
        # Set up menu
        self.setMenuBar(ApplicationMenu(self))
        
        pages = [
            {'name': 'Test Electrodes', 'frame': TestElectrodes},
            {'name': 'Filter Browser', 'frame': FilterBrowser},
            {'name': 'Analyze Data', 'frame': AnalyzeData},
            {'name': 'Collect Data', 'frame': CollectData},
            {'name': 'ERD/S Analysis', 'frame': ERDSAnalysis},
            {'name': 'Browse Recordings', 'frame': BrowseRecordings},
            {'name': 'Prompt Viewer', 'frame': PromptViewer},
            {'name': 'Real-Time Analysis', 'frame': RealTime},
            {'name': 'Adaptive Real-Time Analysis', 'frame': AdaptiveRealTime},
        ]
        
        self.stacked_widget = QStackedWidget(self)
        self.setCentralWidget(self.stacked_widget)
        
        self.frames = {}
        
        # StartPage frame initialization (needs the pages list)
        start_frame = StartPage(self.stacked_widget, self, pages)
        self.frames[StartPage] = start_frame
        self.stacked_widget.addWidget(start_frame)
        
        self.show_frame(StartPage)
        
    def show_frame(self, cont):
        # 1. Notify the current page it is being hidden
        current_widget = self.stacked_widget.currentWidget()
        if current_widget and hasattr(current_widget, 'on_hide') and callable(current_widget.on_hide):
            try:
                current_widget.on_hide()
            except Exception as e:
                print(f"Error in on_hide for {current_widget.__class__.__name__}: {e}")
                
        # 2. Lazily instantiate target page if it is not created yet
        if cont not in self.frames:
            try:
                frame = cont(self.stacked_widget, self)
                self.frames[cont] = frame
                self.stacked_widget.addWidget(frame)
            except Exception as e:
                print(f"Error instantiating page {cont.__name__}: {e}")
                return
        else:
            frame = self.frames[cont]
            
        # 3. Notify the target page it is being shown
        if hasattr(frame, 'on_show') and callable(frame.on_show):
            try:
                frame.on_show()
            except Exception as e:
                print(f"Error in on_show for {frame.__class__.__name__}: {e}")
                
        self.stacked_widget.setCurrentWidget(frame)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    dark_stylesheet = """
        QWidget {
            background-color: #121212;
            color: #ffffff;
        }
        QMainWindow, QDialog, QScrollArea {
            background-color: #121212;
            border: none;
        }
        QLabel {
            background-color: transparent;
            color: #ffffff;
        }
        QPushButton {
            background-color: #1e1e1e;
            color: #ffffff;
            border: 1px solid #2d2d2d;
            border-radius: 4px;
            padding: 8px 15px;
        }
        QPushButton:hover {
            background-color: #2d2d2d;
            border-color: #3e3e3e;
        }
        QPushButton:pressed {
            background-color: #3d3d3d;
        }
        QPushButton:disabled {
            background-color: #121212;
            color: #888888;
            border: 1px solid #222222;
        }
        QLineEdit, QTextEdit {
            background-color: #2b2b2b;
            color: #ffffff;
            border: 1px solid #3d3d3d;
            border-radius: 4px;
            padding: 6px;
        }
        QLineEdit:focus {
            border: 1px solid #2196F3;
        }
        QComboBox {
            background-color: #2b2b2b;
            color: #ffffff;
            border: 1px solid #3d3d3d;
            border-radius: 4px;
            padding: 5px;
        }
        QComboBox QAbstractItemView {
            background-color: #2b2b2b;
            color: #ffffff;
            selection-background-color: #3d3d3d;
        }
        QListWidget, QTableWidget {
            background-color: #2b2b2b;
            color: #ffffff;
            border: 1px solid #3d3d3d;
            border-radius: 4px;
        }
        QTableWidget QTableCornerButton::section {
            background-color: #1e1e1e;
            border: 1px solid #2d2d2d;
        }
        QHeaderView::section {
            background-color: #1e1e1e;
            color: #ffffff;
            padding: 5px;
            border: 1px solid #2d2d2d;
            font-weight: bold;
        }
        QScrollBar:vertical {
            border: none;
            background: #121212;
            width: 10px;
            margin: 0px;
        }
        QScrollBar::handle:vertical {
            background: #2d2d2d;
            min-height: 20px;
            border-radius: 5px;
        }
        QScrollBar::handle:vertical:hover {
            background: #3d3d3d;
        }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            height: 0px;
        }
        QScrollBar:horizontal {
            border: none;
            background: #121212;
            height: 10px;
            margin: 0px;
        }
        QScrollBar::handle:horizontal {
            background: #2d2d2d;
            min-width: 20px;
            border-radius: 5px;
        }
        QScrollBar::handle:horizontal:hover {
            background: #3d3d3d;
        }
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
            width: 0px;
        }
        QMenuBar {
            background-color: #1e1e1e;
            color: #ffffff;
            border-bottom: 1px solid #2d2d2d;
        }
        QMenuBar::item:selected {
            background-color: #2d2d2d;
        }
        QMenu {
            background-color: #1e1e1e;
            color: #ffffff;
            border: 1px solid #2d2d2d;
        }
        QMenu::item:selected {
            background-color: #2d2d2d;
        }
    """
    app.setStyleSheet(dark_stylesheet)
    
    window = App()
    window.show()
    sys.exit(app.exec())

