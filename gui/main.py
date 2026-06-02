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
        ]
        
        self.stacked_widget = QStackedWidget(self)
        self.setCentralWidget(self.stacked_widget)
        
        self.frames = {}
        
        # StartPage frame initialization (needs the pages list)
        start_frame = StartPage(self.stacked_widget, self, pages)
        self.frames[StartPage] = start_frame
        self.stacked_widget.addWidget(start_frame)
        
        # Other frames initialization
        for p in pages:
            F = p['frame']
            frame = F(self.stacked_widget, self)
            self.frames[F] = frame
            self.stacked_widget.addWidget(frame)
            
        self.show_frame(StartPage)
        
    def show_frame(self, cont):
        frame = self.frames[cont]
        self.stacked_widget.setCurrentWidget(frame)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec())
