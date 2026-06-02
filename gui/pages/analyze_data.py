from PySide6.QtWidgets import QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog, QScrollArea
from PySide6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.ticker as ticker
import seaborn as sns

from analyze_data import analyze_edf as analyze_edf_prime
from config.config import Configurations
from gui.colors import colors
from gui.fonts import fonts
from gui.pages.start_page import StartPage

class AnalyzeData(QScrollArea):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setStyleSheet(f"background-color: {colors['white_smoke']}; border: none;")
        self.configurations = Configurations()
        self.canvas = None
        self.toolbar = None
        self.selected_file_path = ""
        
        content_widget = QWidget()
        self.setWidget(content_widget)
        
        self.layout = QVBoxLayout(content_widget)
        self.layout.setAlignment(Qt.AlignTop)
        
        # App Title
        app_title = QLabel("Kombajn EEG")
        app_title.setFont(fonts['large_bold_font'])
        app_title.setStyleSheet("margin: 10px; border: none;")
        self.layout.addWidget(app_title)
        
        # Back button
        back_btn = QPushButton("Back to Start Page")
        back_btn.setFont(fonts['medium_font'])
        back_btn.clicked.connect(lambda: controller.show_frame(StartPage))
        back_btn.setStyleSheet("""
            QPushButton {
                background-color: white;
                border: 1px solid #CCCCCC;
                border-radius: 4px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #EAEAEA;
            }
        """)
        self.layout.addWidget(back_btn)
        
        # File selector row
        file_row = QHBoxLayout()
        self.layout.addLayout(file_row)
        
        self.select_btn = QPushButton("Select EDF file")
        self.select_btn.clicked.connect(self.select_edf_file)
        self.select_btn.setStyleSheet("padding: 8px;")
        file_row.addWidget(self.select_btn)
        
        self.file_label = QLabel("No file selected")
        self.file_label.setStyleSheet("padding: 8px;")
        file_row.addWidget(self.file_label)
        file_row.addStretch()
        
        # Analyze button
        self.analyze_btn = QPushButton("Analyze selected EDF")
        self.analyze_btn.clicked.connect(self.analyze_edf_gui)
        self.analyze_btn.setStyleSheet("""
            QPushButton {
                padding: 10px; 
                background-color: #4CAF50; 
                color: white; 
                border: none; 
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.layout.addWidget(self.analyze_btn)
        
        # Matplotlib chart container
        self.chart_container = QVBoxLayout()
        self.layout.addLayout(self.chart_container)

    def select_edf_file(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Select EDF file", "", "European Data Format files (*.edf)"
        )
        if filename:
            self.selected_file_path = filename
            self.file_label.setText(filename)
            
    def analyze_edf_gui(self):
        if self.selected_file_path:
            accuracy_data = analyze_edf_prime(
                self.selected_file_path,
                classifier_type=self.configurations.read('analyze_data.classifier'),
                verbose='ERROR'
            )
            
            # Create a Figure
            figure = Figure(figsize=(25, 10))
            ax = figure.subplots()
            sns.lineplot(
                data=accuracy_data, 
                x="frequency", 
                y="accuracy", 
                hue="configuration", 
                errorbar=None, 
                ax=ax, 
                markers=True, 
                style='configuration'
            )
            ax.xaxis.set_major_locator(ticker.MultipleLocator(.5))
            ax.grid()
            
            # Remove old canvas and toolbar if any
            if self.canvas is not None:
                self.chart_container.removeWidget(self.canvas)
                self.canvas.deleteLater()
            if self.toolbar is not None:
                self.chart_container.removeWidget(self.toolbar)
                self.toolbar.deleteLater()
                
            # Create and add new canvas & toolbar
            self.canvas = FigureCanvas(figure)
            self.toolbar = NavigationToolbar(self.canvas, self)
            
            self.chart_container.addWidget(self.canvas)
            self.chart_container.addWidget(self.toolbar)
