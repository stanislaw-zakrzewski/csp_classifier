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
from gui.components.back_button import BackButton
from gui.components.title_label import TitleLabel

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
        
        # App Title (extracted component)
        app_title = TitleLabel("Kombajn EEG")
        self.layout.addWidget(app_title)
        
        # Back button (extracted component)
        back_btn = BackButton(controller)
        self.layout.addWidget(back_btn)
        
        # File selector row
        file_row = QHBoxLayout()
        self.layout.addLayout(file_row)
        
        self.select_btn = QPushButton("Select EDF file")
        self.select_btn.clicked.connect(self.select_edf_file)
        self.select_btn.setStyleSheet("""
            QPushButton {
                background-color: #1e1e1e;
                color: #ffffff;
                border: 1px solid #2d2d2d;
                border-radius: 4px;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #2d2d2d;
            }
        """)
        file_row.addWidget(self.select_btn)
        
        self.file_label = QLabel("No file selected")
        self.file_label.setStyleSheet("padding: 8px; color: #aaaaaa;")
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
            
            # Create a Figure with dark styling
            figure = Figure(figsize=(25, 10), facecolor='#121212')
            ax = figure.subplots()
            ax.set_facecolor('#1e1e1e')
            
            # Style the labels, ticks, and spine borders
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')
            ax.grid(True, color='#2d2d2d')
            for spine in ax.spines.values():
                spine.set_color('#2d2d2d')
                
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
            
            # Style the Legend for dark mode
            legend = ax.get_legend()
            if legend:
                legend.get_frame().set_facecolor('#1e1e1e')
                legend.get_frame().set_edgecolor('#2d2d2d')
                for text in legend.get_texts():
                    text.set_color('white')
            
            ax.xaxis.set_major_locator(ticker.MultipleLocator(.5))
            
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
