import os
from PySide6.QtWidgets import (QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, 
                             QFileDialog, QScrollArea, QTableWidget, QTableWidgetItem, QHeaderView)
from PySide6.QtCore import Qt
from pyedflib import highlevel

from gui.annotation_viewer import open_annotation_viewer
from gui.colors import colors
from gui.fonts import fonts
from gui.components.back_button import BackButton
from gui.components.title_label import TitleLabel

SELECTED_FIELDS = [
    'patientname',
    'startdate',
    'channels',
    'annotations'
]

class BrowseRecordings(QScrollArea):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setStyleSheet(f"background-color: {colors['white_smoke']}; border: none;")
        self.edf_paths = []
        self.edf_headers = {}

        content_widget = QWidget()
        self.setWidget(content_widget)

        self.layout = QVBoxLayout(content_widget)
        self.layout.setAlignment(Qt.AlignTop)

        # Title (extracted component)
        app_title = TitleLabel("Kombajn EEG")
        self.layout.addWidget(app_title)

        # Back Button (extracted component)
        back_btn = BackButton(controller)
        self.layout.addWidget(back_btn)

        # Load button
        load_btn = QPushButton("Load Data from Folder")
        load_btn.setFont(fonts['medium_font'])
        load_btn.setStyleSheet("""
            QPushButton {
                padding: 10px; 
                background-color: #2196F3; 
                color: white; 
                border: none; 
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #0b7dda;
            }
        """)
        load_btn.clicked.connect(self.add_files_from_folder)
        self.layout.addWidget(load_btn)

        # Table Widget styled dark
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(['Filename', 'Patient Name', 'Start Date', 'Channels', 'Annotations'])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setStyleSheet("""
            QTableWidget {
                background-color: #1e1e1e;
                gridline-color: #2d2d2d;
                border: 1px solid #2d2d2d;
                color: #ffffff;
            }
            QHeaderView::section {
                background-color: #2d2d2d;
                color: #ffffff;
                padding: 6px;
                border: 1px solid #333333;
                font-weight: bold;
            }
        """)
        self.layout.addWidget(self.table)

    def add_files_from_folder(self):
        folder_selected = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder_selected:
            edf_files = [f for f in os.listdir(folder_selected) if f.endswith('.edf')]
            
            # Clear old rows first
            self.table.setRowCount(0)
            
            for filename in edf_files:
                if filename not in self.edf_headers:
                    full_path = os.path.join(folder_selected, filename)
                    try:
                        header = highlevel.read_edf_header(full_path)
                        self.edf_headers[filename] = header
                        
                        row = self.table.rowCount()
                        self.table.insertRow(row)
                        
                        # Set file name item
                        self.table.setItem(row, 0, QTableWidgetItem(filename))
                        
                        # Set fields
                        col_idx = 1
                        for key in SELECTED_FIELDS:
                            if key == 'annotations':
                                btn = QPushButton("View Annotations")
                                btn.setStyleSheet("""
                                    QPushButton {
                                        background-color: #2b2b2b;
                                        color: white;
                                        border: 1px solid #3d3d3d;
                                        border-radius: 3px;
                                        padding: 4px;
                                    }
                                    QPushButton:hover {
                                        background-color: #3d3d3d;
                                    }
                                """)
                                btn.clicked.connect(
                                    lambda checked=False, f=filename, ann=header[key]: 
                                    open_annotation_viewer(self, f, ann)
                                )
                                self.table.setCellWidget(row, col_idx, btn)
                            else:
                                val = str(header.get(key, ""))
                                self.table.setItem(row, col_idx, QTableWidgetItem(val))
                            col_idx += 1
                    except Exception as e:
                        print(f"Error loading EDF {filename}: {e}")
