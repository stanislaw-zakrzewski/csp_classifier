import os
import pickle
import traceback
from PySide6.QtWidgets import QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog, QScrollArea
from PySide6.QtCore import Qt

# Import modules that are part of the saved pipelines to ensure pickle can deserialize them
import mne
import sklearn
import pyriemann
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression

from gui.colors import colors
from gui.fonts import fonts
from gui.components.back_button import BackButton
from gui.components.title_label import TitleLabel

class RealTime(QScrollArea):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setStyleSheet(f"background-color: {colors['white_smoke']}; border: none;")
        self.selected_pipeline_path = ""
        self.loaded_pipeline = None

        content_widget = QWidget()
        self.setWidget(content_widget)

        self.layout = QVBoxLayout(content_widget)
        self.layout.setAlignment(Qt.AlignTop)

        # App Title
        app_title = TitleLabel("Kombajn EEG")
        self.layout.addWidget(app_title)

        # Back button
        back_btn = BackButton(controller)
        self.layout.addWidget(back_btn)

        # Page Header
        page_title = QLabel("Real-Time Signal Analysis")
        page_title.setStyleSheet(f"color: {colors['text']}; font-size: 20px; font-weight: bold; margin-top: 10px; margin-bottom: 20px;")
        self.layout.addWidget(page_title)

        # File selector row
        file_row = QHBoxLayout()
        self.layout.addLayout(file_row)

        self.select_btn = QPushButton("Select Trained Pipeline File")
        self.select_btn.clicked.connect(self.select_pipeline_file)
        self.select_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {colors['surface']};
                color: {colors['text']};
                border: 1px solid {colors['border']};
                border-radius: 4px;
                padding: 8px 15px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {colors['border']};
            }}
            QPushButton:disabled {{
                background-color: #1a1a1a;
                color: #555555;
                border: 1px solid #222222;
            }}
        """)
        file_row.addWidget(self.select_btn)

        self.pipeline_label = QLabel("No pipeline file selected")
        self.pipeline_label.setStyleSheet("padding: 8px; color: #aaaaaa; font-weight: normal;")
        file_row.addWidget(self.pipeline_label)
        file_row.addStretch()

        # Status Label for loaded status/errors
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("padding: 8px; color: #2196F3; font-weight: bold;")
        self.layout.addWidget(self.status_label)

    def select_pipeline_file(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Select Trained Pipeline File", "", "Pickle files (*.pkl)"
        )
        if filename:
            self.selected_pipeline_path = filename
            self.load_pipeline_file(filename)

    def load_pipeline_file(self, filepath):
        try:
            self.status_label.setStyleSheet("color: #2196F3; font-weight: bold; padding: 4px;")
            self.status_label.setText("Loading pipeline...")

            with open(filepath, 'rb') as f:
                pipeline = pickle.load(f)

            # Validate pipeline structure
            if not hasattr(pipeline, 'steps'):
                raise ValueError("Selected file is not a valid scikit-learn Pipeline (missing 'steps' attribute).")

            self.loaded_pipeline = pipeline

            # Format human-readable pipeline step name
            name_mapping = {
                'CSP': 'CSP',
                'LinearDiscriminantAnalysis': 'LDA',
                'Covariances': 'Cov',
                'TangentSpace': 'Tangent Space',
                'LogisticRegression': 'LR',
                'SVC': 'SVM'
            }

            steps_names = []
            for _, step_obj in pipeline.steps:
                class_name = step_obj.__class__.__name__
                steps_names.append(name_mapping.get(class_name, class_name))

            pipeline_name = " + ".join(steps_names)

            # Update displays
            self.pipeline_label.setText(f"Loaded: {pipeline_name} ({os.path.basename(filepath)})")
            self.pipeline_label.setStyleSheet("padding: 8px; color: #4CAF50; font-weight: bold;")

            self.status_label.setText("Pipeline loaded successfully.")
            self.status_label.setStyleSheet("color: #4CAF50; font-weight: bold; padding: 4px;")

        except Exception as e:
            self.loaded_pipeline = None
            self.pipeline_label.setText("No pipeline file selected")
            self.pipeline_label.setStyleSheet("padding: 8px; color: #aaaaaa; font-weight: normal;")

            tb = traceback.format_exc()
            self.status_label.setText(f"Error loading pipeline: {e}")
            self.status_label.setStyleSheet("color: #f44336; font-weight: bold; padding: 4px;")
            print(f"Error loading pipeline:\n{tb}")

    def on_show(self):
        pass

    def on_hide(self):
        pass
