import os
import pickle
import traceback
import numpy as np
from threading import Thread

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
from config.config import Configurations
from gui.pages.collect_data_components.prompt_viewer import PromptViewer
from src.bci_integration.GtecNautilusProInterface import GtecNautilusProInterface
from gui.pages.collect_data import ThreadSafePromptViewerProxy, ThreadSafeTimelineSignaler

class RealTime(QScrollArea):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setStyleSheet(f"background-color: {colors['white_smoke']}; border: none;")
        self.configurations = Configurations()
        self.selected_pipeline_path = ""
        self.loaded_pipeline = None
        self.bci_interface = GtecNautilusProInterface()
        self.queue = None
        self.current_queue = None
        self.acquisition_thread = None
        self.prompt_viewer = None
        self.progressbar_value = 0.0
        self.is_running = False

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

        # Controls
        self.prepare_experiment = QPushButton("Prepare experiment")
        self.prepare_experiment.clicked.connect(self.open_prompt_window)
        self.prepare_experiment.setEnabled(False)
        self.prepare_experiment.setStyleSheet("""
            QPushButton {
                padding: 10px;
                background-color: #1e1e1e;
                color: #ffffff;
                border: 1px solid #2d2d2d;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #2d2d2d;
            }
            QPushButton:disabled {
                background-color: #333333;
                color: #888888;
            }
        """)
        self.layout.addWidget(self.prepare_experiment)

        self.start_acquisition_button = QPushButton("Start Real Time Experiment")
        self.start_acquisition_button.clicked.connect(self.start_or_stop_acquisition)
        self.start_acquisition_button.setEnabled(False)
        self.start_acquisition_button.setStyleSheet("""
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
            QPushButton:disabled {
                background-color: #CCCCCC;
                color: #888888;
            }
        """)
        self.layout.addWidget(self.start_acquisition_button)

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
            self.prepare_experiment.setEnabled(True)

        except Exception as e:
            self.loaded_pipeline = None
            self.pipeline_label.setText("No pipeline file selected")
            self.pipeline_label.setStyleSheet("padding: 8px; color: #aaaaaa; font-weight: normal;")
            self.prepare_experiment.setEnabled(False)

            tb = traceback.format_exc()
            self.status_label.setText(f"Error loading pipeline: {e}")
            self.status_label.setStyleSheet("color: #f44336; font-weight: bold; padding: 4px;")
            print(f"Error loading pipeline:\n{tb}")

    def on_show(self):
        pass

    def on_hide(self):
        if self.prompt_viewer and not self.prompt_viewer.closed:
            self.prompt_viewer.close()

    def open_prompt_window(self):
        self.prepare_experiment.setEnabled(False)
        self.start_acquisition_button.setEnabled(True)
        # Dummy long-running queue item
        self.queue = [["Initializing...", 999999.0]]
        self.current_queue = [["Initializing...", 999999.0]]
        self.prompt_viewer = PromptViewer(self, self.start_or_stop_acquisition, self.on_prompt_viewer_close, self.progressbar_value)
        self.prompt_viewer.show()

    def on_prompt_viewer_close(self):
        self.prepare_experiment.setEnabled(True)
        self.start_acquisition_button.setEnabled(False)
        self.start_acquisition_button.setText("Start Real Time Experiment")
        self.start_acquisition_button.setStyleSheet("""
            QPushButton {
                padding: 10px;
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #0b7dda; }
            QPushButton:disabled { background-color: #CCCCCC; color: #888888; }
        """)
        self.is_running = False
        if self.current_queue is not None:
            self.current_queue.clear()
        self.queue = None
        self.current_queue = None

    def update_experiment_timeline_plot(self, value):
        pass

    def start_or_stop_acquisition(self):
        if not self.is_running:
            self.is_running = True
            self.start_acquisition_button.setText("Stop Real Time Experiment")
            self.start_acquisition_button.setStyleSheet("""
                QPushButton {
                    padding: 10px;
                    background-color: #f44336;
                    color: white;
                    border: none;
                    border-radius: 4px;
                }
                QPushButton:hover { background-color: #d32f2f; }
            """)
            self.acquisition_thread = Thread(target=self.acquisition, daemon=True)
            self.acquisition_thread.start()
        else:
            self.is_running = False
            self.start_acquisition_button.setText("Start Real Time Experiment")
            self.start_acquisition_button.setEnabled(False)
            if self.current_queue is not None:
                self.current_queue.clear()
            if self.prompt_viewer and not self.prompt_viewer.closed:
                self.prompt_viewer.change_prompt('Stopped')

    def acquisition(self):
        prompt_viewer_proxy = ThreadSafePromptViewerProxy(self.prompt_viewer)
        timeline_signaler = ThreadSafeTimelineSignaler(self.update_experiment_timeline_plot)
        batches_per_second = 10
        
        try:
            self.bci_interface.run_acquisition(
                prompt_viewer_proxy,
                self.current_queue,
                timeline_signaler,
                self.progressbar_value,
                batches_per_second,
                real_time_processor=self.real_time_processor
            )
        except Exception as e:
            print("Acquisition error:", e)

    def real_time_processor(self, current_signal):
        if not self.loaded_pipeline:
            return "No pipeline loaded"
            
        sampling_rate = self.configurations.read("general.sampling_rate")
        trial_length = self.configurations.read("real_time.trial_length")
        required_samples = int(sampling_rate * trial_length)
        
        if len(current_signal[0]) < required_samples:
            return "Gathering data..."
            
        # Extract last `required_samples` from all 32 channels
        # GtecNautilusProInterface fills non-selected channels with zeros, so the shape is exactly 32.
        raw_data = np.array([ch[-required_samples:] for ch in current_signal])
        
        # Apply bandpass filter (2-36 Hz) as used in training (moabb MotorImagery)
        # Filter silently to avoid console spam
        filtered_data = mne.filter.filter_data(raw_data, sfreq=sampling_rate, l_freq=2.0, h_freq=36.0, verbose=False)
        
        # Predict using pipeline (expects shape (n_trials, channels, times))
        X_test = np.expand_dims(filtered_data, axis=0)
        
        try:
            prediction = self.loaded_pipeline.predict(X_test)[0]
            return str(prediction).capitalize()
        except Exception as e:
            return "Error predicting"
