import os
import pickle
import traceback
import numpy as np
from threading import Thread

from PySide6.QtWidgets import QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog, QScrollArea, QCheckBox
from PySide6.QtCore import Qt, Signal, QObject

class ThreadSafePredictionSignaler(QObject):
    update_signal = Signal(str, str)

    def __init__(self, update_callback):
        super().__init__()
        self.update_callback = update_callback
        self.update_signal.connect(self.update_callback)

    def __call__(self, label, probs):
        self.update_signal.emit(label, probs)

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
from src.bci_integration.MockBCIInterface import MockBCIInterface
from gui.pages.collect_data import ThreadSafePromptViewerProxy, ThreadSafeTimelineSignaler
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import math

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
        
        self.mock_edf_path = ""
        self.is_mock_enabled = False
        self.fig = None
        self.gnt = None
        self.plot_canvas = None
        self.labels = None
        self.time_total_val = 0.0
        
        self.prediction_signaler = ThreadSafePredictionSignaler(self.update_prediction_labels)

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
        
        # Mock Data Toggle
        mock_row = QHBoxLayout()
        self.layout.addLayout(mock_row)
        
        self.mock_checkbox = QCheckBox("Use mocked data")
        self.mock_checkbox.setStyleSheet(f"color: {colors['text']}; font-weight: bold;")
        self.mock_checkbox.stateChanged.connect(self.on_mock_toggled)
        mock_row.addWidget(self.mock_checkbox)
        
        self.mock_select_btn = QPushButton("Select Mock EDF File")
        self.mock_select_btn.clicked.connect(self.select_mock_file)
        self.mock_select_btn.setEnabled(False)
        self.mock_select_btn.setStyleSheet(self.select_btn.styleSheet())
        mock_row.addWidget(self.mock_select_btn)
        
        self.mock_file_label = QLabel("No mock file selected")
        self.mock_file_label.setStyleSheet("padding: 8px; color: #aaaaaa; font-weight: normal;")
        mock_row.addWidget(self.mock_file_label)
        mock_row.addStretch()

        # Stats Labels
        stats_row = QHBoxLayout()
        self.layout.addLayout(stats_row)

        self.time_elapsed_label = QLabel("Elapsed time: 0")
        self.time_elapsed_label.setStyleSheet("color: white;")
        stats_row.addWidget(self.time_elapsed_label)

        self.time_total_label = QLabel("Total time: 0")
        self.time_total_label.setStyleSheet("color: white;")
        stats_row.addWidget(self.time_total_label)

        # Plot area
        self.plot_container = QWidget()
        self.plot_layout = QVBoxLayout(self.plot_container)
        self.layout.addWidget(self.plot_container)

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
        
        self.prediction_label = QLabel("Predicted Label: None")
        self.prediction_label.setStyleSheet("padding: 8px; color: #4CAF50; font-weight: bold; font-size: 16px;")
        self.prediction_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.prediction_label)
        
        self.probabilities_label = QLabel("Probabilities: None")
        self.probabilities_label.setStyleSheet("padding: 8px; color: white; font-weight: normal; font-size: 14px;")
        self.probabilities_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.probabilities_label)

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
            if not hasattr(pipeline, 'predict'):
                raise ValueError("Selected file is not a valid scikit-learn estimator (missing 'predict' method).")

            self.loaded_pipeline = pipeline

            if hasattr(pipeline, 'steps'):
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
            else:
                pipeline_name = pipeline.__class__.__name__

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

    def on_mock_toggled(self, state):
        self.is_mock_enabled = (state == Qt.Checked.value)
        self.mock_select_btn.setEnabled(self.is_mock_enabled)
        self.validate_prepare_button()
        
    def select_mock_file(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Select Mock EDF File", "", "European Data Format files (*.edf)"
        )
        if filename:
            self.mock_edf_path = filename
            self.mock_file_label.setText(os.path.basename(filename))
            self.validate_prepare_button()
            
    def validate_prepare_button(self):
        can_prepare = self.loaded_pipeline is not None
        if self.is_mock_enabled and not self.mock_edf_path:
            can_prepare = False
        self.prepare_experiment.setEnabled(can_prepare)

    def on_show(self):
        pass

    def on_hide(self):
        if self.prompt_viewer and not self.prompt_viewer.closed:
            self.prompt_viewer.close()

    def update_prediction_labels(self, label, probs_text):
        self.prediction_label.setText(f"Predicted Label: {label}")
        self.probabilities_label.setText(f"Probabilities: {probs_text}")

    def open_prompt_window(self):
        self.prepare_experiment.setEnabled(False)
        self.start_acquisition_button.setEnabled(True)
        
        if self.is_mock_enabled:
            # Parse EDF annotations
            raw = mne.io.read_raw_edf(self.mock_edf_path, preload=False, verbose=False)
            queue = []
            current_time = 0.0
            for annot in raw.annotations:
                onset = annot['onset']
                duration = annot['duration']
                desc = annot['description']
                
                # fill gaps
                if onset > current_time:
                    queue.append(['break', round(onset - current_time, 1)])
                queue.append([desc, round(duration, 1)])
                current_time = onset + duration
                
            self.queue = queue
            import copy
            self.current_queue = copy.deepcopy(queue)
            self.time_total_val = sum([item[1] for item in self.queue])
            self.time_total_label.setText(f"Total time: {round(self.time_total_val, 1)}")
        else:
            # Dummy long-running queue item
            self.queue = [["Initializing...", 999999.0]]
            self.current_queue = [["Initializing...", 999999.0]]
            self.time_total_val = 0.0
            self.time_total_label.setText("Total time: Infinite")
            
        self._last_drawn_sec = -1
        self.prompt_viewer = PromptViewer(self, self.start_or_stop_acquisition, self.on_prompt_viewer_close, self.progressbar_value)
        self.prompt_viewer.show()
        self.update_experiment_timeline_plot(0)

    def on_prompt_viewer_close(self):
        self.validate_prepare_button()
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
        
        if self.plot_canvas is not None:
            self.plot_layout.removeWidget(self.plot_canvas)
            self.plot_canvas.deleteLater()
            self.plot_canvas = None
            
        self.fig = None
        self.gnt = None

    def update_experiment_timeline_plot(self, value):
        if self.current_queue is None or not self.is_mock_enabled:
            return
            
        total_time_in_queue = sum(list(map(lambda x: x[1], self.current_queue)))
        elapsed = math.ceil((self.time_total_val - total_time_in_queue) * 10) / 10
        self.time_elapsed_label.setText(f"Elapsed time: {elapsed}")

        current_sec = int(elapsed)
        if hasattr(self, '_last_drawn_sec') and getattr(self, '_last_drawn_sec') == current_sec:
            if elapsed != 0:
                return
        self._last_drawn_sec = current_sec

        if self.fig is None:
            self.fig = Figure(figsize=(15, 6), facecolor='#121212')
        if self.gnt is None:
            self.gnt = self.fig.subplots()
            self.gnt.set_facecolor('#1e1e1e')
            self.gnt.set_xlabel('seconds since start', color='white')
            self.gnt.set_ylabel('Prompt', color='white')
            self.gnt.tick_params(colors='white')
            for spine in self.gnt.spines.values():
                spine.set_color('#2d2d2d')
        else:
            self.gnt.clear()
            self.gnt.set_facecolor('#1e1e1e')
            self.gnt.tick_params(colors='white')
            self.gnt.set_xlabel('seconds since start', color='white')
            self.gnt.set_ylabel('Prompt', color='white')
            for spine in self.gnt.spines.values():
                spine.set_color('#2d2d2d')

        # Prepare data
        data = {}
        previous_time_end = 0
        for item in self.current_queue:
            if item[0] not in data:
                data[item[0]] = []
            data[item[0]].append((previous_time_end, item[1]))
            previous_time_end += item[1]
            
        if self.labels is None:
            self.labels = list(data.keys())

        yticks = []
        for i in range(len(self.labels)):
            yticks.append(5 + i * 10)
        self.gnt.set_yticks(yticks)

        self.gnt.set_ylim(0, yticks[-1] + 5)
        self.gnt.set_xlim(0, 100)
        self.gnt.set_yticklabels(self.labels)
        self.gnt.grid(True, color='#2d2d2d')

        for index, item in enumerate(self.labels):
            if item in data:
                self.gnt.broken_barh(data[item], (index * 10, 9))

        if self.plot_canvas is None:
            self.plot_canvas = FigureCanvas(self.fig)
            self.plot_layout.addWidget(self.plot_canvas)
        else:
            self.plot_canvas.draw()

    def start_or_stop_acquisition(self):
        if not self.is_running:
            self.is_running = True
            self.total_predictions = 0
            self.correct_predictions = 0
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
            if self.is_mock_enabled:
                mock_interface = MockBCIInterface(self.mock_edf_path)
                mock_interface.run_acquisition(
                    prompt_viewer_proxy,
                    self.current_queue,
                    timeline_signaler,
                    self.progressbar_value,
                    batches_per_second,
                    real_time_processor=self.real_time_processor
                )
            else:
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
            
        # Clean up if finished
        if self.prompt_viewer and not self.prompt_viewer.closed:
            self.prompt_viewer.change_prompt('Finished')
            self.is_running = False
            self.start_acquisition_button.setText("Start Real Time Experiment")
            self.start_acquisition_button.setEnabled(False)

    def real_time_processor(self, current_signal, current_true_label=None):
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
            pred_str = str(prediction).lower()
            
            prob_str = "N/A"
            if hasattr(self.loaded_pipeline, "predict_proba"):
                probs = self.loaded_pipeline.predict_proba(X_test)[0]
                classes = self.loaded_pipeline.classes_
                prob_parts = []
                for cls, prob in zip(classes, probs):
                    prob_parts.append(f"{cls}: {prob:.2f}")
                prob_str = " | ".join(prob_parts)
                
            # Accuracy tracking
            accuracy_text = ""
            if self.is_mock_enabled and current_true_label is not None and hasattr(self.loaded_pipeline, "classes_"):
                true_label_norm = str(current_true_label).lower()
                classes_norm = [str(c).lower() for c in self.loaded_pipeline.classes_]
                
                if true_label_norm in classes_norm:
                    self.total_predictions += 1
                    if true_label_norm == pred_str:
                        self.correct_predictions += 1
                        
            if self.is_mock_enabled and hasattr(self, "total_predictions") and self.total_predictions > 0:
                acc = (self.correct_predictions / self.total_predictions) * 100
                accuracy_text = f" (Live Accuracy: {acc:.1f}% [{self.correct_predictions}/{self.total_predictions}])"
                
            if hasattr(self, 'prediction_signaler'):
                self.prediction_signaler(pred_str.capitalize() + accuracy_text, prob_str)
                
            return pred_str
        except Exception as e:
            return "error"
