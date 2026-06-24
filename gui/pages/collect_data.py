import copy
import random
import math
from threading import Thread
import numpy as np

from PySide6.QtWidgets import (QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, 
                             QLineEdit, QScrollArea, QProgressBar, QCheckBox)
from PySide6.QtCore import Qt, Signal, QObject
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from config.config import Configurations
from gui.fonts import fonts
from gui.colors import colors
from gui.pages.collect_data_components.prompt_viewer import PromptViewer
from gui.pages.start_page import StartPage
from src.bci_integration.GtecNautilusProInterface import GtecNautilusProInterface
from src.bci_integration.ZeroMockBCIInterface import ZeroMockBCIInterface
from src.edf.EDFWriter import EDFWriter
from gui.components.back_button import BackButton
from gui.components.title_label import TitleLabel

class ThreadSafePromptViewerProxy(QObject):
    change_prompt_signal = Signal(str)

    def __init__(self, prompt_viewer):
        super().__init__()
        self.prompt_viewer = prompt_viewer
        self.change_prompt_signal.connect(self.prompt_viewer.change_prompt)

    def change_prompt(self, prompt_code):
        self.change_prompt_signal.emit(prompt_code)

    @property
    def closed(self):
        return self.prompt_viewer.closed

    def destroy(self):
        pass

class ThreadSafeTimelineSignaler(QObject):
    update_signal = Signal(float)

    def __init__(self, update_callback):
        super().__init__()
        self.update_callback = update_callback
        self.update_signal.connect(self.update_callback)

    def __call__(self, val):
        self.update_signal.emit(val)

class CollectDataSignaler(QObject):
    acquisition_finished = Signal(object, object)

class CollectData(QScrollArea):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setStyleSheet(f"background-color: {colors['white_smoke']}; border: none;")

        self.configurations = Configurations()
        self.batches_per_second = 10
        self.queue = None
        self.current_queue = None
        self.bci_interface = GtecNautilusProInterface()
        self.edf_writer = EDFWriter()
        self.progressbar_value = 0.0
        self.is_mock_enabled = False

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

        # Patient Info Form
        form_layout = QVBoxLayout()
        self.layout.addLayout(form_layout)

        name_row = QHBoxLayout()
        name_lbl = QLabel("Patient Name:")
        name_lbl.setStyleSheet("color: white;")
        name_row.addWidget(name_lbl)
        self.patient_name_input = QLineEdit()
        self.patient_name_input.setText(self.configurations.read('collect_data.patient_name'))
        name_row.addWidget(self.patient_name_input)
        form_layout.addLayout(name_row)

        gender_row = QHBoxLayout()
        gender_lbl = QLabel("Patient Gender:")
        gender_lbl.setStyleSheet("color: white;")
        gender_row.addWidget(gender_lbl)
        self.gender_input = QLineEdit()
        self.gender_input.setText(self.configurations.read('collect_data.patient_gender'))
        gender_row.addWidget(self.gender_input)
        form_layout.addLayout(gender_row)

        # Controls
        self.mock_checkbox = QCheckBox("Use mocked data")
        self.mock_checkbox.setStyleSheet(f"color: white; font-weight: bold;")
        self.mock_checkbox.stateChanged.connect(self.on_mock_toggled)
        self.layout.addWidget(self.mock_checkbox)

        self.prepare_experiment = QPushButton("Prepare experiment")
        self.prepare_experiment.clicked.connect(self.open_prompt_window)
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
        """)
        self.layout.addWidget(self.prepare_experiment)

        self.start_acquisition_button = QPushButton("Start Acquisition")
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

        # Signaler
        self.gui_signaler = CollectDataSignaler()
        self.gui_signaler.acquisition_finished.connect(self.on_acquisition_finished)

        self.acquisition_thread = None
        self.plot_canvas = None
        self.prompt_viewer = None
        self.fig = None
        self.gnt = None
        self.labels = None
        self.is_running = False

    def on_mock_toggled(self, state):
        self.is_mock_enabled = (state == Qt.Checked.value)

    def open_prompt_window(self):
        self.prepare_experiment.setEnabled(False)
        self.patient_name_input.setEnabled(False)
        self.gender_input.setEnabled(False)
        self.start_acquisition_button.setEnabled(True)
        self.create_queue()
        self._last_drawn_sec = -1
        self.prompt_viewer = PromptViewer(self, self.start_or_stop_acquisition, self.on_prompt_viewer_close, self.progressbar_value)
        self.prompt_viewer.show()
        self.update_experiment_timeline_plot(0)

    def on_prompt_viewer_close(self):
        self.prepare_experiment.setEnabled(True)
        self.patient_name_input.setEnabled(True)
        self.gender_input.setEnabled(True)
        self.start_acquisition_button.setEnabled(False)
        self.start_acquisition_button.setText("Start Acquisition")
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
        self.is_running = False
        self.queue = None
        self.current_queue = None
        
        if self.plot_canvas is not None:
            self.plot_layout.removeWidget(self.plot_canvas)
            self.plot_canvas.deleteLater()
            self.plot_canvas = None
            
        self.fig = None
        self.gnt = None

    def update_experiment_timeline_plot(self, value):
        if self.current_queue is None:
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
        self.gnt.set_xlim(0, 40)
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
            self.start_acquisition_button.setText("Stop Acquisition")
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
            self.start_acquisition_button.setText("Start Acquisition")
            self.start_acquisition_button.setEnabled(False)
            if self.current_queue is not None:
                self.current_queue.clear()
            if self.prompt_viewer and not self.prompt_viewer.closed:
                self.prompt_viewer.change_prompt('Stopped')

    def acquisition(self):
        prompt_viewer_proxy = ThreadSafePromptViewerProxy(self.prompt_viewer)
        timeline_signaler = ThreadSafeTimelineSignaler(self.update_experiment_timeline_plot)
        
        if self.is_mock_enabled:
            mock_interface = ZeroMockBCIInterface()
            recorded_signal, start_date = mock_interface.run_acquisition(
                prompt_viewer_proxy,
                self.current_queue,
                timeline_signaler,
                self.progressbar_value,
                self.batches_per_second
            )
        else:
            recorded_signal, start_date = self.bci_interface.run_acquisition(
                prompt_viewer_proxy,
                self.current_queue,
                timeline_signaler,
                self.progressbar_value,
                self.batches_per_second
            )
        
        self.gui_signaler.acquisition_finished.emit(recorded_signal, start_date)

    def on_acquisition_finished(self, recorded_signal, start_date):
        self.is_running = False
        self.start_acquisition_button.setText("Start Acquisition")
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

        if self.prompt_viewer and not self.prompt_viewer.closed:
            self.prompt_viewer.change_prompt('end')
            self.prompt_viewer.close()
            
        self.edf_writer.write(
            recorded_signal, 
            start_date, 
            self.queue, 
            self.patient_name_input.text(),
            self.gender_input.text()
        )

    def create_queue(self):
        trial_count = self.configurations.read('collect_data.trial_count')
        pause_length = self.configurations.read('collect_data.pause_length')
        trial_length = self.configurations.read('collect_data.trial_length')
        labels = self.configurations.read('collect_data.labels')
        label_queue = list(np.repeat(labels, trial_count))
        random.shuffle(label_queue)
        queue = []
        for trial_label in label_queue:
            queue.append(['break', pause_length])
            queue.append([trial_label, trial_length])
        self.queue = queue
        self.time_total_val = sum(list(map(lambda x: x[1], self.queue)))
        self.time_total_label.setText(f"Total time: {self.time_total_val}")
        self.current_queue = copy.deepcopy(queue)

    def on_show(self):
        # Reload latest configurations dynamically
        self.patient_name_input.setText(self.configurations.read('collect_data.patient_name'))
        self.gender_input.setText(self.configurations.read('collect_data.patient_gender'))

    def on_hide(self):
        # Close active overlay prompts dialog if visible
        if self.prompt_viewer and not self.prompt_viewer.closed:
            self.prompt_viewer.close()
