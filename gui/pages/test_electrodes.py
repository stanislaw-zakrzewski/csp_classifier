import traceback
import math
from threading import Thread
import numpy as np

from PySide6.QtWidgets import QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QScrollArea
from PySide6.QtCore import Qt, Signal, QPoint, QObject
from PySide6.QtGui import QPainter, QPixmap, QPen, QColor

import pygds
from config.config import Configurations
from gui.pages.start_page import StartPage
from gui.colors import colors
from gui.fonts import fonts

ELECTRODE_COORDINATES = {
    'Fp1': (479, 185),
    'FpZ': (600, 172),
    'Fp2': (726, 185),

    'AF7': (365, 245),
    'AF3': (463, 265),
    'AFZ': (600, 262),
    'AF4': (745, 266),
    'AF8': (842, 249),

    'F9': (208, 279),
    'F7': (294, 323),
    'F5': (368, 344),
    'F3': (446, 351),
    'F1': (524, 358),
    'FZ': (600, 363),
    'F2': (680, 360),
    'F4': (758, 351),
    'F6': (834, 341),
    'F8': (910, 323),
    'F10': (1001, 280),

    'FT9': (148, 414),
    'FT7': (240, 427),
    'FC5': (332, 435),
    'FC3': (425, 442),
    'FC1': (517, 449),
    'FCZ': (600, 451),
    'FC2': (687, 450),
    'FC4': (778, 442),
    'FC6': (876, 435),
    'FT8': (963, 426),
    'FT10': (1053, 414),

    'T9': (143, 544),
    'T7': (221, 544),
    'C5': (311, 543),
    'C3': (412, 544),
    'C1': (506, 547),
    'CZ': (600, 545),
    'C2': (689, 545),
    'C4': (792, 545),
    'C6': (886, 546),
    'T8': (982, 544),
    'T10': (1066, 542),

    'TP9': (143, 671),
    'TP7': (239, 652),
    'CP5': (336, 645),
    'CP3': (432, 642),
    'CP1': (517, 639),
    'CPZ': (600, 638),
    'CP2': (691, 639),
    'CP4': (777, 641),
    'CP6': (868, 645),
    'TP8': (964, 652),
    'TP10': (1060, 672),

    'P9': (196, 794),
    'P7': (288, 766),
    'P5': (366, 752),
    'P3': (446, 743),
    'P1': (526, 738),
    'PZ': (600, 737),
    'P2': (679, 738),
    'P4': (759, 742),
    'P6': (835, 751),
    'P8': (912, 766),
    'P10': (1007, 795),

    'PO7': (377, 853),
    'PO3': (473, 827),
    'POZ': (600, 824),
    'PO4': (722, 827),
    'PO8': (830, 853),

    'O1': (482, 906),
    'OZ': (600, 926),
    'O2': (722, 905),
}

class ElectrodeCanvas(QWidget):
    electrodeClicked = Signal(str)

    def __init__(self, image_path, coordinates, selected_electrodes, parent=None):
        super().__init__(parent)
        self.pixmap = QPixmap(image_path)
        self.coordinates = coordinates
        self.selected_electrodes = selected_electrodes
        self.active_electrode = None
        
        if not self.pixmap.isNull():
            self.setFixedSize(self.pixmap.size())

    def set_active_electrode(self, electrode_code):
        self.active_electrode = electrode_code
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        if not self.pixmap.isNull():
            painter.drawPixmap(0, 0, self.pixmap)
            
        r = 35
        for name in self.coordinates:
            if name in self.selected_electrodes:
                x, y = self.coordinates[name]
                pen = QPen(QColor("red"))
                if name == self.active_electrode:
                    pen.setWidth(10)
                else:
                    pen.setWidth(5)
                painter.setPen(pen)
                painter.setBrush(Qt.NoBrush)
                painter.drawEllipse(QPoint(x, y), r, r)

    def mousePressEvent(self, event):
        pos = event.position().toPoint() if hasattr(event, 'position') else event.pos()
        r = 35
        for name in self.coordinates:
            if name in self.selected_electrodes:
                x, y = self.coordinates[name]
                dx = pos.x() - x
                dy = pos.y() - y
                if math.sqrt(dx*dx + dy*dy) <= r:
                    self.electrodeClicked.emit(name)
                    break

class AcquisitionSignaler(QObject):
    update_ui = Signal(str, str, bool)  # state ('normal' or 'disable'), text, acquisition_in_progress

class TestElectrodes(QScrollArea):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setStyleSheet(f"background-color: {colors['white_smoke']}; border: none;")

        self.configurations = Configurations()
        self.selected_electrodes = self.configurations.read('general.selected_electrodes')
        self.selected_electrode_code = None

        content_widget = QWidget()
        self.setWidget(content_widget)

        main_layout = QVBoxLayout(content_widget)
        main_layout.setAlignment(Qt.AlignTop)

        # Header Title
        app_title = QLabel("Kombajn EEG")
        app_title.setFont(fonts['large_bold_font'])
        app_title.setStyleSheet("margin: 10px; border: none;")
        main_layout.addWidget(app_title)

        # Back Button
        self.back_button = QPushButton("Back to Start Page")
        self.back_button.setFont(fonts['medium_font'])
        self.back_button.clicked.connect(lambda: controller.show_frame(StartPage))
        self.back_button.setStyleSheet("""
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
        main_layout.addWidget(self.back_button)

        # Horizontal layout for drawing canvas and controls
        content_hbox = QHBoxLayout()
        main_layout.addLayout(content_hbox)

        # Electrode Placement Canvas
        self.canvas = ElectrodeCanvas("gui/electrode_placement_filled.png", ELECTRODE_COORDINATES, self.selected_electrodes)
        self.canvas.electrodeClicked.connect(self.on_electrode_clicked)
        content_hbox.addWidget(self.canvas)

        # Side controls panel
        controls_panel = QWidget()
        controls_layout = QVBoxLayout(controls_panel)
        controls_layout.setAlignment(Qt.AlignTop)
        content_hbox.addWidget(controls_panel)

        self.start_stop_button = QPushButton("Start")
        self.start_stop_button.setFont(fonts['medium_font'])
        self.start_stop_button.clicked.connect(self.toggle_data)
        self.start_stop_button.setStyleSheet("""
            QPushButton {
                background-color: #CCCCCC;
                border: 1px solid #A5A5A5;
                border-radius: 4px;
                padding: 15px 30px;
            }
            QPushButton:hover {
                background-color: #A5A5A5;
            }
        """)
        controls_layout.addWidget(self.start_stop_button)

        self.l1 = QLabel("Selected electrode:")
        self.l1.setFont(fonts['large_bold_font'])
        controls_layout.addWidget(self.l1)

        self.l = QLabel("")
        self.l.setFont(fonts['large_font'])
        controls_layout.addWidget(self.l)

        # Background Thread & Signaler Setup
        self.signaler = AcquisitionSignaler()
        self.signaler.update_ui.connect(self.on_update_ui)
        self.acquisition_thread = None
        self.acquisition_in_progress = False
        self.acquisition_initialized = False
        self.acquisition_stopped = False

    def on_electrode_clicked(self, name):
        if name == self.selected_electrode_code:
            self.selected_electrode_code = None
            self.l.setText("")
            self.canvas.set_active_electrode(None)
        else:
            self.selected_electrode_code = name
            self.l.setText(name)
            self.canvas.set_active_electrode(name)

    def toggle_data(self):
        if not self.acquisition_initialized:
            self.acquisition_thread = Thread(target=self.run_acquisition, daemon=True)
            self.acquisition_thread.start()
            self.start_stop_button.setEnabled(False)
        if self.acquisition_in_progress:
            self.acquisition_stopped = True
            self.start_stop_button.setEnabled(False)

    def on_update_ui(self, state, text, in_progress):
        self.start_stop_button.setEnabled(state == 'normal')
        self.start_stop_button.setText(text)
        self.acquisition_in_progress = in_progress
        if not in_progress:
            self.acquisition_initialized = False
            self.acquisition_stopped = False

    def run_acquisition(self):
        try:
            d = pygds.GDS()
            pygds.configure_demo(d)
            d.SetConfiguration()
        except Exception as e:
            print("Acquisition Init Error:", e)
            self.signaler.update_ui.emit('normal', 'Start', False)
            return

        batches_per_second = 2

        def processCallback(samples):
            if self.acquisition_stopped:
                self.signaler.update_ui.emit('normal', 'Start', False)
                return False
            if not self.acquisition_in_progress:
                self.signaler.update_ui.emit('normal', 'Stop', True)
            try:
                print(np.std(samples[:, [5, 15, 14, 13, 23, 9, 17, 18, 19, 27, 16]], axis=0))
                return True
            except Exception as e:
                print('ERROR:', e)
                print(traceback.format_exc())
                return False

        try:
            d.GetData(d.SamplingRate // batches_per_second, processCallback)
            d.Close()
        except Exception as e:
            print("Data Fetching Error:", e)
        finally:
            del d
