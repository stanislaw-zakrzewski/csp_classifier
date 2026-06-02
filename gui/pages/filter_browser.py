import traceback
import math
from threading import Thread
import numpy as np
import pandas as pd

from PySide6.QtWidgets import (QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, 
                             QGridLayout, QLineEdit, QComboBox, QScrollArea, QFrame)
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

FILTER_NAMES = {
    'hp': 'high-pass',
    'lp': 'low-pass',
    'bp': 'band-pass',
    'bs': 'band-stop'
}

class FilterElectrodeCanvas(QWidget):
    def __init__(self, image_path, coordinates, parent=None):
        super().__init__(parent)
        self.pixmap = QPixmap(image_path)
        self.coordinates = coordinates
        self.highlighted_electrodes = set()

        if not self.pixmap.isNull():
            self.setFixedSize(self.pixmap.size())

    def update_electrode_colors(self, highlighted_list):
        self.highlighted_electrodes = set(highlighted_list)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        if not self.pixmap.isNull():
            painter.drawPixmap(0, 0, self.pixmap)

        r = 35
        for name in self.coordinates:
            if name in self.highlighted_electrodes:
                x, y = self.coordinates[name]
                pen = QPen(QColor("red"))
                pen.setWidth(10)
                painter.setPen(pen)
                painter.setBrush(Qt.NoBrush)
                painter.drawEllipse(QPoint(x, y), r, r)

class AcquisitionSignaler(QObject):
    update_ui = Signal(str, str, bool)

class FilterBrowser(QScrollArea):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setStyleSheet(f"background-color: {colors['white_smoke']}; border: none;")

        self.configurations = Configurations()
        self.selected_electrodes = self.configurations.read('general.selected_electrodes')
        self.filter_data = pd.read_csv('config/filters.csv')

        content_widget = QWidget()
        self.setWidget(content_widget)

        main_layout = QVBoxLayout(content_widget)
        main_layout.setAlignment(Qt.AlignTop)

        # Title
        app_title = QLabel("Kombajn EEG")
        app_title.setFont(fonts['large_bold_font'])
        app_title.setStyleSheet("margin: 10px; border: none;")
        main_layout.addWidget(app_title)

        # Back Button
        back_to_start_page_button = QPushButton("Back to Start Page")
        back_to_start_page_button.setFont(fonts['medium_font'])
        back_to_start_page_button.clicked.connect(lambda: controller.show_frame(StartPage))
        back_to_start_page_button.setStyleSheet("""
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
        main_layout.addWidget(back_to_start_page_button)

        # Content horizontal layout
        content_hbox = QHBoxLayout()
        main_layout.addLayout(content_hbox)

        # Left panel: Image Canvas
        self.electrodes_canvas = FilterElectrodeCanvas("gui/electrode_placement_filled.png", ELECTRODE_COORDINATES)
        content_hbox.addWidget(self.electrodes_canvas)

        # Right panel: Controls & Table
        right_panel = QWidget()
        self.right_layout = QVBoxLayout(right_panel)
        self.right_layout.setAlignment(Qt.AlignTop)
        content_hbox.addWidget(right_panel)

        self.table_title = QLabel("Filters")
        self.table_title.setFont(fonts['large_font'])
        self.right_layout.addWidget(self.table_title)

        # Table container
        self.table_container = QWidget()
        self.table_grid = QGridLayout(self.table_container)
        self.right_layout.addWidget(self.table_container)
        
        self.render_filter_table()

        # Add Button / Add Form layout area
        self.form_area = QWidget()
        self.form_layout = QVBoxLayout(self.form_area)
        self.form_layout.setContentsMargins(0, 0, 0, 0)
        self.right_layout.addWidget(self.form_area)

        self.add_filter_button = QPushButton("Add Filter")
        self.add_filter_button.setFont(fonts['medium_font'])
        self.add_filter_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.add_filter_button.clicked.connect(self.render_add_filter)
        self.form_layout.addWidget(self.add_filter_button)

        # Acquisition threads (preserved but not active in default UI)
        self.signaler = AcquisitionSignaler()
        self.signaler.update_ui.connect(self.on_update_ui)
        self.acquisition_thread = None
        self.acquisition_in_progress = False
        self.acquisition_initialized = False
        self.acquisition_stopped = False

    def render_filter_table(self):
        # Clear existing table layout
        for i in reversed(range(self.table_grid.count())): 
            widget = self.table_grid.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()

        headers = ['Name', 'Type', 'Frequency 1', 'Frequency 2', 'Steepness', 'Channels']
        for col_idx, text in enumerate(headers):
            lbl = QLabel(text)
            lbl.setFont(fonts['medium_bold'])
            self.table_grid.addWidget(lbl, 0, col_idx)

        for row_index, data_row in enumerate(self.filter_data.values):
            name_lbl = QLabel(str(data_row[0]))
            name_lbl.setFont(fonts['medium_font'])
            self.table_grid.addWidget(name_lbl, row_index + 1, 0)

            type_lbl = QLabel(FILTER_NAMES.get(data_row[1], str(data_row[1])))
            type_lbl.setFont(fonts['medium_font'])
            self.table_grid.addWidget(type_lbl, row_index + 1, 1)

            f1_lbl = QLabel(str(data_row[2]))
            f1_lbl.setFont(fonts['medium_font'])
            self.table_grid.addWidget(f1_lbl, row_index + 1, 2)

            f2_lbl = QLabel(str(data_row[3]))
            f2_lbl.setFont(fonts['medium_font'])
            self.table_grid.addWidget(f2_lbl, row_index + 1, 3)

            steep_lbl = QLabel(str(data_row[4]))
            steep_lbl.setFont(fonts['medium_font'])
            self.table_grid.addWidget(steep_lbl, row_index + 1, 4)

            ch_lbl = QLabel(str(data_row[5]))
            ch_lbl.setFont(fonts['medium_font'])
            self.table_grid.addWidget(ch_lbl, row_index + 1, 5)

            # Show Electrodes button
            show_btn = QPushButton("Show Electrodes")
            show_btn.setStyleSheet("background-color: #2196F3; color: white; border: none; padding: 5px; border-radius: 3px;")
            show_btn.clicked.connect(lambda checked=False, r=row_index: self.select_row(r))
            self.table_grid.addWidget(show_btn, row_index + 1, 6)

            # Delete button
            del_btn = QPushButton("DELETE")
            del_btn.setStyleSheet("background-color: #f44336; color: white; border: none; padding: 5px; border-radius: 3px;")
            del_btn.clicked.connect(lambda checked=False, r=row_index: self.remove_row(r))
            self.table_grid.addWidget(del_btn, row_index + 1, 7)

    def render_add_filter(self):
        # Clear form area
        for i in reversed(range(self.form_layout.count())): 
            widget = self.form_layout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()

        add_filter_form = QWidget()
        form_grid = QGridLayout(add_filter_form)
        self.form_layout.addWidget(add_filter_form)

        # Form fields
        form_grid.addWidget(QLabel('Name'), 0, 0)
        self.add_input_name = QLineEdit()
        form_grid.addWidget(self.add_input_name, 0, 1)

        # Band dropdowns
        form_grid.addWidget(QLabel('Band'), 1, 0)
        self.add_variable_band_specific = QComboBox()
        self.add_variable_band_specific.addItems(['whole', 'lower', 'upper', 'middle'])
        form_grid.addWidget(self.add_variable_band_specific, 1, 1)

        self.add_variable_band = QComboBox()
        self.add_variable_band.addItems(['alpha', 'beta', 'gamma'])
        form_grid.addWidget(self.add_variable_band, 1, 2)

        # Steepness dropdown
        form_grid.addWidget(QLabel('Steepness Select'), 2, 0)
        self.add_variable_band_steepness = QComboBox()
        self.add_variable_band_steepness.addItems(['steep', 'semi-steep', 'soft'])
        form_grid.addWidget(self.add_variable_band_steepness, 2, 1)

        # Numeric inputs
        form_grid.addWidget(QLabel('Frequency 1'), 3, 0)
        self.add_input_freq1 = QLineEdit("6")
        form_grid.addWidget(self.add_input_freq1, 3, 1)

        form_grid.addWidget(QLabel('Frequency 2'), 4, 0)
        self.add_input_freq2 = QLineEdit("14")
        form_grid.addWidget(self.add_input_freq2, 4, 1)

        form_grid.addWidget(QLabel('Steepness'), 5, 0)
        self.add_input_steepness = QLineEdit("15")
        form_grid.addWidget(self.add_input_steepness, 5, 1)
        form_grid.addWidget(QLabel('%'), 5, 2)

        form_grid.addWidget(QLabel('Channels'), 6, 0)
        self.add_input_channels = QLineEdit()
        form_grid.addWidget(self.add_input_channels, 6, 1)

        # Connect signals for dynamic changes
        self.add_variable_band_specific.currentTextChanged.connect(self.band_change_callback)
        self.add_variable_band.currentTextChanged.connect(self.band_change_callback)
        self.add_variable_band_steepness.currentTextChanged.connect(self.band_steepness_callback)

        # Action buttons
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet("background-color: #ff9800; color: white; padding: 8px;")
        cancel_btn.clicked.connect(self.render_add_button)
        form_grid.addWidget(cancel_btn, 7, 0)

        add_btn = QPushButton("Add")
        add_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 8px;")
        add_btn.clicked.connect(self.add_filter)
        form_grid.addWidget(add_btn, 7, 1)

    def band_change_callback(self):
        band = self.add_variable_band.currentText()
        specific = self.add_variable_band_specific.currentText()
        if band == 'alpha':
            if specific == 'whole':
                self.add_input_freq1.setText("6")
                self.add_input_freq2.setText("14")
            elif specific == 'lower':
                self.add_input_freq1.setText("6")
                self.add_input_freq2.setText("10")
            elif specific == 'upper':
                self.add_input_freq1.setText("10")
                self.add_input_freq2.setText("14")
            else:
                self.add_input_freq1.setText("8")
                self.add_input_freq2.setText("12")
        elif band == 'beta':
            if specific == 'whole':
                self.add_input_freq1.setText("15")
                self.add_input_freq2.setText("29")
            elif specific == 'lower':
                self.add_input_freq1.setText("15")
                self.add_input_freq2.setText("22")
            elif specific == 'upper':
                self.add_input_freq1.setText("22")
                self.add_input_freq2.setText("29")
            else:
                self.add_input_freq1.setText("18")
                self.add_input_freq2.setText("26")
        else: # gamma
            if specific == 'whole':
                self.add_input_freq1.setText("30")
                self.add_input_freq2.setText("40")
            elif specific == 'lower':
                self.add_input_freq1.setText("30")
                self.add_input_freq2.setText("35")
            elif specific == 'upper':
                self.add_input_freq1.setText("35")
                self.add_input_freq2.setText("40")
            else:
                self.add_input_freq1.setText("32")
                self.add_input_freq2.setText("38")

    def band_steepness_callback(self):
        steepness = self.add_variable_band_steepness.currentText()
        if steepness == 'steep':
            self.add_input_steepness.setText("15")
        elif steepness == 'semi-steep':
            self.add_input_steepness.setText("30")
        else:
            self.add_input_steepness.setText("50")

    def render_add_button(self):
        # Clear form area
        for i in reversed(range(self.form_layout.count())): 
            widget = self.form_layout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()

        self.add_filter_button = QPushButton("Add Filter")
        self.add_filter_button.setFont(fonts['medium_font'])
        self.add_filter_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.add_filter_button.clicked.connect(self.render_add_filter)
        self.form_layout.addWidget(self.add_filter_button)

    def add_filter(self):
        name = self.add_input_name.text()
        freq1 = self.add_input_freq1.text()
        freq2 = self.add_input_freq2.text()
        steepness = self.add_input_steepness.text()
        channels = self.add_input_channels.text()
        self.filter_data.loc[len(self.filter_data.index)] = [name, 'bp', freq1, freq2, steepness, channels]
        self.filter_data.to_csv('config/filters.csv', index=False)
        self.render_add_button()
        self.render_filter_table()

    def select_row(self, row_index):
        selected_row = self.filter_data.iloc[[row_index]].values[0]
        channels_str = str(selected_row[5])
        self.electrodes_canvas.update_electrode_colors(channels_str.split(' '))

    def remove_row(self, row_index):
        if not self.filter_data.empty:
            self.filter_data = self.filter_data.drop(self.filter_data.index[row_index])
        self.filter_data.to_csv('config/filters.csv', index=False)
        self.render_filter_table()

    def toggle_data(self):
        if not self.acquisition_initialized:
            self.acquisition_thread = Thread(target=self.run_acquisition, daemon=True)
            self.acquisition_thread.start()
            self.signaler.update_ui.emit('disable', 'Start', False)
        if self.acquisition_in_progress:
            self.acquisition_stopped = True
            self.signaler.update_ui.emit('disable', 'Stop', True)

    def on_update_ui(self, state, text, in_progress):
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
            return

        batches_per_second = 2

        def processCallback(samples):
            if self.acquisition_stopped:
                return False
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
            print("Acquisition Error:", e)
        finally:
            del d
