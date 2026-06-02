import copy
from PySide6.QtWidgets import (QDialog, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, 
                             QLineEdit, QListWidget, QScrollArea, QFormLayout)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

from config.config import Configurations
from config.configuration_label_dictionary import configuration_label_dictionary
from gui.colors import colors
from gui.fonts import fonts

class Settings(QDialog):
    def __init__(self, root):
        super().__init__(root)
        self.configurations = Configurations()
        self.setWindowTitle("Settings")
        self.resize(600, 800)
        
        # Main dialog layout
        dialog_layout = QVBoxLayout(self)
        dialog_layout.setContentsMargins(10, 10, 10, 10)
        
        # Scroll Area
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet(f"background-color: {colors['white_smoke']}; border: none;")
        dialog_layout.addWidget(self.scroll_area)
        
        scroll_content = QWidget()
        self.scroll_area.setWidget(scroll_content)
        
        self.form_layout = QFormLayout(scroll_content)
        self.form_layout.setSpacing(10)
        
        self.values = {}
        self.inputs = {}
        self.new_values = {}
        
        # Create settings fields recursively
        self.create_group(self.configurations.default_configuration)
        
        # Load values into the inputs
        self.restore_configuration(self.configurations.current_configuration)
        
        # Actions bar
        actions_widget = QWidget()
        actions_layout = QHBoxLayout(actions_widget)
        actions_layout.setContentsMargins(0, 10, 0, 0)
        dialog_layout.addWidget(actions_widget)
        
        self.discard_changes_button = QPushButton('Discard changes')
        self.discard_changes_button.clicked.connect(
            lambda: self.restore_configuration(self.configurations.current_configuration)
        )
        actions_layout.addWidget(self.discard_changes_button)
        
        self.restore_default_button = QPushButton('Restore default configuration')
        self.restore_default_button.clicked.connect(
            lambda: self.restore_configuration(self.configurations.default_configuration)
        )
        actions_layout.addWidget(self.restore_default_button)
        
        self.save_configuration_button = QPushButton('Save configuration')
        self.save_configuration_button.setStyleSheet("background-color: #4CAF50; color: white; padding: 6px;")
        self.save_configuration_button.clicked.connect(self.save_current_configuration)
        actions_layout.addWidget(self.save_configuration_button)

    def get_value_from_configuration(self, configuration, key):
        try:
            path = key.split('.')
            if len(path) == 1:
                return configuration[key]['value']
            else:
                return self.get_value_from_configuration(configuration[path[0]], '.'.join(path[1:]))
        except KeyError:
            return None

    def set_value_to_configuration(self, configuration, key, value):
        try:
            path = key.split('.')
            if len(path) == 1:
                if configuration[key]['type'] == 'list':
                    if isinstance(value, str):
                        # Clean up string representation if it comes as list string
                        configuration[key]['value'] = value[1:-1].replace("'", '').split(', ')
                    else:
                        configuration[key]['value'] = value
                else:
                    configuration[key]['value'] = value
                return True
            else:
                return self.set_value_to_configuration(configuration[path[0]], '.'.join(path[1:]), value)
        except KeyError:
            return False

    def restore_configuration(self, configuration):
        for key in self.inputs:
            value = self.get_value_from_configuration(configuration, key)
            if value is None:
                value = self.get_value_from_configuration(self.configurations.default_configuration, key)
            
            self.values[key] = copy.deepcopy(value)
            
            # Update the Qt input widget
            widget = self.inputs[key]
            if isinstance(widget, QLineEdit):
                widget.setText(str(value))
            elif isinstance(widget, QListWidget):
                widget.clear()
                widget.addItems(value if isinstance(value, list) else [value])

    def save_current_configuration(self):
        new_configuration = copy.deepcopy(self.configurations.default_configuration)
        for key in self.inputs:
            widget = self.inputs[key]
            if isinstance(widget, QLineEdit):
                val = widget.text()
            elif isinstance(widget, QListWidget):
                val = [widget.item(i).text() for i in range(widget.count())]
            else:
                val = self.values[key]
            self.set_value_to_configuration(new_configuration, key, val)
            
        self.configurations.change_current_configuration(new_configuration)
        self.accept()  # Close the dialog on save

    @staticmethod
    def get_text_with_depth(depth, text):
        depth_text_value = ''
        for _ in range(depth):
            depth_text_value += '  '
        return depth_text_value + text

    def create_text_entry(self, final_key, label_widget):
        entry = QLineEdit()
        self.inputs[final_key] = entry
        self.form_layout.addRow(label_widget, entry)

    def create_list_entry(self, final_key, label_widget, values):
        f = QWidget()
        layout = QHBoxLayout(f)
        layout.setContentsMargins(0, 0, 0, 0)
        
        box = QListWidget()
        box.setMaximumHeight(150)
        layout.addWidget(box)
        
        btn_panel = QWidget()
        btn_layout = QVBoxLayout(btn_panel)
        btn_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(btn_panel)
        
        new_val_input = QLineEdit()
        btn_layout.addWidget(new_val_input)
        
        def add():
            txt = new_val_input.text().strip()
            if txt:
                box.addItem(txt)
                new_val_input.clear()
                
        add_btn = QPushButton('Add')
        add_btn.clicked.connect(add)
        btn_layout.addWidget(add_btn)
        
        def remove():
            for item in box.selectedItems():
                box.takeItem(box.row(item))
                
        rem_btn = QPushButton('Remove Selected')
        rem_btn.clicked.connect(remove)
        btn_layout.addWidget(rem_btn)
        
        self.inputs[final_key] = box
        self.form_layout.addRow(label_widget, f)

    def create_field(self, key, depth, structure_path, val):
        final_key = ".".join([*structure_path, key])
        label_text = self.get_text_with_depth(depth, self.get_label(key))
        
        lbl = QLabel(label_text)
        lbl.setFont(fonts['medium_bold'] if depth <= 0 else fonts['medium_bold'])
        
        if isinstance(val['value'], list):
            self.create_list_entry(final_key, lbl, val['value'])
        else:
            self.create_text_entry(final_key, lbl)

    def get_depth(self, x):
        if isinstance(x, dict) and x:
            if 'type' in x.keys():
                return 1
            return 1 + max(self.get_depth(x[a]) for a in x)
        if isinstance(x, list) and x:
            return 1 + max(self.get_depth(a) for a in x)
        return 0

    def get_label(self, key):
        try:
            return configuration_label_dictionary[key]
        except KeyError:
            return key

    def create_group(self, group_data, structure_path=None, current_depth=-1, current_key=None):
        if current_key is not None:
            title_lbl = QLabel(self.get_text_with_depth(current_depth, self.get_label(current_key)))
            f = QFont("Segoe UI", 9)
            f.setBold(True)
            title_lbl.setFont(f)
            title_lbl.setStyleSheet("margin-top: 10px; margin-bottom: 5px;")
            self.form_layout.addRow(title_lbl)

        if structure_path is None:
            structure_path = []

        if self.get_depth(group_data) > 2:
            for key in group_data:
                self.create_group(group_data[key], [*structure_path, key], current_depth + 1, key)
        else:
            for key in group_data:
                self.create_field(key, current_depth, structure_path, group_data[key])
