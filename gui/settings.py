import copy
from PySide6.QtWidgets import (QDialog, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, 
                             QLineEdit, QListWidget, QScrollArea, QFormLayout, QCheckBox)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

from config.config import Configurations
from config.configuration_label_dictionary import configuration_label_dictionary
from gui.colors import colors
from gui.fonts import fonts


class CollapsibleSection(QWidget):
    """
    Collapsible section container styled with a distinct border and background.
    """
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 12)
        self.layout.setSpacing(0)
        
        self.title_text = title
        self.is_collapsed = False
        
        # Toggle Button
        self.toggle_btn = QPushButton()
        self.toggle_btn.clicked.connect(self.toggle_collapsed)
        self.layout.addWidget(self.toggle_btn)
        
        # Content Widget
        self.content_widget = QWidget()
        self.content_widget.setObjectName("SectionContent")
        self.content_layout = QFormLayout(self.content_widget)
        self.content_layout.setContentsMargins(15, 12, 15, 12)
        self.content_layout.setSpacing(10)
        self.layout.addWidget(self.content_widget)
        
        # Content styling
        self.content_widget.setStyleSheet("""
            QWidget#SectionContent {
                background-color: #161616;
                border: 1px solid #2d2d2d;
                border-top: none;
                border-bottom-left-radius: 4px;
                border-bottom-right-radius: 4px;
            }
        """)
        self._update_button_style()

    def _update_button_style(self):
        arrow = "▶" if self.is_collapsed else "▼"
        self.toggle_btn.setText(f"{arrow}  {self.title_text}")
        
        if self.is_collapsed:
            self.toggle_btn.setStyleSheet("""
                QPushButton {
                    text-align: left;
                    font-weight: bold;
                    font-size: 13px;
                    padding: 10px 15px;
                    background-color: #1e1e1e;
                    border: 1px solid #2d2d2d;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #2a2a2a;
                    border-color: #3d3d3d;
                }
            """)
        else:
            self.toggle_btn.setStyleSheet("""
                QPushButton {
                    text-align: left;
                    font-weight: bold;
                    font-size: 13px;
                    padding: 10px 15px;
                    background-color: #1e1e1e;
                    border: 1px solid #2d2d2d;
                    border-top-left-radius: 4px;
                    border-top-right-radius: 4px;
                    border-bottom-left-radius: 0px;
                    border-bottom-right-radius: 0px;
                }
                QPushButton:hover {
                    background-color: #2a2a2a;
                    border-color: #3d3d3d;
                }
            """)

    def toggle_collapsed(self):
        self.is_collapsed = not self.is_collapsed
        self.content_widget.setVisible(not self.is_collapsed)
        self._update_button_style()


class Settings(QDialog):
    def __init__(self, root):
        super().__init__(root)
        self.configurations = Configurations()
        self.setWindowTitle("Settings")
        self.resize(600, 800)
        
        # Main dialog layout
        dialog_layout = QVBoxLayout(self)
        dialog_layout.setContentsMargins(15, 15, 15, 15)
        dialog_layout.setSpacing(10)
        
        # Title Label
        title_label = QLabel("Application Settings")
        title_label.setFont(QFont("Segoe UI", 16, QFont.Bold))
        title_label.setStyleSheet("padding-bottom: 5px; color: #ffffff;")
        dialog_layout.addWidget(title_label)
        
        # Scroll Area
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet(f"background-color: {colors['background']}; border: none;")
        dialog_layout.addWidget(self.scroll_area)
        
        scroll_content = QWidget()
        scroll_content.setStyleSheet(f"background-color: {colors['background']};")
        self.scroll_area.setWidget(scroll_content)
        
        # Vertical layout inside scroll area for sections
        self.main_scroll_layout = QVBoxLayout(scroll_content)
        self.main_scroll_layout.setContentsMargins(0, 0, 10, 0)
        self.main_scroll_layout.setSpacing(15)
        self.main_scroll_layout.setAlignment(Qt.AlignTop)
        
        self.values = {}
        self.inputs = {}
        
        # Populate sections dynamically based on top-level keys
        for key in self.configurations.default_configuration:
            section_title = self.get_label(key)
            section = CollapsibleSection(section_title, self)
            self.main_scroll_layout.addWidget(section)
            
            # Populate fields inside section
            self.populate_section(self.configurations.default_configuration[key], [key], section.content_layout)
            
        # Load values into the inputs
        self.restore_configuration(self.configurations.current_configuration)
        
        # Actions bar (bottom buttons)
        actions_widget = QWidget()
        actions_layout = QHBoxLayout(actions_widget)
        actions_layout.setContentsMargins(0, 15, 0, 0)
        actions_layout.setSpacing(12)
        dialog_layout.addWidget(actions_widget)
        
        # 1. Restore defaults button
        self.restore_default_button = QPushButton('Restore Defaults')
        self.restore_default_button.setStyleSheet("""
            QPushButton {
                background-color: #1e1e1e;
                color: #aaaaaa;
                border: 1px solid #2d2d2d;
            }
            QPushButton:hover {
                background-color: #2d2d2d;
                color: #ffffff;
            }
        """)
        self.restore_default_button.clicked.connect(
            lambda: self.restore_configuration(self.configurations.default_configuration)
        )
        actions_layout.addWidget(self.restore_default_button)
        
        # Spacer
        actions_layout.addStretch()
        
        # 2. Discard changes button
        self.discard_changes_button = QPushButton('Discard')
        self.discard_changes_button.setStyleSheet("""
            QPushButton {
                background-color: #1e1e1e;
                color: #f44336;
                border: 1px solid #2d2d2d;
            }
            QPushButton:hover {
                background-color: #2d2d2d;
                color: #f44336;
            }
        """)
        self.discard_changes_button.clicked.connect(
            lambda: self.restore_configuration(self.configurations.current_configuration)
        )
        actions_layout.addWidget(self.discard_changes_button)
        
        # 3. Save configuration button
        self.save_configuration_button = QPushButton('Save')
        self.save_configuration_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {colors['success']};
                color: white;
                border: none;
                font-weight: bold;
                padding: 8px 20px;
            }}
            QPushButton:hover {{
                background-color: {colors['success_hover']};
            }}
        """)
        self.save_configuration_button.clicked.connect(self.save_current_configuration)
        actions_layout.addWidget(self.save_configuration_button)

    def to_bool(self, val):
        if isinstance(val, bool):
            return val
        if isinstance(val, (int, float)):
            return bool(val)
        if isinstance(val, str):
            val_lower = val.strip().lower()
            return val_lower in ('true', '1', 'yes', 'on')
        return False

    def get_value_from_configuration(self, configuration, key):
        try:
            path = key.split('.')
            if len(path) == 1:
                return configuration[path[0]]
            else:
                return self.get_value_from_configuration(configuration[path[0]], '.'.join(path[1:]))
        except (KeyError, TypeError):
            return None

    def set_value_to_configuration(self, configuration, key, value):
        try:
            path = key.split('.')
            if len(path) == 1:
                configuration[path[0]] = value
                return True
            else:
                return self.set_value_to_configuration(configuration[path[0]], '.'.join(path[1:]), value)
        except (KeyError, TypeError):
            return False

    def restore_configuration(self, configuration):
        for key in self.inputs:
            value = self.get_value_from_configuration(configuration, key)
            if value is None:
                value = self.get_value_from_configuration(self.configurations.default_configuration, key)
            
            self.values[key] = copy.deepcopy(value)
            
            # Update the Qt input widget
            widget = self.inputs[key]
            if isinstance(widget, QCheckBox):
                widget.setChecked(self.to_bool(value))
            elif isinstance(widget, QLineEdit):
                widget.setText(str(value))
            elif isinstance(widget, QListWidget):
                widget.clear()
                widget.addItems(value if isinstance(value, list) else [value])

    def save_current_configuration(self):
        new_configuration = copy.deepcopy(self.configurations.default_configuration)
        for key in self.inputs:
            widget = self.inputs[key]
            
            default_val = self.get_value_from_configuration(self.configurations.default_configuration, key)
            
            if isinstance(widget, QCheckBox):
                val = widget.isChecked()
            elif isinstance(widget, QLineEdit):
                txt = widget.text().strip()
                if isinstance(default_val, bool):
                    val = self.to_bool(txt)
                elif isinstance(default_val, int):
                    try:
                        val = int(txt)
                    except ValueError:
                        val = 0
                elif isinstance(default_val, float):
                    try:
                        val = float(txt)
                    except ValueError:
                        val = 0.0
                else:
                    val = txt
            elif isinstance(widget, QListWidget):
                val = [widget.item(i).text() for i in range(widget.count())]
            else:
                val = self.values[key]
                
            self.set_value_to_configuration(new_configuration, key, val)
            
        self.configurations.change_current_configuration(new_configuration)
        self.accept()

    def populate_section(self, section_data, structure_path, form_layout):
        for key in section_data:
            val = section_data[key]
            final_key = ".".join([*structure_path, key])
            label_text = self.get_label(key)
            
            lbl = QLabel(label_text)
            lbl.setFont(fonts.get('medium_bold', QFont("Segoe UI", 9, QFont.Bold)))
            lbl.setStyleSheet("color: #aaaaaa;")
            
            if isinstance(val, bool):
                self.create_bool_entry(final_key, lbl, form_layout)
            elif isinstance(val, list):
                self.create_list_entry(final_key, lbl, val, form_layout)
            else:
                self.create_text_entry(final_key, lbl, form_layout)

    def create_text_entry(self, final_key, label_widget, form_layout):
        entry = QLineEdit()
        self.inputs[final_key] = entry
        form_layout.addRow(label_widget, entry)

    def create_bool_entry(self, final_key, label_widget, form_layout):
        checkbox = QCheckBox()
        checkbox.setStyleSheet("""
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
        """)
        self.inputs[final_key] = checkbox
        form_layout.addRow(label_widget, checkbox)

    def create_list_entry(self, final_key, label_widget, values, form_layout):
        f = QWidget()
        layout = QHBoxLayout(f)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        
        box = QListWidget()
        box.setMaximumHeight(100)
        layout.addWidget(box)
        
        btn_panel = QWidget()
        btn_layout = QVBoxLayout(btn_panel)
        btn_layout.setContentsMargins(0, 0, 0, 0)
        btn_layout.setSpacing(4)
        layout.addWidget(btn_panel)
        
        new_val_input = QLineEdit()
        new_val_input.setPlaceholderText("New item...")
        btn_layout.addWidget(new_val_input)
        
        def add():
            txt = new_val_input.text().strip()
            if txt:
                box.addItem(txt)
                new_val_input.clear()
                
        add_btn = QPushButton('Add')
        add_btn.setStyleSheet("""
            QPushButton {
                padding: 4px;
                font-size: 11px;
            }
        """)
        add_btn.clicked.connect(add)
        btn_layout.addWidget(add_btn)
        
        def remove():
            for item in box.selectedItems():
                box.takeItem(box.row(item))
                
        rem_btn = QPushButton('Remove')
        rem_btn.setStyleSheet("""
            QPushButton {
                padding: 4px;
                font-size: 11px;
            }
        """)
        rem_btn.clicked.connect(remove)
        btn_layout.addWidget(rem_btn)
        
        self.inputs[final_key] = box
        form_layout.addRow(label_widget, f)

    def get_label(self, key):
        try:
            return configuration_label_dictionary[key]
        except KeyError:
            return key
