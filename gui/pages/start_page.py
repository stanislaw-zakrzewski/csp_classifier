from PySide6.QtWidgets import QWidget, QLabel, QPushButton, QGridLayout, QVBoxLayout, QScrollArea
from PySide6.QtCore import Qt
from gui.colors import colors
from gui.fonts import fonts

class StartPage(QScrollArea):
    def __init__(self, parent, controller, pages):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setStyleSheet(f"background-color: {colors['white_smoke']}; border: none;")
        
        content_widget = QWidget()
        self.setWidget(content_widget)
        
        layout = QVBoxLayout(content_widget)
        layout.setAlignment(Qt.AlignTop)
        
        app_title = QLabel("Kombajn EEG")
        app_title.setFont(fonts['large_bold_font'])
        app_title.setStyleSheet("margin: 10px; border: none;")
        layout.addWidget(app_title)
        
        grid_widget = QWidget()
        grid_layout = QGridLayout(grid_widget)
        grid_layout.setContentsMargins(10, 10, 10, 10)
        grid_layout.setSpacing(15)
        
        GRID_COLUMN_THRESHOLD = 2
        current_grid_row = 0
        current_grid_column = 0
        
        self.buttons = {}
        for page in pages:
            frame_class = page['frame']
            btn = QPushButton(page['name'])
            btn.setFont(fonts['large_font'])
            btn.setStyleSheet("""
                QPushButton {
                    background-color: white;
                    border: 2px solid #CCCCCC;
                    border-radius: 8px;
                    padding: 20px;
                    min-width: 250px;
                }
                QPushButton:hover {
                    background-color: #EAEAEA;
                    border-color: #A5A5A5;
                }
                QPushButton:pressed {
                    background-color: #CCCCCC;
                }
            """)
            
            # Use default parameter in lambda to capture loop variables correctly
            btn.clicked.connect(lambda checked=False, f=frame_class: controller.show_frame(f))
            self.buttons[page['name']] = btn
            
            grid_layout.addWidget(btn, current_grid_row, current_grid_column)
            current_grid_column += 1
            if current_grid_column >= GRID_COLUMN_THRESHOLD:
                current_grid_row += 1
                current_grid_column = 0
                
        layout.addWidget(grid_widget)
