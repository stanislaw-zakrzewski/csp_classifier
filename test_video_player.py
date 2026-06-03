import sys
import os
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
from PySide6.QtCore import QTimer, Qt

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from gui.visual_player import Screen

class VideoTestWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Prompt Viewer Video Test")
        self.resize(800, 600)
        self.setStyleSheet("background-color: #121212; color: #ffffff;")
        
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        
        # Heading
        self.status_label = QLabel("Click a button below to test video playback")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("font-size: 16px; font-weight: bold; padding: 10px;")
        layout.addWidget(self.status_label)
        
        # Embed the Screen player
        self.player = Screen(self)
        self.player.setMinimumHeight(400)
        layout.addWidget(self.player)
        
        # Controls HBox
        controls_layout = QHBoxLayout()
        layout.addLayout(controls_layout)
        
        # List of resources to test
        self.resources = [
            {"label": "Left Video", "path": "commands/visual_commands/left.mov"},
            {"label": "Right Video", "path": "commands/visual_commands/right.mov"},
            {"label": "Movement Video", "path": "commands/visual_commands/movement.mov"},
            {"label": "Rest Image", "path": "commands/visual_commands/rest.png"},
            {"label": "Pause Image", "path": "commands/visual_commands/pause.jpg"},
            {"label": "End Image", "path": "commands/visual_commands/end.jpg"}
        ]
        
        # Create buttons
        for item in self.resources:
            btn = QPushButton(item["label"])
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #1e1e1e;
                    color: #ffffff;
                    border: 1px solid #2d2d2d;
                    border-radius: 4px;
                    padding: 8px;
                }
                QPushButton:hover {
                    background-color: #2d2d2d;
                }
            """)
            btn.clicked.connect(lambda checked=False, path=item["path"], lbl=item["label"]: self.play_item(lbl, path))
            controls_layout.addWidget(btn)
            
        # Sequential play control
        self.seq_btn = QPushButton("Start Auto-Cycle (5s)")
        self.seq_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0b7dda;
            }
        """)
        self.seq_btn.clicked.connect(self.toggle_sequential)
        layout.addWidget(self.seq_btn)
        
        # Timer for sequential cycling
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.cycle_next)
        self.current_index = 0
        self.is_cycling = False
        
    def play_item(self, label, path):
        self.status_label.setText(f"Playing: {label} ({path})")
        # Check if the file exists
        if not os.path.exists(os.path.abspath(path)):
            self.status_label.setText(f"ERROR: File not found: {path}")
            return
        self.player.play(path)
        
    def toggle_sequential(self):
        if self.is_cycling:
            self.timer.stop()
            self.seq_btn.setText("Start Auto-Cycle (5s)")
            self.seq_btn.setStyleSheet("background-color: #2196F3; color: white; padding: 10px; font-weight: bold;")
            self.is_cycling = False
        else:
            self.current_index = 0
            self.cycle_next()
            self.timer.start(5000) # Switch every 5 seconds
            self.seq_btn.setText("Stop Auto-Cycle")
            self.seq_btn.setStyleSheet("background-color: #f44336; color: white; padding: 10px; font-weight: bold;")
            self.is_cycling = True
            
    def cycle_next(self):
        item = self.resources[self.current_index]
        self.play_item(item["label"], item["path"])
        self.current_index = (self.current_index + 1) % len(self.resources)
        
    def closeEvent(self, event):
        self.player.terminate()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoTestWindow()
    window.show()
    sys.exit(app.exec())
