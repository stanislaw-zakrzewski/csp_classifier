from PySide6.QtWidgets import QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QScrollArea
from PySide6.QtCore import Qt

from gui.colors import colors
from gui.fonts import fonts
from gui.pages.start_page import StartPage
from gui.visual_player import Screen
from gui.components.back_button import BackButton
from gui.components.title_label import TitleLabel

PROMPTS = {
    'video': [
        {'label': 'Left', 'path': 'commands//visual_commands//left.mov'},
        {'label': 'Right', 'path': 'commands//visual_commands//right.mov'},
        {'label': 'Movement', 'path': 'commands//visual_commands//movement.mov'}],
    'image': [
        {'label': 'Rest', 'path': 'commands//visual_commands//rest.png'},
        {'label': 'Pause', 'path': 'commands//visual_commands//pause.jpg'}],
}

class PromptViewerState:
    def __init__(self):
        self.current_prompt = None

    def set_prompt(self, new_prompt):
        self.current_prompt = new_prompt

    def get_prompt(self):
        return self.current_prompt

class PromptViewer(QScrollArea):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setStyleSheet(f"background-color: {colors['white_smoke']}; border: none;")

        self.prompt_viewer_state = PromptViewerState()
        self.player = None

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

        # Controls HBox
        controls_hbox = QHBoxLayout()
        controls_hbox.setAlignment(Qt.AlignLeft)
        self.layout.addLayout(controls_hbox)

        # Video prompts
        video_label = QLabel("Video:")
        video_label.setFont(fonts['medium_bold'])
        video_label.setStyleSheet("color: white;")
        controls_hbox.addWidget(video_label)

        for video_prompt in PROMPTS['video']:
            btn = QPushButton(video_prompt['label'])
            btn.setStyleSheet("padding: 8px 15px;")
            btn.clicked.connect(
                lambda checked=False, p=video_prompt['path']: self.set_current_prompt('video', p)
            )
            controls_hbox.addWidget(btn)

        controls_hbox.addSpacing(20)

        # Image prompts
        image_label = QLabel("Image:")
        image_label.setFont(fonts['medium_bold'])
        image_label.setStyleSheet("color: white;")
        controls_hbox.addWidget(image_label)

        for image_prompt in PROMPTS['image']:
            btn = QPushButton(image_prompt['label'])
            btn.setStyleSheet("padding: 8px 15px;")
            btn.clicked.connect(
                lambda checked=False, p=image_prompt['path']: self.set_current_prompt('image', p)
            )
            controls_hbox.addWidget(btn)

        controls_hbox.addSpacing(20)

        # Clear button
        clear_btn = QPushButton("Clear")
        clear_btn.setStyleSheet("background-color: #ff9800; color: white; padding: 8px 15px; border: none; border-radius: 3px;")
        clear_btn.clicked.connect(self.clear_prompt)
        controls_hbox.addWidget(clear_btn)

    def set_current_prompt(self, prompt_type, prompt_source):
        if self.player is None:
            self.player = Screen(self)
            self.player.setFixedSize(500, 300)
            self.layout.addWidget(self.player)
            
        self.player.play(prompt_source)

    def clear_prompt(self):
        if self.player is not None:
            self.layout.removeWidget(self.player)
            self.player.terminate()
            self.player.deleteLater()
            self.player = None

    def on_hide(self):
        # Stop and clear prompt media player when leaving page
        self.clear_prompt()
