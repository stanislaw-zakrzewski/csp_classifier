import os
from PySide6.QtWidgets import QWidget, QLabel, QStackedLayout
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtGui import QPixmap
from PySide6.QtCore import QUrl, Qt

class Screen(QWidget):
    """
    Screen widget: Embedded PySide6 native media player using QMediaPlayer 
    and QVideoWidget/QLabel for seamless video looping and image viewing.
    """
    def __init__(self, parent=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.layout = QStackedLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        # Image view (index 0)
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: black;")
        self.layout.addWidget(self.image_label)
        
        # Video view (index 1)
        self.video_widget = QVideoWidget(self)
        self.video_widget.setStyleSheet("background-color: black;")
        self.layout.addWidget(self.video_widget)
        
        # Setup Qt native Multimedia player
        self.media_player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.media_player.setAudioOutput(self.audio_output)
        self.media_player.setVideoOutput(self.video_widget)
        self.media_player.setLoops(QMediaPlayer.Infinite)  # Loop infinitely
        
        self.media_map = {
            'rest': 'commands/visual_commands/rest.png',
            'break': 'commands/visual_commands/pause.jpg',
            'movement': 'commands/visual_commands/movement.mov'
        }
        
        self._current_image_path = None
        self.play('rest')

    def play(self, _source):
        filepath = self.media_map.get(_source, _source)
        filepath = os.path.abspath(filepath)
        ext = os.path.splitext(filepath)[1].lower()
        
        if ext in ['.png', '.jpg', '.jpeg']:
            # Stop any playing video
            self.media_player.stop()
            self._current_image_path = filepath
            self.layout.setCurrentIndex(0)
            self._update_image_display()
        else:
            self._current_image_path = None
            self.layout.setCurrentIndex(1)
            self.media_player.setSource(QUrl.fromLocalFile(filepath))
            self.media_player.play()

    def stop(self):
        self.play('rest')

    def _update_image_display(self):
        if self._current_image_path and os.path.exists(self._current_image_path):
            pixmap = QPixmap(self._current_image_path)
            if not pixmap.isNull():
                scaled_size = self.size()
                if scaled_size.width() <= 1 or scaled_size.height() <= 1:
                    # Fallback to pixmap's original size if widget size is not initialized yet
                    scaled_size = pixmap.size()
                self.image_label.setPixmap(pixmap.scaled(
                    scaled_size, 
                    Qt.KeepAspectRatio, 
                    Qt.SmoothTransformation
                ))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.layout.currentIndex() == 0:
            self._update_image_display()

    def terminate(self):
        self.media_player.stop()