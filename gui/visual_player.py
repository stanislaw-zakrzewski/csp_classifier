import os
from PySide6.QtWidgets import QWidget, QLabel, QStackedLayout
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput, QVideoFrame
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtGui import QPixmap
from PySide6.QtCore import QUrl, Qt

class Screen(QWidget):
    """
    Screen widget: Embedded PySide6 native media player.
    Uses dual QMediaPlayer + QVideoWidget instances to achieve 100% seamless
    transitions (zero black frames) when switching between videos.
    """
    def __init__(self, parent=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.layout = QStackedLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        # 1. Image view (index 0)
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: black;")
        self.layout.addWidget(self.image_label)
        
        # 2. Player 1 (index 1)
        self.video_widget_1 = QVideoWidget(self)
        self.video_widget_1.setStyleSheet("background-color: black;")
        self.layout.addWidget(self.video_widget_1)
        
        self.media_player_1 = QMediaPlayer(self)
        self.audio_output_1 = QAudioOutput(self)
        self.media_player_1.setAudioOutput(self.audio_output_1)
        self.media_player_1.setVideoOutput(self.video_widget_1)
        self.media_player_1.setLoops(QMediaPlayer.Infinite)
        
        # 3. Player 2 (index 2)
        self.video_widget_2 = QVideoWidget(self)
        self.video_widget_2.setStyleSheet("background-color: black;")
        self.layout.addWidget(self.video_widget_2)
        
        self.media_player_2 = QMediaPlayer(self)
        self.audio_output_2 = QAudioOutput(self)
        self.media_player_2.setAudioOutput(self.audio_output_2)
        self.media_player_2.setVideoOutput(self.video_widget_2)
        self.media_player_2.setLoops(QMediaPlayer.Infinite)
        
        # Playback transition states
        self.active_player_index = 1  # Currently active/visible video player (1 or 2)
        self.loading_player_index = None  # Player index currently loading/buffering
        
        # Connect video sink signals to trigger seamless swaps when the first frame is ready
        self.media_player_1.videoSink().videoFrameChanged.connect(lambda: self.on_video_frame_changed(1))
        self.media_player_2.videoSink().videoFrameChanged.connect(lambda: self.on_video_frame_changed(2))
        
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
            # Stop video playback on both players, clear their sinks, and switch to image view
            self.media_player_1.stop()
            self.media_player_2.stop()
            self.media_player_1.videoSink().setVideoFrame(QVideoFrame())
            self.media_player_2.videoSink().setVideoFrame(QVideoFrame())
            self.loading_player_index = None
            
            self._current_image_path = filepath
            self.layout.setCurrentIndex(0)
            self._update_image_display()
        else:
            self._current_image_path = None
            # Target the inactive player to load the new video silently
            inactive_player_idx = 2 if self.active_player_index == 1 else 1
            inactive_player = self.media_player_2 if inactive_player_idx == 2 else self.media_player_1
            
            self.loading_player_index = inactive_player_idx
            
            # Clear any stale cached frame on the player we are about to use
            inactive_player.videoSink().setVideoFrame(QVideoFrame())
            
            inactive_player.setSource(QUrl.fromLocalFile(filepath))
            inactive_player.play()

    def stop(self):
        # Stop and clear everything, reset to rest image
        self.media_player_1.stop()
        self.media_player_2.stop()
        self.media_player_1.videoSink().setVideoFrame(QVideoFrame())
        self.media_player_2.videoSink().setVideoFrame(QVideoFrame())
        self.play('rest')

    def on_video_frame_changed(self, player_index):
        if player_index == self.loading_player_index:
            # First frame of the new video is decoded and ready!
            # Swap visibility to this player's widget index
            widget_index = 1 if player_index == 1 else 2
            self.layout.setCurrentIndex(widget_index)
            
            # Stop the previous player and purge its last frame so it doesn't leak/flash later
            old_player = self.media_player_2 if player_index == 1 else self.media_player_1
            old_player.stop()
            old_player.videoSink().setVideoFrame(QVideoFrame())
            
            self.active_player_index = player_index
            self.loading_player_index = None

    def _update_image_display(self):
        if self._current_image_path and os.path.exists(self._current_image_path):
            pixmap = QPixmap(self._current_image_path)
            if not pixmap.isNull():
                scaled_size = self.size()
                if scaled_size.width() <= 1 or scaled_size.height() <= 1:
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
        self.media_player_1.stop()
        self.media_player_2.stop()
        self.media_player_1.videoSink().setVideoFrame(QVideoFrame())
        self.media_player_2.videoSink().setVideoFrame(QVideoFrame())