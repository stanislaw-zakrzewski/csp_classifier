import os
from PySide6.QtWidgets import QDialog, QLabel, QVBoxLayout
from PySide6.QtCore import Qt, QUrl
from PySide6.QtMultimedia import QSoundEffect

from config.config import Configurations
from gui.visual_player import Screen
import SenderLib

class QtAudioCommands:
    def __init__(self):
        self.sounds = {}
        self._load_sound('left', 'commands/sound_commands/lewo.wav')
        self._load_sound('right', 'commands/sound_commands/prawo.wav')
        self._load_sound('rest', 'commands/sound_commands/brak.wav')
        self._load_sound('movement', 'commands/sound_commands/ruch.wav')
        self._load_sound('pause', 'commands/sound_commands/pauza.wav')
        self._load_sound('end', 'commands/sound_commands/koniec.wav')

    def _load_sound(self, name, file_path):
        abs_path = os.path.abspath(file_path)
        if os.path.exists(abs_path):
            effect = QSoundEffect()
            effect.setSource(QUrl.fromLocalFile(abs_path))
            self.sounds[name] = effect

    def perform_command(self, name):
        if name in self.sounds:
            self.sounds[name].play()

class PromptViewer(QDialog):
    def __init__(self, parent, start_command, close_command, progressbar_value):
        super().__init__(parent)
        self.configurations = Configurations()
        self.setWindowTitle("Browse annotations for")
        self.resize(1200, 720)
        self.setStyleSheet("background-color: black;")
        
        self.player = None
        self.current_prompt_code = None
        self.closed = False
        self.prompt_label = None
        self.progressbar_value = progressbar_value
        
        self.close_command = close_command
        
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        # Audio & Sender setup
        self.audio_commands = QtAudioCommands()
        self.sender = None
        self.control = None
        self.ipaddress = self.configurations.read('collect_data.ipaddress')
        self.port = self.configurations.read('collect_data.port')
        
        self.prompt_type = self.configurations.read('collect_data.prompt_type')
        if self.prompt_type == 'visual':
            self.change_prompt = self.set_visual_prompt
        elif self.prompt_type == 'audio':
            self.change_prompt = self.set_audio_prompt
        elif self.prompt_type == 'vr':
            self.sender = SenderLib.Sender(self.ipaddress, self.port)
            self.control = SenderLib.GameControl()
            self.change_prompt = self.set_vr_prompt
        else: # text prompt
            self.change_prompt = self.set_text_prompt

    def closeEvent(self, event):
        self.closed = True
        self.close_command()
        event.accept()

    def set_visual_prompt(self, prompt_code):
        if prompt_code == self.current_prompt_code:
            return
            
        if self.player is None:
            self.player = Screen(self)
            self.layout.addWidget(self.player)
            
        if prompt_code == 'movement':
            self.player.play('commands//visual_commands//movement.mov')
            self.audio_commands.perform_command('movement')
        elif prompt_code == 'left':
            self.player.play('commands//visual_commands//left.mov')
            self.audio_commands.perform_command('left')
        elif prompt_code == 'right':
            self.player.play('commands//visual_commands//right.mov')
            self.audio_commands.perform_command('right')
        elif prompt_code == 'rest':
            self.player.play('commands//visual_commands/rest.png')
            self.audio_commands.perform_command('rest')
        elif prompt_code == 'break':
            self.player.play('commands//visual_commands/pause.jpg')
            self.audio_commands.perform_command('pause')
        elif prompt_code == 'end':
            self.player.play('commands//visual_commands//end.jpg')
            self.audio_commands.perform_command('end')
            
        self.current_prompt_code = prompt_code

    def set_audio_prompt(self, prompt_code):
        if prompt_code == self.current_prompt_code:
            return
            
        self._ensure_prompt_label()
        
        if prompt_code == 'movement':
            self.audio_commands.perform_command('movement')
            self.prompt_label.setText("MOVEMENT")
        elif prompt_code == 'left':
            self.audio_commands.perform_command('left')
            self.prompt_label.setText("LEFT")
        elif prompt_code == 'right':
            self.audio_commands.perform_command('right')
            self.prompt_label.setText("RIGHT")
        elif prompt_code == 'rest':
            self.audio_commands.perform_command('rest')
            self.prompt_label.setText("REST")
        elif prompt_code == 'break':
            self.audio_commands.perform_command('pause')
            self.prompt_label.setText("BREAK")
        elif prompt_code == 'end':
            self.audio_commands.perform_command('end')
            self.prompt_label.setText("END")
            
        self.current_prompt_code = prompt_code

    def set_vr_prompt(self, prompt_code):
        if prompt_code == self.current_prompt_code:
            return
        print(prompt_code)
        
        self._ensure_prompt_label()
        
        if prompt_code == 'movement':
            self.audio_commands.perform_command('movement')
            self.control.left = True
            self.control.right = True
            self.control.mode = self.configurations.read('collect_data.vr_mode')
            state = self.sender.send_data(self.control)
            self.prompt_label.setText('MOVEMENT')
        elif prompt_code == 'left':
            self.audio_commands.perform_command('left')
            self.control.left = True
            self.control.right = False
            state = self.sender.send_data(self.control)
            self.prompt_label.setText('LEFT')
        elif prompt_code == 'right':
            self.audio_commands.perform_command('right')
            self.control.left = False
            self.control.right = True
            state = self.sender.send_data(self.control)
            self.prompt_label.setText('RIGHT')
        elif prompt_code == 'rest':
            self.audio_commands.perform_command('rest')
            self.control.left = False
            self.control.right = False
            self.control.mode = self.configurations.read('collect_data.vr_mode')
            state = self.sender.send_data(self.control)
            self.prompt_label.setText('REST')
        elif prompt_code == 'break':
            self.audio_commands.perform_command('pause')
            self.control.left = False
            self.control.right = False
            self.control.mode = self.configurations.read('collect_data.vr_mode')
            state = self.sender.send_data(self.control)
            self.prompt_label.setText('BREAK')
        elif prompt_code == 'end':
            self.audio_commands.perform_command('end')
            self.control.left = False
            self.control.right = False
            state = self.sender.send_data(self.control)
            self.prompt_label.setText('END')
            
        self.current_prompt_code = prompt_code

    def set_text_prompt(self, prompt_code):
        if prompt_code == self.current_prompt_code:
            return
            
        self._ensure_prompt_label()
        
        if prompt_code == 'movement':
            self.prompt_label.setText('MOVEMENT')
        elif prompt_code == 'left':
            self.prompt_label.setText('LEFT')
        elif prompt_code == 'right':
            self.prompt_label.setText('RIGHT')
        elif prompt_code == 'rest':
            self.prompt_label.setText('REST')
        elif prompt_code == 'break':
            self.prompt_label.setText('BREAK')
        elif prompt_code == 'end':
            self.prompt_label.setText('END')
            
        self.current_prompt_code = prompt_code

    def _ensure_prompt_label(self):
        if self.prompt_label is None:
            self.prompt_label = QLabel("BREAK")
            self.prompt_label.setAlignment(Qt.AlignCenter)
            self.prompt_label.setStyleSheet("color: white; font-family: 'Segoe UI'; font-size: 70px; font-weight: bold;")
            self.layout.addWidget(self.prompt_label)
