from tkinter import *

import SenderLib
from commands.audio_commands_pyaudio import AudioCommands
from config.config import Configurations
from gui.visual_player import Screen


class PromptViewer(Toplevel):
    def __init__(self, root, start_command, close_command):
        Toplevel.__init__(self, root)
        self.configurations = Configurations()
        self.title("Browse annotations for")
        self.geometry("1200x720")

        self.start_button = Button(self, text='START ACQUISITION',
                                   command=lambda: self.start_acquisition(start_command))
        self.start_button.place(relx=.5, rely=.5, anchor=CENTER)
        self.player = None
        self.current_prompt_code = None
        self.closed = False
        self.queue_canvas = None
        self.prompt_label = None
        self.prompt_label_text = None

        self.close_command = close_command
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.audio_commands = AudioCommands()
        self.sender = None
        self.control = None
        self.ipaddress = self.configurations.read('all.collect_data.ipaddress')
        self.port = self.configurations.read('all.collect_data.port')

        self.prompt_type = self.configurations.read('all.collect_data.prompt_type')
        if self.prompt_type == 'visual':
            self.change_prompt = self.set_visual_prompt
        elif self.prompt_type == 'audio':
            self.change_prompt = self.set_audio_prompt
        elif self.prompt_type == 'vr':
            self.sender = SenderLib.Sender(self.ipaddress, self.port)
            self.control = SenderLib.GameControl()
            self.change_prompt = self.set_vr_prompt
        else:  # text prompt
            self.change_prompt = self.set_text_prompt

    def start_acquisition(self, start_command):
        start_command()
        self.start_button.destroy()

    def on_closing(self):
        if self.current_prompt_code == 'end':
            self.destroy()
            self.close_command()
        else:
            self.closed = True
            self.destroy()
            self.close_command()

    def set_visual_prompt(self, prompt_code):
        if prompt_code == self.current_prompt_code:
            return

        if self.player is None:
            self.player = Screen(self)
            self.player.pack(side='top', fill='both', expand=True)

        if prompt_code == 'movement':
            self.player.play('commands//visual_commands//movement.mov')
        if prompt_code == 'left':
            self.player.play('commands//visual_commands//left.mov')
        if prompt_code == 'right':
            self.player.play('commands//visual_commands//right.mov')
        elif prompt_code == 'rest':
            self.player.play('commands//visual_commands//rest.png')
        elif prompt_code == 'break':
            self.player.play('commands//visual_commands//pause.jpg')
        elif prompt_code == 'end':
            self.player.play('commands//visual_commands//end.jpg')

        self.current_prompt_code = prompt_code

    def set_audio_prompt(self, prompt_code):
        if prompt_code == self.current_prompt_code:
            return

        if self.prompt_label is None:
            self.prompt_label_text = StringVar()
            self.prompt_label_text.set('BREAK')
            self.prompt_label = Label(self, textvariable=self.prompt_label_text, font=("Segoe UI", 70))
            self.prompt_label.pack(side='top', fill='both', expand=True)

        if prompt_code == 'movement':
            self.audio_commands.perform_command('movement')
        if prompt_code == 'left':
            self.audio_commands.perform_command('left')
        if prompt_code == 'right':
            self.audio_commands.perform_command('right')
        elif prompt_code == 'rest':
            self.audio_commands.perform_command('rest')
        elif prompt_code == 'break':
            self.audio_commands.perform_command('pause')
        elif prompt_code == 'end':
            self.audio_commands.perform_command('end')

        self.current_prompt_code = prompt_code

    def set_vr_prompt(self, prompt_code):
        if prompt_code == self.current_prompt_code:
            return
        print(prompt_code)

        if self.prompt_label is None:
            self.prompt_label_text = StringVar()
            self.prompt_label_text.set('BREAK')
            self.prompt_label = Label(self, textvariable=self.prompt_label_text, font=("Segoe UI", 70))
            self.prompt_label.pack(side='top', fill='both', expand=True)

        if prompt_code == 'movement':
            self.control.left = True
            self.control.right = True
            self.control.mode = self.configurations.read('all.collect_data.vr_mode')
            state = self.sender.send_data(self.control)
            self.prompt_label_text.set('MOVEMENT')
        if prompt_code == 'left':
            self.control.left = True
            self.control.right = False
            state = self.sender.send_data(self.control)
            self.prompt_label_text.set('LEFT')
        if prompt_code == 'right':
            self.control.left = False
            self.control.right = True
            state = self.sender.send_data(self.control)
            self.prompt_label_text.set('RIGHT')
        elif prompt_code == 'rest':
            self.control.left = False
            self.control.right = False
            self.control.mode = self.configurations.read('all.collect_data.vr_mode')
            state = self.sender.send_data(self.control)
            self.prompt_label_text.set('REST')
        elif prompt_code == 'break':
            self.control.left = False
            self.control.right = False
            self.control.mode = self.configurations.read('all.collect_data.vr_mode')
            state = self.sender.send_data(self.control)
            self.prompt_label_text.set('BREAK')
        elif prompt_code == 'end':
            self.control.left = False
            self.control.right = False
            state = self.sender.send_data(self.control)
            self.prompt_label_text.set('END')

        self.current_prompt_code = prompt_code

    def set_text_prompt(self, prompt_code):
        if prompt_code == self.current_prompt_code:
            return

        if self.prompt_label is None:
            self.prompt_label_text = StringVar()
            self.prompt_label_text.set('BREAK')
            self.prompt_label = Label(self, textvariable=self.prompt_label_text, font=("Segoe UI", 70))
            self.prompt_label.pack(side='top', fill='both', expand=True)

        if prompt_code == 'movement':
            self.prompt_label_text.set('MOVEMENT')
        if prompt_code == 'left':
            self.prompt_label_text.set('LEFT')
        if prompt_code == 'right':
            self.prompt_label_text.set('RIGHT')
        elif prompt_code == 'rest':
            self.prompt_label_text.set('REST')
        elif prompt_code == 'break':
            self.prompt_label_text.set('BREAK')
        elif prompt_code == 'end':
            self.prompt_label_text.set('END')

        self.current_prompt_code = prompt_code

