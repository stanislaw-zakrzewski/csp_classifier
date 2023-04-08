from tkinter import *

from gui.visual_player import Screen


class PromptViewer(Toplevel):
    def __init__(self, root, start_command, close_command):
        Toplevel.__init__(self, root)
        self.title("Browse annotations for")
        self.geometry("1200x720")

        self.start_button = Button(self, text='START ACQUISITION',
                                   command=lambda: self.start_acquisition(start_command))
        self.start_button.place(relx=.5, rely=.5, anchor=CENTER)
        self.player = None
        self.current_prompt_code = None
        self.closed = False
        self.queue_canvas = None

        self.close_command = close_command
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

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

    def change_prompt(self, prompt_code):
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
