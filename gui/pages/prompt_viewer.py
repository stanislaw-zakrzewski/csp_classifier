from tkinter import *

from gui.colors import colors
from gui.components.double_scrolled_frame import DoubleScrolledFrame
from gui.fonts import fonts
from gui.pages.start_page import StartPage
from gui.visual_player import Screen

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


class PromptViewer(DoubleScrolledFrame):
    def __init__(self, parent, controller):
        DoubleScrolledFrame.__init__(self, parent)

        app_title = Label(self, text="Kombajn EEG", font=fonts['large_bold_font'], bg=colors['white_smoke'])
        app_title.grid(row=0, column=0, padx=10, pady=10, columnspan=10, sticky='W')

        back_to_start_page_button = Button(self, text="Back to Start Page",
                                           command=lambda: controller.show_frame(StartPage))
        back_to_start_page_button.grid(row=1, column=0, padx=10, pady=10, sticky='w')

        self.prompt_viewer_state = PromptViewerState

        self.button_canvas = Canvas(self)
        self.button_canvas.grid(row=2, column=0, sticky='w', pady=10)

        Label(self.button_canvas, text='Video:').grid(row=0, column=0, padx=5, pady=5)
        for index, video_prompt in enumerate(PROMPTS['video']):
            Button(self.button_canvas, text=video_prompt['label'],
                   command=lambda prompt_path=video_prompt['path']: self.set_current_prompt('video', prompt_path)) \
                .grid(row=0, column=index + 1, padx=5, pady=5)

        Label(self.button_canvas, text='Image:').grid(row=1, column=0, padx=5, pady=5)
        for index, image_prompt in enumerate(PROMPTS['image']):
            Button(self.button_canvas, text=image_prompt['label'],
                   command=lambda prompt_path=image_prompt['path']: self.set_current_prompt('image', prompt_path)) \
                .grid(row=1, column=index + 1, padx=5, pady=5)

        Button(self.button_canvas, text='Clear', command=lambda: self.clear_prompt()).grid(row=2, column=0, padx=5,
                                                                                           pady=5)

        self.video_canvas = None
        self.player = None

    def set_current_prompt(self, prompt_type, prompt_source):
        if prompt_type == 'video':
            self.set_video_prompt(prompt_source)
        if prompt_type == 'image':
            self.set_image_prompt(prompt_source)

    def set_video_prompt(self, prompt_source):
        if self.player is None:
            self.video_canvas = Canvas(self, width=500, height=300)
            self.player = Screen(self.video_canvas)
            self.video_canvas.grid(row=3, column=0)
            self.player.place(x=0, y=0, width=500, height=300)
        self.player.play(prompt_source)

    def set_image_prompt(self, prompt_source):
        if self.player is None:
            self.video_canvas = Canvas(self, width=500, height=300)
            self.player = Screen(self.video_canvas)
            self.video_canvas.grid(row=3, column=0)
            self.player.place(x=0, y=0, width=500, height=300)
        self.player.play(prompt_source)

    def clear_prompt(self):
        self.player.destroy()
        self.player = None
        self.video_canvas.destroy()
        self.video_canvas = None
