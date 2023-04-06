import time
import random
from threading import Thread
from tkinter import *
from tkinter import filedialog as fd

import matplotlib.ticker as ticker
import seaborn as sns
from matplotlib.backends._backend_tk import NavigationToolbar2Tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from analyze_data import analyze_edf
from config.config import Configurations
from gui.colors import colors
from gui.components.double_scrolled_frame import DoubleScrolledFrame
from gui.fonts import fonts
from gui.pages.start_page import StartPage
from gui.visual_player import Screen
from tkinter import messagebox
import numpy as np


class VisualState:
    def __init__(self, initial_state):
        self.state = initial_state

    def set_state(self, new_state):
        self.state = new_state

    def get_state(self):
        return self.state


class PromptViewer(Toplevel):
    def __init__(self, root, start_command):
        Toplevel.__init__(self, root)
        self.grab_set()
        self.title("Browse annotations for")
        self.geometry("1200x720")

        self.start_button = Button(self, text='START ACQUISITION',
                                   command=lambda: self.start_acquisition(start_command))
        self.start_button.place(relx=.5, rely=.5, anchor=CENTER)
        self.player = None
        self.current_prompt_code = None
        self.closed = False

        def on_closing():
            if self.player:
                self.player.stop()
                self.closed = True
                self.destroy()


        self.protocol("WM_DELETE_WINDOW", on_closing)

    def start_acquisition(self, start_command):
        start_command()
        self.start_button.destroy()

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


class CollectData(DoubleScrolledFrame):
    def __init__(self, parent, controller):
        DoubleScrolledFrame.__init__(self, parent)
        self.visual_state = VisualState('rest')
        self.configurations = Configurations()
        self.create_queue()

        app_title = Label(self, text="Kombajn EEG", font=fonts['large_bold_font'], bg=colors['white_smoke'])
        app_title.grid(row=0, column=0, padx=10, pady=10, columnspan=10, sticky='W')

        back_to_start_page_button = Button(self, text="Back to Start Page",
                                           command=lambda: controller.show_frame(StartPage))
        back_to_start_page_button.grid(row=1, column=0, padx=10, pady=10)

        Button(self, text='Prepare data collection', command=self.open_prompt_window).grid(row=3, column=0, padx=10,
                                                                                           pady=10)
        self.canvas = None
        self.prompt_viewer = None
        self.queue = None
        # self.player = Screen(self)
        # self.player.place(x=0, y=0, width=500, height=300)
        # self.player.play('commands//visual_commands//left.mov')

    def open_prompt_window(self):
        self.prompt_viewer = PromptViewer(self, self.start_acquisition)

    def start_acquisition(self):
        self.create_queue()
        acquisition_thread = Thread(target=self.run_acquisition)
        acquisition_thread.start()

    def run_acquisition(self):
        for queue_element in self.queue:
            print(self.prompt_viewer.closed)
            if not self.prompt_viewer.closed:
                self.prompt_viewer.change_prompt(queue_element[0])
                time.sleep(queue_element[1])
        if not self.prompt_viewer.closed:
            self.prompt_viewer.change_prompt('end')

    def create_queue(self):
        trial_count = self.configurations.read('all.collect_data.trial_count')
        pause_length = self.configurations.read('all.collect_data.pause_length')
        trial_length = self.configurations.read('all.collect_data.trial_length')
        labels = self.configurations.read('all.collect_data.labels')
        label_queue = list(np.repeat(labels, trial_count))
        random.shuffle(label_queue)
        queue = []
        for trial_label in label_queue:
            queue.append(['break', pause_length])
            queue.append([trial_label, trial_length])
        self.queue = queue
