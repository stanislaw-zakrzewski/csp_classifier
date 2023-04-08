import copy
import random
import time
from threading import Thread
from tkinter import *

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from config.config import Configurations
from gui.colors import colors
from gui.components.double_scrolled_frame import DoubleScrolledFrame
from gui.fonts import fonts
from gui.pages.collect_data_components.prompt_viewer import PromptViewer
from gui.pages.start_page import StartPage


class CollectData(DoubleScrolledFrame):
    def __init__(self, parent, controller):
        DoubleScrolledFrame.__init__(self, parent)
        self.configurations = Configurations()
        self.queue = None
        self.current_queue = None

        app_title = Label(self, text="Kombajn EEG", font=fonts['large_bold_font'], bg=colors['white_smoke'])
        app_title.grid(row=0, column=0, padx=10, pady=10, columnspan=10, sticky='W')

        back_to_start_page_button = Button(self, text="Back to Start Page",
                                           command=lambda: controller.show_frame(StartPage))
        back_to_start_page_button.grid(row=1, column=0, padx=10, pady=10)

        self.prepare_experiment = Button(self, text='Prepare experiment', command=self.open_prompt_window)
        self.prepare_experiment.grid(row=3, column=0, padx=10, pady=10)

        self.acquisition_thread = None
        self.queue_canvas = None

        self.plot_canvas = None
        self.prompt_viewer = None

        self.fig = None
        self.gnt = None
        self.labels = None

    def open_prompt_window(self):
        self.create_queue()
        self.prompt_viewer = PromptViewer(self, self.start_acquisition, self.on_prompt_viewer_close)
        self.update_experiment_timeline_plot()
        self.prepare_experiment['state'] = DISABLED

    def on_prompt_viewer_close(self):
        self.prepare_experiment['state'] = NORMAL
        self.queue = None
        self.current_queue = None
        self.plot_canvas.get_tk_widget().destroy()
        self.plot_canvas = None
        self.fig = None
        self.gnt = None

    def update_experiment_timeline_plot(self):
        if self.current_queue is None:
            return

        if self.fig is None:
            self.fig = Figure(figsize=(25, 10))
        if self.gnt is None:
            self.gnt = self.fig.subplots()
            self.gnt.set_xlabel('seconds since start')
            self.gnt.set_ylabel('Prompt')
        else:
            self.gnt.clear()

        # Prepare data for plot
        data = {}
        previous_time_end = 0
        for item in self.current_queue:
            if item[0] not in data:
                data[item[0]] = []
            data[item[0]].append((previous_time_end, item[1]))
            previous_time_end += item[1]
        if self.labels is None:
            self.labels = data.keys()

        yticks = []
        for i in range(len(self.labels)):
            yticks.append(5 + i * 10)
        self.gnt.set_yticks(yticks)

        # Set limits
        self.gnt.set_ylim(0, yticks[-1] + 5)
        self.gnt.set_xlim(0, 40)

        self.gnt.set_yticklabels(self.labels)
        self.gnt.grid(True)

        for index, item in enumerate(self.labels):
            if item in data:
                self.gnt.broken_barh(data[item], (index * 10, 9))

        if self.plot_canvas is None:
            self.plot_canvas = FigureCanvasTkAgg(self.fig, master=self)
            self.plot_canvas.get_tk_widget().grid(row=4, column=0, columnspan=10)
        else:
            self.plot_canvas.draw()

    def start_acquisition(self):
        self.acquisition_thread = Thread(target=self.run_acquisition)
        self.acquisition_thread.start()
        self.queue_canvas = Canvas(self)
        self.update_experiment_timeline_plot()

    def run_acquisition(self):
        while self.current_queue is not None and len(self.current_queue) > 0:
            item = self.current_queue[0]
            if not self.prompt_viewer.closed:
                self.prompt_viewer.change_prompt(item[0])
                time.sleep(.5)
                item[1] -= .5
                if item[1] < .5 and self.current_queue is not None:
                    self.current_queue.pop(0)
                self.update_experiment_timeline_plot()
            else:
                break
        # for queue_element in self.queue:
        #     print(self.prompt_viewer.closed)
        #     if not self.prompt_viewer.closed:
        #         self.prompt_viewer.change_prompt(queue_element[0])
        #         time.sleep(queue_element[1])
        #         elapsed += queue_element[1]
        #     self.show_plt(elapsed)
        if not self.prompt_viewer.closed:
            self.prompt_viewer.change_prompt('end')
        else:
            self.prompt_viewer.destroy()

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
        self.current_queue = copy.deepcopy(queue)
