import copy
import random
import time
from threading import Thread
from tkinter import *

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from pyedflib import highlevel

import pygds
from config.config import Configurations
from gui.colors import colors
from gui.components.double_scrolled_frame import DoubleScrolledFrame
from gui.fonts import fonts
from gui.pages.collect_data_components.prompt_viewer import PromptViewer
from gui.pages.start_page import StartPage
from datetime import datetime

# commands = get_commands()
signal = []
trial_order = []
current_label = -1
current_trial_remaining_length = 0
current_length_in_seconds = 0
annotations = []
last = 0


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

        self.patient_name_label = Label(self, text='Patient Name:')
        self.patient_name_label.grid(row=3, column=0)
        self.patient_name_value = StringVar()
        self.patient_name_value.set(self.configurations.read('all.collect_data.patient_name'))
        self.patient_name_input = Entry(self, text=self.patient_name_value)
        self.patient_name_input.grid(row=3, column=1)

        self.gender_label = Label(self, text='Patient Gender:')
        self.gender_label.grid(row=4, column=0)
        self.gender_value = StringVar()
        self.gender_value.set(self.configurations.read('all.collect_data.patient_gender'))
        self.gender_input = Entry(self, text=self.gender_value)
        self.gender_input.grid(row=4, column=1)

        self.prepare_experiment = Button(self, text='Prepare experiment', command=self.open_prompt_window)
        self.prepare_experiment.grid(row=5, column=0, padx=10, pady=10)

        self.acquisition_thread = None
        self.queue_canvas = None

        self.plot_canvas = None
        self.prompt_viewer = None

        self.fig = None
        self.gnt = None
        self.labels = None

    def open_prompt_window(self):
        self.prepare_experiment['state'] = DISABLED
        self.patient_name_input['state'] = DISABLED
        self.gender_input['state'] = DISABLED
        self.create_queue()
        self.prompt_viewer = PromptViewer(self, self.start_acquisition, self.on_prompt_viewer_close)
        self.update_experiment_timeline_plot()

    def on_prompt_viewer_close(self):
        self.prepare_experiment['state'] = NORMAL
        self.patient_name_input['state'] = NORMAL
        self.gender_input['state'] = NORMAL
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
            self.fig = Figure(figsize=(15, 6))
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
            self.plot_canvas.get_tk_widget().grid(row=6, column=0, columnspan=10)
        else:
            self.plot_canvas.draw()

    def start_acquisition(self):
        self.acquisition_thread = Thread(target=self.run_acquisition)
        self.acquisition_thread.start()
        self.queue_canvas = Canvas(self)
        self.update_experiment_timeline_plot()

    def run_acquisition(self):
        global current_trial_remaining_length
        global current_label
        global trial_order
        global signal
        global current_length_in_seconds
        global annotations
        global last
        global commands

        d = pygds.GDS()
        pygds.configure_demo(d)
        d.SetConfiguration()

        batches_per_second = 2
        trial_length_random_addition_in_seconds = 0
        instructions_dict = {-1: 'pause', 0: 'rest', 1: 'movement'}
        electrode_names = self.configurations.read('all.general.all_electrodes')
        sampling_frequency = self.configurations.read('all.general.sampling_rate')

        for _ in range(32):
            signal.append([])

        def processCallback(samples):
            try:
                global current_trial_remaining_length
                global current_label
                global trial_order
                global signal
                global current_length_in_seconds
                global annotations
                global last
                global commands
                dt = datetime.now()
                last = dt

                for channel in range(32):
                    signal[channel] = np.concatenate((signal[channel], list(samples[:, channel])))

                # Podglad aktywnosci kanalow:
                np.set_printoptions(suppress=True, linewidth=10000, precision=2)
                # print(np.std(samples, axis=0)) # wszystkie kanały
                # print(np.std(samples[:, [32, 33, 34]], axis=0)) # akcelerometry - dla kontroli ;-)
                # print(np.std(samples[:, [5, 15, 14, 13, 23, 9, 17, 18, 19, 27, 16]], axis=0)) # FC3, C1, C3, C5, CP3, FC4, C2, C4, C6, CP4, CZ

                if self.current_queue is None or len(self.current_queue) == 0:
                    return False
                item = self.current_queue[0]
                if not self.prompt_viewer.closed:
                    self.prompt_viewer.change_prompt(item[0])
                    time.sleep(.5)
                    item[1] -= .5
                    if item[1] < .5 and self.current_queue is not None:
                        self.current_queue.pop(0)
                    self.update_experiment_timeline_plot()
                else:
                    return False

                # current_trial_remaining_length -= 1
                # current_length_in_seconds += 1 / batches_per_second
                #
                # if current_trial_remaining_length == 0:
                #     if current_label == -1:
                #         current_label = trial_order.pop(0)
                #
                #         current_trial_remaining_length = \
                #             np.random.randint(
                #                 trial_length_random_addition_in_seconds * batches_per_second + 1) + trial_length_in_seconds * batches_per_second
                #         annotations.append(
                #             [current_length_in_seconds, current_trial_remaining_length / 2,
                #              instructions_dict[current_label]])
                #     else:
                #         if len(trial_order) == 0:
                #             return False
                #         current_label = -1
                #         current_trial_remaining_length = \
                #             np.random.randint(
                #                 trial_timeout_random_addition_in_seconds * batches_per_second + 1) + trial_timeout_in_seconds * batches_per_second
                #
                # commands.perform_command(instructions_dict[current_label])

                return True
            except Exception as e:
                print('ERROR:', e)

        # while self.current_queue is not None and len(self.current_queue) > 0:
        #     item = self.current_queue[0]
        #     if not self.prompt_viewer.closed:
        #         self.prompt_viewer.change_prompt(item[0])
        #         time.sleep(.5)
        #         item[1] -= .5
        #         if item[1] < .5 and self.current_queue is not None:
        #             self.current_queue.pop(0)
        #         self.update_experiment_timeline_plot()
        #     else:
        #         break
        last = datetime.now()
        all = datetime.now()
        start_date = datetime.now()
        d.GetData(d.SamplingRate // batches_per_second, processCallback)
        d.Close()

        del d
        t = time.localtime()
        timestamp = time.strftime('%Y-%m-%dT%H-%M-%S', t)
        filename = 'data/{}.edf'.format(timestamp)

        sig_headers = highlevel.make_signal_headers(electrode_names, sample_rate=sampling_frequency,
                                                    physical_max=2000000,
                                                    physical_min=-2000000)

        annotations = []
        len_for_annot = 0
        for index, queue_element in enumerate(self.queue):
            if queue_element[0] != 'break':
                annotations.append([len_for_annot, queue_element[1], queue_element[0]])
            len_for_annot += queue_element[1]

        header = highlevel.make_header(patientname=self.patient_name_value.get(), gender=self.gender_value.get(),
                                       startdate=start_date)
        header.update({'annotations': annotations})

        if not self.prompt_viewer.closed:
            self.prompt_viewer.change_prompt('end')
        else:
            self.prompt_viewer.destroy()
        print(sig_headers)
        highlevel.write_edf(filename, signal, sig_headers, header)

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
