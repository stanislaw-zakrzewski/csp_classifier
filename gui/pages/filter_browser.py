from threading import Thread
from tkinter import *

import numpy as np
from PIL import Image, ImageTk

import pygds
from config.config import Configurations
from gui.pages.start_page import StartPage
from gui.colors import colors
from gui.fonts import fonts
from gui.components.double_scrolled_frame import DoubleScrolledFrame
import pandas as pd

ELECTRODE_COORDINATES = {
    'Fp1': (479, 185),
    'FpZ': (600, 172),
    'Fp2': (726, 185),

    'AF7': (365, 245),
    'AF3': (463, 265),
    'AFZ': (600, 262),
    'AF4': (745, 266),
    'AF8': (842, 249),

    'F9': (208, 279),
    'F7': (294, 323),
    'F5': (368, 344),
    'F3': (446, 351),
    'F1': (524, 358),
    'FZ': (600, 363),
    'F2': (680, 360),
    'F4': (758, 351),
    'F6': (834, 341),
    'F8': (910, 323),
    'F10': (1001, 280),

    'FT9': (148, 414),
    'FT7': (240, 427),
    'FC5': (332, 435),
    'FC3': (425, 442),
    'FC1': (517, 449),
    'FCZ': (600, 451),
    'FC2': (687, 450),
    'FC4': (778, 442),
    'FC6': (876, 435),
    'FT8': (963, 426),
    'FT10': (1053, 414),

    'T9': (143, 544),
    'T7': (221, 544),
    'C5': (311, 543),
    'C3': (412, 544),
    'C1': (506, 547),
    'CZ': (600, 545),
    'C2': (689, 545),
    'C4': (792, 545),
    'C6': (886, 546),
    'T8': (982, 544),
    'T10': (1066, 542),

    'TP9': (143, 671),
    'TP7': (239, 652),
    'CP5': (336, 645),
    'CP3': (432, 642),
    'CP1': (517, 639),
    'CPZ': (600, 638),
    'CP2': (691, 639),
    'CP4': (777, 641),
    'CP6': (868, 645),
    'TP8': (964, 652),
    'TP10': (1060, 672),

    'P9': (196, 794),
    'P7': (288, 766),
    'P5': (366, 752),
    'P3': (446, 743),
    'P1': (526, 738),
    'PZ': (600, 737),
    'P2': (679, 738),
    'P4': (759, 742),
    'P6': (835, 751),
    'P8': (912, 766),
    'P10': (1007, 795),

    'PO7': (377, 853),
    'PO3': (473, 827),
    'POZ': (600, 824),
    'PO4': (722, 827),
    'PO8': (830, 853),

    'O1': (482, 906),
    'OZ': (600, 926),
    'O2': (722, 905),
}

FILTER_NAMES = {
    'hp': 'high-pass',
    'lp': 'low-pass',
    'bp': 'band-pass',
    'bs': 'band-stop'
}


class FilterBrowser(DoubleScrolledFrame):

    def __init__(self, parent, controller):
        DoubleScrolledFrame.__init__(self, parent)
        self.config(bg=colors['white_smoke'])
        self.configurations = Configurations()
        self.selected_electrodes = self.configurations.read('all.general.selected_electrodes')
        self.filter_data = pd.read_csv('filters.csv')

        self.filter_table = None
        self.add_filter_button = Button(self, text="Add Filter", bg='green',
                                        command=self.render_add_filter)
        self.add_filter_button.grid(row=5, column=1)
        self.add_filter_form = None
        self.add_input_name = None
        self.add_variable_band = StringVar(self)
        self.add_variable_band_specific = StringVar(self)
        self.add_variable_band_steepness = StringVar(self)
        self.add_input_type = None
        self.add_input_steepness = None
        self.add_input_freq1 = None
        self.add_input_freq2 = None
        self.add_input_channels = None

        app_title = Label(self, text="Kombajn EEG", font=fonts['large_bold_font'], bg=colors['white_smoke'])
        app_title.grid(row=0, column=0, padx=10, pady=10, columnspan=10, sticky='W')

        back_to_start_page_button = Button(self, text="Back to Start Page",
                                           command=lambda: controller.show_frame(StartPage))
        back_to_start_page_button.grid(row=2, column=0, padx=10, pady=10, sticky='W')

        test = ImageTk.PhotoImage(file="gui//electrode_placement_filled.png")
        frame2 = Frame(self, bg="red", width=test.width(), height=test.height())
        frame2.grid(row=3, column=0, columnspan=1, rowspan=10)
        frame2.image = test
        self.electrodes_canvas = Canvas(frame2, width=test.width(), height=test.height(), bg='blue')
        self.electrodes_canvas.pack(expand=YES, fill=BOTH)
        self.electrodes_canvas.create_image(2, 2, image=test, anchor=NW)
        self.table_title = Label(self, text='Filters', font=fonts['large_font'])
        self.table_title.grid(row=3, column=1)
        self.render_filter_table()

        # # label1 = Label(myCanvas, image=test)
        # # label1.image = test
        #
        # # Position image
        # # label1.place(x=0, y=0)
        # l = Label(frame2, bg='red', text='oko', width=50, height=50, borderwidth=0)
        # l.corner_radius = 5
        #
        #
        def create_circle(x, y, canvas, tag):  # center coordinates, radius
            r = 35
            x0 = x - r
            y0 = y - r
            x1 = x + r
            y1 = y + r
            hitbox = canvas.create_rectangle(x0, y0, x1, y1, outline='blue', width=0, tags=tag,
                                             stipple='@transparent.xbm', fill='gray')
            circle = canvas.create_oval(x0, y0, x1, y1, outline='red', width=0, tags=tag)
            return {'hitbox': hitbox, 'circle': circle}

        #
        self.electrode_indicators = {}

        # def change_color(new_selected_electrode_code):
        #     if new_selected_electrode_code == self.selected_electrode_code:
        #         self.electrodes_canvas.itemconfig(self.electrode_indicators[new_selected_electrode_code]['circle'], width=5)
        #         self.selected_electrode_code = None
        #         self.l.config(text='')
        #         self.l.config(text='')
        #     else:
        #         if self.selected_electrode_code is not None:
        #             self.electrodes_canvas.itemconfig(self.electrode_indicators[self.selected_electrode_code]['circle'],
        #                                          width=5)
        #         self.electrodes_canvas.itemconfig(self.electrode_indicators[new_selected_electrode_code]['circle'], width=10)
        #         self.selected_electrode_code = new_selected_electrode_code
        #         self.l.config(text=self.selected_electrode_code)

        for electrode in ELECTRODE_COORDINATES:
            electrode_x, electrode_y = ELECTRODE_COORDINATES[electrode]
            self.electrode_indicators[electrode] = create_circle(electrode_x, electrode_y, self.electrodes_canvas,
                                                                 electrode)

            # self.electrodes_canvas.tag_bind(electrode, "<Button-1>",
            #                            lambda event='', dup_el=electrode: change_color(dup_el))

            # Button(frame2, text="Change Color", command=change_color).place(x=600, y=600)
        # l.place(x=600,y=600)
        self.acquisition_thread = None
        self.acquisition_in_progress = False
        self.acquisition_initialized = False
        self.acquisition_stopped = False

    def update_electrode_colors(self, selected_electrodes):
        for electrode in ELECTRODE_COORDINATES:
            if electrode in selected_electrodes:
                self.electrodes_canvas.itemconfig(self.electrode_indicators[electrode]['circle'], width=10)
            else:
                self.electrodes_canvas.itemconfig(self.electrode_indicators[electrode]['circle'], width=0)

    def render_add_filter(self):
        self.add_filter_button.destroy()
        add_filter_form = Frame(self)
        Label(add_filter_form, text='Name', font=fonts['medium_font']).grid(row=0, column=0)
        self.add_input_name = Entry(add_filter_form)
        self.add_input_name.grid(row=0, column=1)

        def band_change_callback(*args):
            add_variable_band = self.add_variable_band.get()
            add_variable_band_specific = self.add_variable_band_specific.get()
            if add_variable_band == 'alpha':
                if add_variable_band_specific == 'whole':
                    self.add_input_freq1.delete(0, END)
                    self.add_input_freq1.insert(0, 6)
                    self.add_input_freq2.delete(0, END)
                    self.add_input_freq2.insert(0, 14)
                elif add_variable_band_specific == 'lower':
                    self.add_input_freq1.delete(0, END)
                    self.add_input_freq1.insert(0, 6)
                    self.add_input_freq2.delete(0, END)
                    self.add_input_freq2.insert(0, 10)
                elif add_variable_band_specific == 'upper':
                    self.add_input_freq1.delete(0, END)
                    self.add_input_freq1.insert(0, 10)
                    self.add_input_freq2.delete(0, END)
                    self.add_input_freq2.insert(0, 14)
                else:
                    self.add_input_freq1.delete(0, END)
                    self.add_input_freq1.insert(0, 8)
                    self.add_input_freq2.delete(0, END)
                    self.add_input_freq2.insert(0, 12)
            elif add_variable_band == 'beta':
                if add_variable_band_specific == 'whole':
                    self.add_input_freq1.delete(0, END)
                    self.add_input_freq1.insert(0, 15)
                    self.add_input_freq2.delete(0, END)
                    self.add_input_freq2.insert(0, 29)
                elif add_variable_band_specific == 'lower':
                    self.add_input_freq1.delete(0, END)
                    self.add_input_freq1.insert(0, 15)
                    self.add_input_freq2.delete(0, END)
                    self.add_input_freq2.insert(0, 22)
                elif add_variable_band_specific == 'upper':
                    self.add_input_freq1.delete(0, END)
                    self.add_input_freq1.insert(0, 22)
                    self.add_input_freq2.delete(0, END)
                    self.add_input_freq2.insert(0, 29)
                else:
                    self.add_input_freq1.delete(0, END)
                    self.add_input_freq1.insert(0, 18)
                    self.add_input_freq2.delete(0, END)
                    self.add_input_freq2.insert(0, 26)
            else:
                if add_variable_band_specific == 'whole':
                    self.add_input_freq1.delete(0, END)
                    self.add_input_freq1.insert(0, 30)
                    self.add_input_freq2.delete(0, END)
                    self.add_input_freq2.insert(0, 40)
                elif add_variable_band_specific == 'lower':
                    self.add_input_freq1.delete(0, END)
                    self.add_input_freq1.insert(0, 30)
                    self.add_input_freq2.delete(0, END)
                    self.add_input_freq2.insert(0, 35)
                elif add_variable_band_specific == 'upper':
                    self.add_input_freq1.delete(0, END)
                    self.add_input_freq1.insert(0, 35)
                    self.add_input_freq2.delete(0, END)
                    self.add_input_freq2.insert(0, 40)
                else:
                    self.add_input_freq1.delete(0, END)
                    self.add_input_freq1.insert(0, 32)
                    self.add_input_freq2.delete(0, END)
                    self.add_input_freq2.insert(0, 38)

        def band_steepness_callback(*args):
            steepness = self.add_variable_band_steepness.get()
            if steepness == 'steep':
                self.add_input_steepness.delete(0, END)
                self.add_input_steepness.insert(0, 15)
            elif steepness == 'semi-steep':
                self.add_input_steepness.delete(0, END)
                self.add_input_steepness.insert(0, 30)
            else:
                self.add_input_steepness.delete(0, END)
                self.add_input_steepness.insert(0, 50)

        Label(add_filter_form, text='Band', font=fonts['medium_font']).grid(row=1, column=0)
        self.add_variable_band_specific.set('whole')
        self.add_variable_band_specific.trace('w', band_change_callback)
        OptionMenu(add_filter_form, self.add_variable_band_specific, *['whole', 'lower', 'upper', 'middle']).grid(row=1,
                                                                                                                  column=1)
        self.add_variable_band.set('alpha')
        self.add_variable_band.trace('w', band_change_callback)
        OptionMenu(add_filter_form, self.add_variable_band, *['alpha', 'beta', 'gamma']).grid(row=1, column=2)

        Label(add_filter_form, text='Steepness', font=fonts['medium_font']).grid(row=2, column=0)
        self.add_variable_band_steepness.set('steep')
        self.add_variable_band_steepness.trace('w', band_steepness_callback)
        OptionMenu(add_filter_form, self.add_variable_band_steepness, *['steep', 'semi-steep', 'soft']).grid(row=2,
                                                                                                             column=1)

        Label(add_filter_form, text='Frequency 1', font=fonts['medium_font']).grid(row=3, column=0)
        self.add_input_freq1 = Entry(add_filter_form)
        self.add_input_freq1.grid(row=3, column=1)
        Label(add_filter_form, text='Frequency 2', font=fonts['medium_font']).grid(row=4, column=0)
        self.add_input_freq2 = Entry(add_filter_form)
        self.add_input_freq2.grid(row=4, column=1)
        Label(add_filter_form, text='Steepness', font=fonts['medium_font']).grid(row=5, column=0)
        self.add_input_steepness = Entry(add_filter_form)
        self.add_input_steepness.grid(row=5, column=1)
        Label(add_filter_form, text='%', font=fonts['medium_font']).grid(row=5, column=3)
        Label(add_filter_form, text='Channels', font=fonts['medium_font']).grid(row=6, column=0)
        self.add_input_channels = Entry(add_filter_form)
        self.add_input_channels.grid(row=6, column=1)

        Button(add_filter_form, text='Cancel', bg='tan1',
               command=self.render_add_button).grid(row=7,
                                                    column=0)
        Button(add_filter_form, text='Add', bg='seagreen2',
               command=self.add_filter).grid(row=7,
                                             column=1)
        self.add_filter_form = add_filter_form
        self.add_filter_form.grid(row=5, column=1)

        self.add_input_freq1.insert(0, 6)
        self.add_input_freq2.insert(0, 14)
        self.add_input_steepness.insert(0, 15)

    def add_filter(self):
        name = self.add_input_name.get()
        # type = self.add_input_type.get()
        freq1 = self.add_input_freq1.get()
        freq2 = self.add_input_freq2.get()
        steepness = self.add_input_steepness.get()
        channels = self.add_input_channels.get()
        self.filter_data.loc[len(self.filter_data.index)] = [name, 'bp', freq1, freq2, steepness, channels]
        self.filter_data.to_csv('filters.csv', index=False)
        self.render_add_button()
        self.render_filter_table()

    def render_add_button(self):
        self.add_filter_form.destroy()
        self.add_filter_button = Button(self, text="Add Filter", bg='green',
                                        command=self.render_add_filter)
        self.add_filter_button.grid(row=5, column=1)

    def render_filter_table(self):
        if self.filter_table:
            self.filter_table.destroy()
        filter_table = Frame(self)
        Label(filter_table, text='Name', font=fonts['medium_font']).grid(row=0, column=0)
        Label(filter_table, text='Type', font=fonts['medium_font']).grid(row=0, column=1)
        Label(filter_table, text='Frequency 1', font=fonts['medium_font']).grid(row=0, column=2)
        Label(filter_table, text='Frequency 2', font=fonts['medium_font']).grid(row=0, column=3)
        Label(filter_table, text='Steepness', font=fonts['medium_font']).grid(row=0, column=4)
        Label(filter_table, text='Channels', font=fonts['medium_font']).grid(row=0, column=5)
        for row_index, data_row in enumerate(self.filter_data.values):
            Label(filter_table, text=data_row[0], font=fonts['medium_font']).grid(row=row_index + 1, column=0)
            Label(filter_table, text=FILTER_NAMES[data_row[1]], font=fonts['medium_font']).grid(row=row_index + 1,
                                                                                                column=1)
            Label(filter_table, text=data_row[2], font=fonts['medium_font']).grid(row=row_index + 1, column=2)
            Label(filter_table, text=data_row[3], font=fonts['medium_font']).grid(row=row_index + 1, column=3)
            Label(filter_table, text=data_row[4], font=fonts['medium_font']).grid(row=row_index + 1, column=4)
            Label(filter_table, text=data_row[5], font=fonts['medium_font']).grid(row=row_index + 1, column=5)
            Button(filter_table, text='Show Electrodes', bg='blue',
                   command=lambda bound_row_index=row_index: self.select_row(bound_row_index)).grid(row=row_index + 1,
                                                                                                    column=6)
            Button(filter_table, text='DELETE', bg='red',
                   command=lambda bound_row_index=row_index: self.remove_row(bound_row_index)).grid(row=row_index + 1,
                                                                                                    column=7)
        self.filter_table = filter_table
        self.filter_table.grid(row=4, column=1)

    def select_row(self, row_index):
        selected_row = self.filter_data.iloc[[row_index]].values[0]
        self.update_electrode_colors(selected_row[5].split(' '))

    def remove_row(self, row_index):
        if not self.filter_data.empty:
            self.filter_data = self.filter_data.drop(self.filter_data.index[row_index])
        self.filter_data.to_csv('filters.csv', index=False)
        self.render_filter_table()

    def toggle_data(self):
        if not self.acquisition_initialized:
            self.acquisition_thread = Thread(target=self.run_acquisition)
            self.acquisition_thread.start()
            self.start_stop_button['state'] = 'disable'
        if self.acquisition_in_progress:
            self.acquisition_stopped = True
            self.start_stop_button['state'] = 'disable'

    def run_acquisition(self):
        # global current_trial_remaining_length
        # global current_label
        # global trial_order
        # global signal
        # global current_length_in_seconds
        # global annotations
        # global last
        # global commands

        d = pygds.GDS()
        pygds.configure_demo(d)
        d.SetConfiguration()

        batches_per_second = 2
        trial_length_random_addition_in_seconds = 0
        instructions_dict = {-1: 'pause', 0: 'rest', 1: 'movement'}
        electrode_names = self.configurations.read('all.general.all_electrodes')
        sampling_frequency = self.configurations.read('all.general.sampling_rate')

        # for _ in range(32):
        #     signal.append([])

        def processCallback(samples):
            if self.acquisition_stopped:
                self.start_stop_button['state'] = 'normal'
                self.start_stop_button['text'] = 'start'
                self.acquisition_in_progress = False
                self.acquisition_initialized = False
                self.acquisition_stopped = False
                return False
            if not self.acquisition_in_progress:
                self.start_stop_button['text'] = 'Stop'
                self.start_stop_button['state'] = 'normal'
            try:
                print(np.std(samples[:, [5, 15, 14, 13, 23, 9, 17, 18, 19, 27, 16]], axis=0))
                # global current_trial_remaining_length
                # global current_label
                # global trial_order
                # global signal
                # global current_length_in_seconds
                # global annotations
                # global last
                # global commands
                # dt = datetime.now()
                # last = dt
                #
                # for channel in range(32):
                #     signal[channel] = np.concatenate((signal[channel], list(samples[:, channel])))
                #
                # # Podglad aktywnosci kanalow:
                # np.set_printoptions(suppress=True, linewidth=10000, precision=2)
                # # print(np.std(samples, axis=0)) # wszystkie kanały
                # # print(np.std(samples[:, [32, 33, 34]], axis=0)) # akcelerometry - dla kontroli ;-)
                # # print(np.std(samples[:, [5, 15, 14, 13, 23, 9, 17, 18, 19, 27, 16]], axis=0)) # FC3, C1, C3, C5, CP3, FC4, C2, C4, C6, CP4, CZ
                #
                # if self.current_queue is None or len(self.current_queue) == 0:
                #     return False
                # item = self.current_queue[0]
                # if not self.prompt_viewer.closed:
                #     self.prompt_viewer.change_prompt(item[0])
                #     time.sleep(.5)
                #     item[1] -= .5
                #     if item[1] < .5 and self.current_queue is not None:
                #         self.current_queue.pop(0)
                #     self.update_experiment_timeline_plot()
                # else:
                #     return False
                #
                #
                #
                #
                # # current_trial_remaining_length -= 1
                # # current_length_in_seconds += 1 / batches_per_second
                # #
                # # if current_trial_remaining_length == 0:
                # #     if current_label == -1:
                # #         current_label = trial_order.pop(0)
                # #
                # #         current_trial_remaining_length = \
                # #             np.random.randint(
                # #                 trial_length_random_addition_in_seconds * batches_per_second + 1) + trial_length_in_seconds * batches_per_second
                # #         annotations.append(
                # #             [current_length_in_seconds, current_trial_remaining_length / 2,
                # #              instructions_dict[current_label]])
                # #     else:
                # #         if len(trial_order) == 0:
                # #             return False
                # #         current_label = -1
                # #         current_trial_remaining_length = \
                # #             np.random.randint(
                # #                 trial_timeout_random_addition_in_seconds * batches_per_second + 1) + trial_timeout_in_seconds * batches_per_second
                # #
                # # commands.perform_command(instructions_dict[current_label])

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
        # last = datetime.now()
        # all = datetime.now()
        # start_date = datetime.now()
        d.GetData(d.SamplingRate // batches_per_second, processCallback)
        d.Close()
        #
        del d
        # t = time.localtime()
        # timestamp = time.strftime('%Y-%m-%dT%H-%M-%S', t)
        # filename = 'data/{}.edf'.format(timestamp)
        #
        # sig_headers = highlevel.make_signal_headers(electrode_names, sample_rate=sampling_frequency,
        #                                             physical_max=2000000,
        #                                             physical_min=-2000000)
        #
        # annotations = []
        # len_for_annot = 0
        # for index, queue_element in enumerate(self.queue):
        #     if queue_element[0] != 'break':
        #         annotations.append([len_for_annot,queue_element[1], queue_element[0]])
        #     len_for_annot += queue_element[1]
        #
        # header = highlevel.make_header(patientname='patient_x', gender='Male', startdate=start_date)
        # header.update({'annotations': annotations})
        #
        # if not self.prompt_viewer.closed:
        #     self.prompt_viewer.change_prompt('end')
        # else:
        #     self.prompt_viewer.destroy()
        # print(sig_headers)
        # highlevel.write_edf(filename, signal, sig_headers, header)
