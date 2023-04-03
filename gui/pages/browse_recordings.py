import os
from tkinter import *
from tkinter import filedialog

from pyedflib import highlevel

from gui.annotation_viewer import open_annotation_viewer
from gui.colors import colors
from gui.components.double_scrolled_frame import DoubleScrolledFrame
from gui.fonts import fonts
from gui.pages.start_page import StartPage

SELECTED_FIELDS = [
    'patientname',
    'startdate',
    'channels',
    'annotations'
]


class BrowseRecordings(DoubleScrolledFrame):

    def add_files_from_folder(self):
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            for filename in os.listdir(folder_selected):
                if filename[-4:] == '.edf':
                    if filename not in self.edf_headers:
                        self.edf_headers[filename] = highlevel.read_edf_header(os.path.join(folder_selected, filename))
                        edf_header = self.edf_headers[filename]
                        Label(self.table, text=filename).grid(row=self.current_row, column=0, padx=5, pady=5, sticky=W)

                        column_index = 1
                        for key in edf_header.keys():

                            if key in SELECTED_FIELDS:

                                if key == 'annotations':
                                    Button(self.table, text='View Annotations',
                                           command=lambda key_copy=filename,
                                                          annotations_copy=edf_header[key]: open_annotation_viewer(
                                               self.table, key_copy, annotations_copy)).grid(
                                        row=self.current_row,
                                        column=column_index)
                                else:
                                    Label(self.table, text=edf_header[key]).grid(row=self.current_row,
                                                                                 column=column_index, padx=5,
                                                                                 pady=5)
                                column_index += 1
                        self.current_row += 1

    def __init__(self, parent, controller):
        DoubleScrolledFrame.__init__(self, parent)
        self.edf_paths = []
        self.current_row = 0
        self.edf_headers = {}

        app_title = Label(self, text="Kombajn EEG", font=fonts['large_bold_font'], bg=colors['white_smoke'])
        app_title.grid(row=0, column=0, padx=10, pady=10, columnspan=10, sticky=W)

        back_to_start_page_button = Button(self, text="Back to Start Page",
                                           command=lambda: controller.show_frame(StartPage))
        back_to_start_page_button.grid(row=1, column=0, padx=10, pady=10, sticky=W)

        load_data_from_folder_button = Button(self, text="Load Data from Folder", command=self.add_files_from_folder)
        load_data_from_folder_button.grid(row=2, column=0, padx=10, pady=10, sticky=W)

        edf_headers = {}
        for filename in os.listdir('data_s'):
            if filename[-4:] == '.edf':
                edf_headers[filename] = highlevel.read_edf_header(os.path.join('data_s', filename))

        self.table = Frame(self)
        self.table.grid(row=3, column=0)
        Label(self.table, text='filename', font=fonts['medium_bold']).grid(row=0, column=0, padx=5, pady=5)
        for index, selected_field in enumerate(SELECTED_FIELDS):
            Label(self.table, text=selected_field, font=fonts['medium_bold']).grid(row=0, column=index + 1, padx=5,
                                                                                   pady=5)
        self.current_row = 1
