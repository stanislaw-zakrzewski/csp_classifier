import os
import timeit
from tkinter import *

from mne.io import read_raw_edf
from pyedflib import highlevel

from gui.annotation_viewer import open_annotation_viewer
from gui.colors import colors
from gui.fonts import fonts
from gui.pages.start_page import StartPage

SELECTED_FIELDS = [
    'patientname',
    'startdate',
    'channels',
    'annotations'
]


class BrowseRecordings(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent)

        app_title = Label(self, text="Kombajn EEG", font=fonts['large_bold_font'], bg=colors['white_smoke'])
        app_title.grid(row=0, column=0, padx=10, pady=10, columnspan=10, sticky=W)

        back_to_start_page_button = Button(self, text="Back to Start Page",
                                           command=lambda: controller.show_frame(StartPage))
        back_to_start_page_button.grid(row=1, column=0, padx=10, pady=10, sticky=W)
        edf_headers = {}
        for filename in os.listdir('data_s'):

            if filename[-4:] == '.edf':
                # a.append(read_raw_edf(os.path.join('data_s', filename)))

                edf_headers[filename] = highlevel.read_edf_header(os.path.join('data_s', filename))

                # print(a[-1].info)
        table = Frame(self)
        table.grid(row=2, column=0)
        Label(table, text='filename', font=fonts['medium_bold']).grid(row=0, column=0, padx=5, pady=5)
        for index, selected_field in enumerate(SELECTED_FIELDS):
            Label(table, text=selected_field, font=fonts['medium_bold']).grid(row=0, column=index+1, padx=5, pady=5)

        row_index = 1
        for edf_header_key in edf_headers:
            edf_header = edf_headers[edf_header_key]
            Label(table, text=edf_header_key).grid(row=row_index, column=0, padx=5, pady=5, sticky=W)

            column_index = 1
            for key in edf_header.keys():

                if key in SELECTED_FIELDS:
                    # print(key)
                    if key == 'annotations':
                        Button(table, text='View Annotations',
                               command=lambda key_copy=edf_header_key, annotations_copy=edf_header[key]: open_annotation_viewer(table, key_copy, annotations_copy)).grid(
                            row=row_index,
                            column=column_index)
                    else:
                        Label(table, text=edf_header[key]).grid(row=row_index, column=column_index, padx=5, pady=5)
                    column_index += 1
            row_index += 1
        # for i in edf_headers:

        # print('pol')
