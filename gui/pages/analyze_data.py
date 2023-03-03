from tkinter import *

from gui.colors import colors
from gui.fonts import fonts
from gui.pages.start_page import StartPage


class AnalyzeData(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent)

        app_title = Label(self, text="Kombajn EEG", font=fonts['large_bold_font'], bg=colors['white_smoke'])
        app_title.grid(row=0, column=0, padx=10, pady=10, columnspan=10, sticky='W')

        back_to_start_page_button = Button(self, text="Back to Start Page",
                                           command=lambda: controller.show_frame(StartPage))
        back_to_start_page_button.grid(row=2, column=1, padx=10, pady=10)
