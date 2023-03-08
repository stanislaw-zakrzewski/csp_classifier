from tkinter import *

from gui.menu.menu import ApplicationMenu
from gui.pages.analyze_data import AnalyzeData
from gui.pages.browse_recordings import BrowseRecordings
from gui.pages.start_page import StartPage
from gui.pages.test_electrodes import TestElectrodes


class App(Tk):
    def __init__(self, *args, **kwargs):
        Tk.__init__(self, *args, **kwargs)
        self.title("Kombajn EEG")
        self.state('zoomed')
        self.config(menu=ApplicationMenu(self))
        self.config(bg="white")
        pages = [
            {'name': 'Test Electrodes', 'frame': TestElectrodes},
            {'name': 'Analyze Data', 'frame': AnalyzeData},
            {'name': 'Browse Recordings', 'frame': BrowseRecordings},
        ]

        container = Frame(self)
        container.pack(side="top", fill="both", expand=True)

        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        start_frame = StartPage(container, self, pages)
        self.frames[StartPage] = start_frame

        start_frame.grid(row=0, column=0, sticky="nsew")
        for F in (x['frame'] for x in pages):
            frame = F(container, self)

            self.frames[F] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(BrowseRecordings)
        # Example of adding a video
        # self.player = Screen(self)
        # self.player.place(x=0, y=0, width=500, height=300)
        # self.player.play('commands//visual_commands//left.mov')

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()


app = App()
app.mainloop()
