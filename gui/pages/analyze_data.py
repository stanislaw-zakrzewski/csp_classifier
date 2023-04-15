from tkinter import *
from tkinter import filedialog as fd

import matplotlib.ticker as ticker
import seaborn as sns
from matplotlib.backends._backend_tk import NavigationToolbar2Tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from analyze_data import analyze_edf
from gui.colors import colors
from gui.components.double_scrolled_frame import DoubleScrolledFrame
from gui.fonts import fonts
from gui.pages.start_page import StartPage


class AnalyzeData(DoubleScrolledFrame):
    def __init__(self, parent, controller):
        DoubleScrolledFrame.__init__(self, parent)

        app_title = Label(self, text="Kombajn EEG", font=fonts['large_bold_font'], bg=colors['white_smoke'])
        app_title.grid(row=0, column=0, padx=10, pady=10, columnspan=10, sticky='W')

        back_to_start_page_button = Button(self, text="Back to Start Page",
                                           command=lambda: controller.show_frame(StartPage))
        back_to_start_page_button.grid(row=1, column=0, padx=10, pady=10)

        Button(self, text='Select EDF file', command=self.select_edf_file).grid(row=2, column=0, padx=10, pady=10)
        self.selected_edf_file = StringVar()
        self.selected_edf_file.set('')
        Label(self, textvariable=self.selected_edf_file).grid(row=2, column=1)
        Button(self, text='Analyze selected EDF', command=self.analyze_edf).grid(row=3, column=0, padx=10, pady=10)
        self.canvas = None

    def select_edf_file(self):
        filename = fd.askopenfilename(filetypes=[("European Data Format files", "*.edf")])
        if filename:
            self.selected_edf_file.set(filename)

    def analyze_edf(self):
        if self.selected_edf_file.get() != '':
            accuracy_data = analyze_edf(self.selected_edf_file.get(), verbose='ERROR')
            figure = Figure(figsize=(15, 6))
            ax = figure.subplots()
            sns.lineplot(data=accuracy_data, x="frequency", y="accuracy", hue="configuration", errorbar=None, ax=ax)

            ax.xaxis.set_major_locator(ticker.MultipleLocator(.5))
            ax.grid()
            if self.canvas is None:
                self.canvas = FigureCanvasTkAgg(figure, master=self)  # A tk.DrawingArea.
            else:
                self.canvas.get_tk_widget().destroy()
                self.canvas = FigureCanvasTkAgg(figure, master=self)
            self.canvas.draw()
            self.canvas.get_tk_widget().grid(row=4, column=0, columnspan=10)
            toolbar = NavigationToolbar2Tk(self.canvas, self, pack_toolbar=False)
            toolbar.update()
            toolbar.grid(row=5, column=0, columnspan=10)
