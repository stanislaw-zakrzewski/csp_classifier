from tkinter import *

from gui.menu.visualization.edf import visualize_edf
from gui.menu.visualization.menu import VisualizationMenu
from gui.settings import Settings


class ApplicationMenu(Menu):
    def __init__(self, root):
        Menu.__init__(self, root)
        self.root = root
        self.filemenu = Menu(self, tearoff=0)
        self.filemenu.add_command(label="New", command=self.donothing)
        self.filemenu.add_command(label="Open", command=self.donothing)
        self.filemenu.add_command(label="Save", command=self.donothing)
        self.filemenu.add_separator()
        self.filemenu.add_command(label="Exit", command=root.quit)
        self.add_cascade(label="File", menu=self.filemenu)

        visualization_menu = VisualizationMenu(self)
        self.add_cascade(label="Visualize", menu=visualization_menu)
        self.add_command(label='Settings', command=self.open_settings)

    def donothing(self):
        x = 0

    def open_settings(self):
        Settings(self.root)




def create_menu(root):
    menubar = ApplicationMenu(root)

    return menubar