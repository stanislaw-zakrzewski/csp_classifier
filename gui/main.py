from tkinter import *

from gui.menu.menu import ApplicationMenu


class MainWindow(Tk):
    def __init__(self):
        Tk.__init__(self)
        self.title("Kombajn EEG")
        self.state('zoomed')
        self.config(menu=ApplicationMenu(self))


window = MainWindow()
window.mainloop()
