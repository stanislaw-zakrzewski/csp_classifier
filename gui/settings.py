import copy
from tkinter import *

from config.config import Configurations


class Settings(Toplevel):
    def __init__(self, root):
        Toplevel.__init__(self, root)
        self.configurations = Configurations()
        self.title("Settings")
        self.geometry("400x600")
        self.grab_set()

        self.sampling_rate_label = Label(self, text="Sampling Rate").grid(row=0, column=0)
        self.sampling_rate_value = StringVar()
        self.sampling_rate_input = Entry(self, textvariable=self.sampling_rate_value).grid(row=0, column=1)

        self.restore_configuration(self.configurations.current_configuration)

        self.button = Button(self, text='Discard changes', command=lambda: self.restore_configuration(
            self.configurations.current_configuration)).grid(row=2, column=0)
        self.button = Button(self, text='Restore default configuration', command=lambda: self.restore_configuration(
            self.configurations.default_configuration)).grid(row=2, column=1)
        self.button = Button(self, text='Save configuration', command=self.save_current_configuration).grid(row=2,
                                                                                                            column=2)

    def restore_configuration(self, configuration):
        self.sampling_rate_value.set(configuration['sampling_rate'])

    def save_current_configuration(self):
        new_configuration = copy.deepcopy(self.configurations.current_configuration)
        new_configuration['sampling_rate'] = self.sampling_rate_value.get()
        self.configurations.change_current_configuration(new_configuration)
