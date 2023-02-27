import copy
from tkinter import *

from config.config import Configurations
from config.configuration_label_dictionary import configuration_label_dictionary


class Settings(Toplevel):
    def __init__(self, root):
        Toplevel.__init__(self, root)
        self.configurations = Configurations()
        self.title("Settings")
        self.geometry("400x600")
        self.grab_set()

        self.grid_row_last_index = 0
        self.labels = {}
        self.values = {}
        self.inputs = {}

        self.create_fields()

        self.restore_configuration(self.configurations.current_configuration)

        self.discard_changes_button = Button(self, text='Discard changes', command=lambda: self.restore_configuration(
            self.configurations.current_configuration)).grid(row=2, column=0)
        self.restore_default_button = Button(self, text='Restore default configuration',
                                             command=lambda: self.restore_configuration(
                                                 self.configurations.default_configuration)).grid(row=2, column=1)
        self.save_configuration_button = Button(self, text='Save configuration',
                                                command=self.save_current_configuration).grid(row=2,
                                                                                              column=2)

    def restore_configuration(self, configuration):
        for key in self.configurations.default_configuration:
            if key in configuration:
                self.values[key].set(configuration[key])
            else:
                self.values[key].set(self.configurations.default_configuration[key])

    def save_current_configuration(self):
        new_configuration = copy.deepcopy(self.configurations.current_configuration)
        for key in self.values:
            new_configuration[key] = self.values[key].get()
        self.configurations.change_current_configuration(new_configuration)

    def create_fields(self):
        for key in self.configurations.default_configuration:
            self.labels[key] = Label(self, text=configuration_label_dictionary[key]).grid(
                row=self.grid_row_last_index, column=0)
            self.values[key] = StringVar()
            self.inputs[key] = Entry(self, textvariable=self.values[key]).grid(
                row=self.grid_row_last_index, column=1)
            self.grid_row_last_index += 1
