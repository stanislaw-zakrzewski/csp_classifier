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

        self.create_group(self.configurations.default_configuration)

        self.restore_configuration(self.configurations.current_configuration)

        self.discard_changes_button = Button(self, text='Discard changes', command=lambda: self.restore_configuration(
            self.configurations.current_configuration)).grid(row=2, column=0)
        self.restore_default_button = Button(self, text='Restore default configuration',
                                             command=lambda: self.restore_configuration(
                                                 self.configurations.default_configuration)).grid(row=2, column=1)
        self.save_configuration_button = Button(self, text='Save configuration',
                                                command=self.save_current_configuration).grid(row=2,
                                                                                              column=2)

    def get_value_from_configuration(self, configuration, key):
        try:
            path = key.split('.')
            if len(path) == 1:
                return configuration[key]['value']
            else:
                return self.get_value_from_configuration(configuration[path[0]], '.'.join(path[1:]))
        except KeyError:
            return None

    def set_value_to_configuration(self, configuration, key, value):
        try:
            path = key.split('.')
            if len(path) == 1:
                configuration[key]['value'] = value
                return True
            else:
                return self.set_value_to_configuration(configuration[path[0]], '.'.join(path[1:]), value)
        except KeyError:
            return False

    def restore_configuration(self, configuration):
        for key in self.values:
            value = self.get_value_from_configuration(configuration, key)
            if value is not None:
                self.values[key].set(value)
            else:
                self.values[key].set(self.get_value_from_configuration(self.configurations.default_configuration, key))

    def save_current_configuration(self):
        new_configuration = copy.deepcopy(self.configurations.default_configuration)
        for key in self.values:
            self.set_value_to_configuration(new_configuration, key, self.values[key].get())
        self.configurations.change_current_configuration(new_configuration)

    @staticmethod
    def get_depth_indicator(depth):
        depth_text_value = '|'
        for _ in range(depth):
            depth_text_value += '--'
        return depth_text_value

    def create_field(self, key, depth, structure_path):
        final_key = ".".join([*structure_path, key])
        Label(self, text=self.get_depth_indicator(depth)).grid(
            row=self.grid_row_last_index, column=0)
        self.labels[final_key] = Label(self, text=configuration_label_dictionary[key]).grid(
            row=self.grid_row_last_index, column=1)
        self.values[final_key] = StringVar()
        self.inputs[final_key] = Entry(self, textvariable=self.values[final_key]).grid(
            row=self.grid_row_last_index, column=2)
        self.grid_row_last_index += 1
        pass

    def get_depth(self, x):
        if type(x) is dict and x:
            return 1 + max(self.get_depth(x[a]) for a in x)
        if type(x) is list and x:
            return 1 + max(self.get_depth(a) for a in x)
        return 0

    def create_group(self, group_data, structure_path=None, current_depth=0):
        if structure_path is None:
            structure_path = []

        if self.get_depth(group_data) > 2:
            for key in group_data:
                self.create_group(group_data[key], [*structure_path, key], current_depth + 1)
        else:
            for key in group_data:
                self.create_field(key, current_depth, structure_path)
