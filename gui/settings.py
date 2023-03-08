import copy
from tkinter import *

from config.config import Configurations
from config.configuration_label_dictionary import configuration_label_dictionary


class Settings(Toplevel):
    def __init__(self, root):
        Toplevel.__init__(self, root)
        self.configurations = Configurations()
        self.title("Settings")
        self.geometry("600x800")
        self.grab_set()

        self.grid_row_last_index = 0
        self.labels = {}
        self.values = {}
        self.string_values = {}
        self.new_values = {}
        self.inputs = {}

        self.create_group(self.configurations.default_configuration)

        button_frame = Frame(self)
        button_frame.grid(row=self.grid_row_last_index, column=0, columnspan=10)

        self.restore_configuration(self.configurations.current_configuration)

        self.discard_changes_button = Button(button_frame, text='Discard changes',
                                             command=lambda: self.restore_configuration(
                                                 self.configurations.current_configuration))
        self.discard_changes_button.grid(row=self.grid_row_last_index, column=1, padx=10, pady=10)

        self.restore_default_button = Button(button_frame, text='Restore default configuration',
                                             command=lambda: self.restore_configuration(
                                                 self.configurations.default_configuration))
        self.restore_default_button.grid(row=self.grid_row_last_index, column=2, padx=10, pady=10)

        self.save_configuration_button = Button(button_frame, text='Save configuration',
                                                command=self.save_current_configuration)
        self.save_configuration_button.grid(row=self.grid_row_last_index, column=3, padx=10, pady=10)

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
                self.values[key] = copy.deepcopy(value)
                self.string_values[key].set(value)
            else:
                self.values[key] = copy.deepcopy(
                    self.get_value_from_configuration(self.configurations.default_configuration, key))
                self.string_values[key].set(
                    self.get_value_from_configuration(self.configurations.default_configuration, key))

    def save_current_configuration(self):
        new_configuration = copy.deepcopy(self.configurations.default_configuration)
        for key in self.values:
            self.set_value_to_configuration(new_configuration, key, self.values[key])
        self.configurations.change_current_configuration(new_configuration)

    @staticmethod
    def get_text_with_depth(depth, text):
        depth_text_value = ''
        for _ in range(depth):
            depth_text_value += '  '
        return depth_text_value + text

    def create_text_entry(self, final_key):
        self.values[final_key] = None
        self.string_values[final_key] = StringVar()
        self.inputs[final_key] = Entry(self, textvariable=self.string_values[final_key])
        self.inputs[final_key].grid(row=self.grid_row_last_index, column=1, sticky=W)

    def create_list_entry(self, final_key, values):
        f = Frame(self)
        f.grid(row=self.grid_row_last_index, column=1)

        self.string_values[final_key] = StringVar(value=values)
        box = Listbox(f, listvariable=self.string_values[final_key])
        box.grid(row=0, column=0, rowspan=4)
        self.values[final_key] = copy.deepcopy(values)
        self.new_values[final_key] = StringVar(value='')
        new_value_entry = Entry(f, textvariable=self.new_values[final_key])
        new_value_entry.grid(row=0, column=1)

        def add():
            if self.new_values[final_key].get() != '':
                self.values[final_key].append(self.new_values[final_key].get())
                self.string_values[final_key].set(self.values[final_key])
                self.new_values[final_key].set('')

        b = Button(f, text='Add', command=add)
        b.grid(row=0, column=2)

        def remove():
            selected = box.curselection()
            for index_to_delete in selected:
                del self.values[final_key][index_to_delete]
            self.string_values[final_key].set(self.values[final_key])

        b = Button(f, text='Remove Selected', command=remove)
        b.grid(row=3, column=1)

    def create_field(self, key, depth, structure_path, val):
        final_key = ".".join([*structure_path, key])
        label_text = self.get_text_with_depth(depth, configuration_label_dictionary[key])

        self.labels[final_key] = Label(self, text=label_text, anchor="e").grid(
            row=self.grid_row_last_index, column=0, sticky='W')
        if type(val['value']) is list:
            self.create_list_entry(final_key, val['value'])
        else:
            self.create_text_entry(final_key)
        # self.values[final_key] = StringVar()
        # self.inputs[final_key] = Entry(self, textvariable=self.values[final_key]).grid(
        #     row=self.grid_row_last_index, column=1)
        self.grid_row_last_index += 1
        pass

    def get_depth(self, x):
        if type(x) is dict and x:
            if 'type' in x.keys():
                return 1
            return 1 + max(self.get_depth(x[a]) for a in x)
        if type(x) is list and x:
            return 1 + max(self.get_depth(a) for a in x)
        return 0

    def create_group(self, group_data, structure_path=None, current_depth=-1, current_key=None):
        if current_key is not None:
            Label(self, text=self.get_text_with_depth(current_depth, configuration_label_dictionary[current_key]),
                  font="SegoeUI 9 bold").grid(row=self.grid_row_last_index, column=0, sticky='W')
            self.grid_row_last_index += 1

        if structure_path is None:
            structure_path = []

        if self.get_depth(group_data) > 2:
            for key in group_data:
                self.create_group(group_data[key], [*structure_path, key], current_depth + 1, key)
        else:
            for key in group_data:
                self.create_field(key, current_depth, structure_path, group_data[key])
