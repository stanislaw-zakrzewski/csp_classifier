from commands.commands import Commands
from threading import *
import tkinter
from tkinter import ttk
from PIL import ImageTk, Image


class VisualState:
    def __init__(self, initial_state):
        self.state = initial_state

    def set_state(self, new_state):
        self.state = new_state

    def get_state(self):
        return self.state


class VisualCommands(Commands):
    def __init__(self):
        self.previous_command = False
        self.visual_state = VisualState('okos')
        t = Thread(target=self.window, args=(self.visual_state,))
        t.start()

    @staticmethod
    def get_image_for_label(label):
        if label == 'left':
            return Image.open("commands//visual_commands//left.jpg")
        elif label == 'right':
            return Image.open("commands//visual_commands//right.png")
        elif label == 'rest':
            return Image.open("commands//visual_commands//rest.png")
        elif label == 'movement':
            return Image.open("commands//visual_commands//movement.png")
        elif label == 'pause':
            return Image.open("commands//visual_commands//pause.jpg")
        elif label == 'end':
            return Image.open("commands//visual_commands//end.jpg")
        return False

    def window(self, vs):
        image1 = self.get_image_for_label('pause') # TODO make sure this is not set always to pause in the beginning

        root = tkinter.Tk()
        big_frame = ttk.Frame(root)
        big_frame.pack(fill='both', expand=True)

        test = ImageTk.PhotoImage(image1)

        label = tkinter.Label(image=test)
        label.image = test
        label.place(relx=0.5, rely=0.5, anchor='center')

        def change_text(label, previous_state):
            current_label_in_state = vs.get_state()
            new_label = label
            if current_label_in_state != previous_state:
                image = self.get_image_for_label(current_label_in_state)
                if image:
                    label.destroy()
                    photo_image = ImageTk.PhotoImage(image)
                    new_label = tkinter.Label(image=photo_image)
                    new_label.image = photo_image
                    new_label.place(relx=0.5, rely=0.5, anchor='center')
            root.after(10, change_text, new_label, current_label_in_state)

        change_text(label, vs.get_state())
        root.geometry('1000x1000')
        root.mainloop()

    def perform_command(self, command_code):
        if self.previous_command != command_code:
            if command_code == 'left':
                self.left()
            elif command_code == 'right':
                self.right()
            elif command_code == 'rest':
                self.rest()
            elif command_code == 'movement':
                self.movement()
            elif command_code == 'pause':
                self.pause()
            elif command_code == 'end':
                self.end()
            self.previous_command = command_code

    def left(self):
        self.visual_state.set_state('left')

    def right(self):
        self.visual_state.set_state('right')

    def rest(self):
        self.visual_state.set_state('rest')

    def movement(self):
        self.visual_state.set_state('movement')

    def pause(self):
        self.visual_state.set_state('pause')

    def end(self):
        self.visual_state.set_state('end')
