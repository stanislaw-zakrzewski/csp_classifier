import time

from commands.commands import Commands
from threading import *
import tkinter
from tkinter import ttk
from PIL import ImageTk, Image, ImageSequence


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
        self.gif_states = ['left', 'right', 'movement']

    @staticmethod
    def get_image_for_label(label):
        if label == 'left':
            return Image.open("commands//visual_commands//left.gif")
        elif label == 'right':
            return Image.open("commands//visual_commands//right.gif")
        elif label == 'rest':
            return Image.open("commands//visual_commands//rest.png")
        elif label == 'movement':
            return Image.open("commands//visual_commands//movement.gif")
        elif label == 'pause':
            return Image.open("commands//visual_commands//pause.jpg")
        elif label == 'end':
            return Image.open("commands//visual_commands//end.jpg")
        return False

    def window(self, vs):
        image1 = self.get_image_for_label('left') # TODO make sure this is not set always to pause in the beginning
        self.visual_state.set_state('left')
        root = tkinter.Tk()
        big_frame = ttk.Frame(root)
        big_frame.pack(fill='both', expand=True)

        photo_image = ImageTk.PhotoImage(image1)

        global label
        label= tkinter.Label(image=photo_image)
        label.image = photo_image
        label.place(relx=0.5, rely=0.5, anchor='center')



        def change_text(label, image):
            previous_state = False
            while True:
                current_label_in_state = vs.get_state()

                if current_label_in_state != previous_state:
                    image = self.get_image_for_label(current_label_in_state)
                    if image:
                        # # label.destroy()
                        # photo_image = ImageTk.PhotoImage(image)
                        # # new_label = tkinter.Label(image=photo_image)
                        # # new_label.image = photo_image
                        # # new_label.place(relx=0.5, rely=0.5, anchor='center')
                        # label.config(image=ImageTk.PhotoImage(self.get_image_for_label(current_label_in_state)))
                        # root.update()
                        i = self.get_image_for_label(current_label_in_state)
                        i = i.resize((i.width * 3, i.height * 3), Image.LANCZOS)
                        photo_image = ImageTk.PhotoImage(i)

                        label.config(image=photo_image)
                        label.image = photo_image
                        # root.update()
                        previous_state = current_label_in_state

                if current_label_in_state in self.gif_states:
                    for img in ImageSequence.Iterator(image):
                        if current_label_in_state != vs.get_state():
                            break
                        resized = img.resize((img.width * 3, img.height * 3), Image.LANCZOS)
                        img = ImageTk.PhotoImage(resized)
                        label.config(image=img)
                        label.image = img
                        time.sleep(1/20) # powinno być 1/30, tak jest, żeby wolniej było
                        # root.update()
                # else:
                #     image = self.get_image_for_label(current_label_in_state)
                #     print(image)
                #     if image:
                #         # label.destroy()
                #         photo_image = ImageTk.PhotoImage(image)
                #         # new_label = tkinter.Label(image=photo_image)
                #         # new_label.image = photo_image
                #         # new_label.place(relx=0.5, rely=0.5, anchor='center')
                #         label.config(image=ImageTk.PhotoImage(self.get_image_for_label(current_label_in_state)))
                #         root.update()
                #         label.config(image=ImageTk.PhotoImage(self.get_image_for_label(current_label_in_state)))
                #         root.update()

            # root.after(10, change_text, label, current_label_in_state, image)


        # change_text(label, vs.get_state(), image1)
        root.geometry('1800x1014')
        root.configure(background="red")
        # play_gif(label)
        audio_thread = Thread(target=change_text, args=(label, image1,))  # create thread
        audio_thread.start()
        # root.update()
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
