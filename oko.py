# Import Module
from tkinter import *
import time
from threading import *
import time
import tkinter
from tkinter import ttk

# Create Object
from commands.audio_commands import AudioCommands

vc = AudioCommands()
vc.perform_command('rest')
time.sleep(2)
vc.perform_command('movement')
time.sleep(2)
vc.perform_command('left')
time.sleep(2)
vc.perform_command('right')
time.sleep(2)
vc.perform_command('pause')
time.sleep(2)
vc.perform_command('end')
#
# # use threading
#
# def threading():
#     # Call work function
#     t1 = Thread(target=work)
#     t1.start()
#
#
# # work function
# def work():
#     print("sleep time start")
#
#     for i in range(10):
#         print(i)
#         time.sleep(1)
#
#     print("sleep time stop")
#
# r = Tk()
# # Create Button
# class VisualState():
#     def __init__(self, initial_state):
#         self.state = initial_state
#     def set_state(self, new_state):
#         self.state = new_state
#     def get_state(self):
#         return self.state
#
# vs = VisualState('okos')
#
# def window(vs):
#
#
#     # this must return soon after starting this
#     def change_text():
#         label['text'] = vs.get_state()
#
#         # now we need to run this again after one second, there's no better
#         # way to do this than timeout here
#         root.after(1000, change_text)
#
#     root = tkinter.Tk()
#     big_frame = ttk.Frame(root)
#     big_frame.pack(fill='both', expand=True)
#
#     label = ttk.Label(big_frame, text='0')
#     label.pack()
#
#     change_text()  # don't forget to actually start it :)
#
#     root.geometry('200x200')
#     root.mainloop()
#
#
#     # root.geometry("400x400")
#     #
#     # Button(root, text="Click Me", command=threading).pack()
#     #
#     # # Execute Tkinter
#     # root.mainloop()
#
# t = Thread(target=window, args=(vs,))
# t.start()
# time.sleep(1)
# vs.set_state('2')
# print('oko')
#
#



