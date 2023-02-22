import tkinter
from tkinter import ttk
import time
from PIL import ImageTk, Image, ImageSequence

root = tkinter.Tk()
root.geometry('600x400')

def play_gif():
    global img
    img = Image.open("commands//visual_commands//rest.gif")
    lbl = tkinter.Label(root)
    lbl.place(x=0,y=0)

    for img in ImageSequence.Iterator(img):
        img = ImageTk.PhotoImage(img)
        lbl.config(image=img)
        time.sleep(0.1)
        root.update()

tkinter.Button(root, text='play', command=play_gif).place(x=500,y=300)
root.mainloop()