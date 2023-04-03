from tkinter import *
from tkinter.ttk import *

from test import DoubleScrolledFrame


def open_annotation_viewer(root, edf_path, annotations):
    AnnotationViewer(root, edf_path, annotations)


class AnnotationViewer(Toplevel):
    def __init__(self, root, edf_path, annotations):
        Toplevel.__init__(self, root)
        self.grab_set()
        self.title("Browse annotations for {}".format(edf_path))
        self.geometry("500x500")

        table = DoubleScrolledFrame(self)

        Label(table, text='Start (s)').grid(row=0, column=0)
        Label(table, text='Length (s)').grid(row=0, column=1)
        Label(table, text='Label').grid(row=0, column=2)

        row_index = 1
        for annotation in annotations:
            Label(table, text=annotation[0]).grid(row=row_index, column=0)
            Label(table, text=annotation[1]).grid(row=row_index, column=1)
            Label(table, text=annotation[2]).grid(row=row_index, column=2)
            row_index += 1
        table.pack(side="top", fill="both", expand=True)
