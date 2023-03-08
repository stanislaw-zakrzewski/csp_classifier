from tkinter import *
from tkinter.ttk import *


def open_annotation_viewer(root, edf_path, annotations):
    AnnotationViewer(root, edf_path, annotations)


class AnnotationViewer(Toplevel):
    def __init__(self, root, edf_path, annotations):
        Toplevel.__init__(self, root)
        self.grab_set()

        # sets the title of the
        # Toplevel widget
        self.title("Browse annotations for {}".format(edf_path))

        # sets the geometry of toplevel
        self.geometry("500x500")



        # table = Frame(self)
        # myscrollbar = Scrollbar(table, orient=VERTICAL)
        # myscrollbar.grid(row=0,column=3, rowspan=100, sticky=NS)
        #
        # print(annotations)
        # Label(table, text='Start (s)').grid(row=0, column=0)
        # Label(table, text='Length (s)').grid(row=0, column=1)
        # Label(table, text='Label').grid(row=0, column=2)
        #
        # row_index = 1
        # for annotation in annotations:
        #     Label(table, text=annotation[0]).grid(row=row_index, column=0)
        #     Label(table, text=annotation[1]).grid(row=row_index, column=1)
        #     Label(table, text=annotation[2]).grid(row=row_index, column=2)
        #     row_index += 1
        # table.pack()

        # # Create a frame for the canvas with non-zero row&column weights
        # frame_canvas = Frame(self)
        # frame_canvas.grid(row=0, column=0, sticky='nw')
        # frame_canvas.grid_rowconfigure(0, weight=1)
        # frame_canvas.grid_columnconfigure(0, weight=1)
        # # Set grid_propagate to False to allow 5-by-5 buttons resizing later
        # frame_canvas.grid_propagate(False)

        # Add a canvas in that frame
        canvas = Canvas(self, height=500, width=450)
        canvas.grid(row=0, column=0, sticky="news")

        # Link a scrollbar to the canvas
        vsb = Scrollbar(self, orient="vertical", command=canvas.yview)
        vsb.grid(row=0, column=1, sticky='ns')
        canvas.configure(yscrollcommand=vsb.set)

        # Create a frame to contain the buttons
        table = Frame(canvas)
        canvas.create_window((0, 0), window=table, anchor='nw')

        Label(table, text='Start (s)').grid(row=0, column=0)
        Label(table, text='Length (s)').grid(row=0, column=1)
        Label(table, text='Label').grid(row=0, column=2)

        row_index = 1
        for annotation in annotations:
            Label(table, text=annotation[0]).grid(row=row_index, column=0)
            Label(table, text=annotation[1]).grid(row=row_index, column=1)
            Label(table, text=annotation[2]).grid(row=row_index, column=2)
            row_index += 1


        # Add 9-by-5 buttons to the frame
        # rows = 9
        # columns = 5
        # buttons = [[Button() for j in range(columns)] for i in range(rows)]
        # for i in range(0, rows):
        #     for j in range(0, columns):
        #         buttons[i][j] = Button(frame_buttons, text=("%d,%d" % (i + 1, j + 1)))
        #         buttons[i][j].grid(row=i, column=j, sticky='news')

        # Update buttons frames idle tasks to let tkinter calculate buttons sizes
        table.update_idletasks()

        # Resize the canvas frame to show exactly 5-by-5 buttons and the scrollbar
        # first5columns_width = sum([buttons[0][j].winfo_width() for j in range(0, 5)])
        # first5rows_height = sum([buttons[i][0].winfo_height() for i in range(0, 5)])
        # frame_canvas.config(width=300,
        #                     height=300)

        # Set the canvas scrolling region
        canvas.config(scrollregion=canvas.bbox("all"))

