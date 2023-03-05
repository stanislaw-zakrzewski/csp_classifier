from tkinter import *
from PIL import Image, ImageTk

from gui.pages.start_page import StartPage
from gui.colors import colors
from gui.fonts import fonts

ELECTRODE_COORDINATES = {
    'Fp1': (479, 185),
    'FpZ': (600, 172),
    'Fp2': (726, 185),

    'AF7': (365, 245),
    'AF3': (463, 265),
    'AFZ': (600, 262),
    'AF4': (745, 266),
    'AF8': (842, 249),

    'F9': (208, 279),
    'F7': (294, 323),
    'F5': (368, 344),
    'F3': (446, 351),
    'F1': (524, 358),
    'Fz': (600, 363),
    'F2': (680, 360),
    'F4': (758, 351),
    'F6': (834, 341),
    'F8': (910, 323),
    'F10': (1001, 280),

    'FCZ': (600, 451),
    'Cz': (600, 545),
    'CPZ': (600, 638),
    'Pz': (600, 737),
    'COZ': (600, 824),
    'OZ': (600, 926),
}


class TestElectrodes(Frame):

    def __init__(self, parent, controller):
        Frame.__init__(self, parent)
        self.config(bg=colors['white_smoke'])

        app_title = Label(self, text="Kombajn EEG", font=fonts['large_bold_font'], bg=colors['white_smoke'])
        app_title.grid(row=0, column=0, padx=10, pady=10, columnspan=10, sticky='W')

        back_to_start_page_button = Button(self, text="Back to Start Page",
                                           command=lambda: controller.show_frame(StartPage))
        back_to_start_page_button.grid(row=2, column=0, padx=10, pady=10, sticky='W')

        test = ImageTk.PhotoImage(file="gui//electrode_placement_filled.png")
        frame2 = Frame(self, bg="red", width=test.width(), height=test.height())
        frame2.grid(row=3, column=0, columnspan=1, rowspan=2)
        frame2.image = test
        electrodes_canvas = Canvas(frame2, width=test.width(), height=test.height(), bg='blue')
        electrodes_canvas.pack(expand=YES, fill=BOTH)
        electrodes_canvas.create_image(2, 2, image=test, anchor=NW)
        self.selected_electrode_code = None
        self.l1 = Label(self, text='Selected electrode:', font=fonts['large_bold_font'])
        self.l1.grid(row=3, column=1)
        self.l = Label(self, text='', font=fonts['large_font'])
        self.l.grid(row=4, column=1)

        # # label1 = Label(myCanvas, image=test)
        # # label1.image = test
        #
        # # Position image
        # # label1.place(x=0, y=0)
        # l = Label(frame2, bg='red', text='oko', width=50, height=50, borderwidth=0)
        # l.corner_radius = 5
        #
        #
        def create_circle(x, y, canvas, tag):  # center coordinates, radius
            r = 35
            x0 = x - r
            y0 = y - r
            x1 = x + r
            y1 = y + r
            hitbox = canvas.create_rectangle(x0, y0, x1, y1, outline='blue', width=0, tags=tag,
                                             stipple='@transparent.xbm', fill='gray')
            circle = canvas.create_oval(x0, y0, x1, y1, outline='red', width=5, tags=tag)
            print(circle)
            return {'hitbox': hitbox, 'circle': circle}

        #
        self.electrode_indicators = {}

        def change_color(new_selected_electrode_code):
            if new_selected_electrode_code == self.selected_electrode_code:
                electrodes_canvas.itemconfig(self.electrode_indicators[new_selected_electrode_code]['circle'], width=5)
                self.selected_electrode_code = None
                self.l.config(text='')
                self.l.config(text='')
            else:
                if self.selected_electrode_code is not None:
                    electrodes_canvas.itemconfig(self.electrode_indicators[self.selected_electrode_code]['circle'],
                                                 width=5)
                electrodes_canvas.itemconfig(self.electrode_indicators[new_selected_electrode_code]['circle'], width=10)
                self.selected_electrode_code = new_selected_electrode_code
                self.l.config(text=self.selected_electrode_code)

        for electrode in ELECTRODE_COORDINATES:
            electrode_x, electrode_y = ELECTRODE_COORDINATES[electrode]
            self.electrode_indicators[electrode] = create_circle(electrode_x, electrode_y, electrodes_canvas, electrode)

            electrodes_canvas.tag_bind(electrode, "<Button-1>", lambda event='', dup_el=electrode: change_color(dup_el))

            # Button(frame2, text="Change Color", command=change_color).place(x=600, y=600)
        # l.place(x=600,y=600)


