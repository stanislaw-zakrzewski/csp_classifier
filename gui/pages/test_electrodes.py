from tkinter import *
from PIL import Image, ImageTk

from config.config import Configurations
from gui.pages.start_page import StartPage
from gui.colors import colors
from gui.fonts import fonts
from gui.components.double_scrolled_frame import DoubleScrolledFrame

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
    'FZ': (600, 363),
    'F2': (680, 360),
    'F4': (758, 351),
    'F6': (834, 341),
    'F8': (910, 323),
    'F10': (1001, 280),

    'FT9': (148, 414),
    'FT7': (240, 427),
    'FC5': (332, 435),
    'FC3': (425, 442),
    'FC1': (517, 449),
    'FCZ': (600, 451),
    'FC2': (687, 450),
    'FC4': (778, 442),
    'FC6': (876, 435),
    'FT8': (963, 426),
    'FT10': (1053, 414),

    'T9': (143, 544),
    'T7': (221, 544),
    'C5': (311, 543),
    'C3': (412, 544),
    'C1': (506, 547),
    'CZ': (600, 545),
    'C2': (689, 545),
    'C4': (792, 545),
    'C6': (886, 546),
    'T8': (982, 544),
    'T10': (1066, 542),

    'TP9': (143, 671),
    'TP7': (239, 652),
    'CP5': (336, 645),
    'CP3': (432, 642),
    'CP1': (517, 639),
    'CPZ': (600, 638),
    'CP2': (691, 639),
    'CP4': (777, 641),
    'CP6': (868, 645),
    'TP8': (964, 652),
    'TP10': (1060, 672),

    'P9': (196, 794),
    'P7': (288, 766),
    'P5': (366, 752),
    'P3': (446, 743),
    'P1': (526, 738),
    'PZ': (600, 737),
    'P2': (679, 738),
    'P4': (759, 742),
    'P6': (835, 751),
    'P8': (912, 766),
    'P10': (1007, 795),

    'PO7': (377, 853),
    'PO3': (473, 827),
    'POZ': (600, 824),
    'PO4': (722, 827),
    'PO8': (830, 853),

    'O1': (482, 906),
    'OZ': (600, 926),
    'O2': (722, 905),
}


class TestElectrodes(DoubleScrolledFrame):

    def __init__(self, parent, controller):
        DoubleScrolledFrame.__init__(self, parent)
        self.config(bg=colors['white_smoke'])
        self.configurations = Configurations()
        self.selected_electrodes = self.configurations.read('all.general.selected_electrodes')

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
            if electrode in self.selected_electrodes:
                electrode_x, electrode_y = ELECTRODE_COORDINATES[electrode]
                self.electrode_indicators[electrode] = create_circle(electrode_x, electrode_y, electrodes_canvas, electrode)

                electrodes_canvas.tag_bind(electrode, "<Button-1>", lambda event='', dup_el=electrode: change_color(dup_el))

            # Button(frame2, text="Change Color", command=change_color).place(x=600, y=600)
        # l.place(x=600,y=600)


