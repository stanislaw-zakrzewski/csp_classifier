from tkinter import *

from gui.colors import colors
from gui.fonts import fonts

GRID_COLUMN_THRESHOLD = 4


class StartPage(Frame):
    def __init__(self, parent, controller, pages):
        Frame.__init__(self, parent)
        self.config(bg=colors['white_smoke'])

        app_title = Label(self, text="Kombajn EEG", font=fonts['large_bold_font'], bg=colors['white_smoke'])
        app_title.grid(row=0, column=0, padx=10, pady=10, columnspan=10, sticky='W')

        current_grid_row = 1
        current_grid_column = 0
        self.buttons = {}
        for page in pages:

            frame = page['frame']
            self.buttons[page['name']] = Button(self, text=page['name'],
                                                command=lambda bound_frame=frame: controller.show_frame(bound_frame),
                                                width=30,
                                                height=3, font=fonts['large_font'])

            self.buttons[page['name']].grid(row=current_grid_row, column=current_grid_column, padx=10, pady=10)
            current_grid_column += 1
            if current_grid_column > GRID_COLUMN_THRESHOLD:
                current_grid_row += 1
                current_grid_column = 0
