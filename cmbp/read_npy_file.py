import numpy as np
from tkinter import filedialog as fd
import mne

a = mne.channels.get_builtin_montages()

file_data = np.load(fd.askopenfilename(), allow_pickle=True)
file_data = file_data.item()
print('oko')
