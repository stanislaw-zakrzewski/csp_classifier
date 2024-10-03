from data_classes.subject import Subject
from tkinter import filedialog as fd
from pyedflib import highlevel
from mne.filter import filter_data

subject_path = fd.askopenfilename(filetypes=[("European Data Format files", "*.edf")])
subject = Subject(subject_path)

filtered_signal = filter_data(subject.signals, subject.sampling_frequency, 12, 20)
# print('filter')
# filtered_signal.filter(2, 24, l_trans_bandwidth=2, h_trans_bandwidth=2, filter_length=subject.sampling_frequency * 2,
#                        fir_design='firwin',
#                        skip_by_annotation='edge', )
# print('done')

highlevel.write_edf('tmp.edf', filtered_signal, subject.signal_headers, subject.header)
