import pyedflib
from pyedflib import highlevel
import numpy as np

filename = 'edf_file.edf'
fs = 250
data = np.random.normal(size=(5,4*60*fs))
ch_names = ['EEG1', 'EEG2', 'EEG3', 'EEG4', 'EEG5']


sig_headers = highlevel.make_signal_headers(ch_names, sample_rate=fs)

annotations = [[2, -1, "welcome"],[30, 20, "welcome"],[5, 10, "welcome"]]
header = highlevel.make_header(patientname='patient_x', gender='Female')
header.update({'annotations':annotations})

highlevel.write_edf(filename, data, sig_headers, header)

signal, signalheaders, header = highlevel.read_edf(filename)
annot = header['annotations']
print(annot)


#
#
# # signals = np.random.rand(5, 256*300)*200 # 5 minutes of random signal
# # channel_names = ['ch1', 'ch2', 'ch3', 'ch4', 'ch5']
# # signal_headers = highlevel.make_signal_headers(channel_names, sample_frequency=250)
# # header = highlevel.make_header(patientname='patient_x', gender='Female')
# # highlevel.write_edf(filename, signals, signal_headers, header)
# w = pyedflib.EdfWriter(filename, 5, file_type=pyedflib.FILETYPE_EDFPLUS)
# w.writeSamples([[1],[1],[1],[1],[1]])
# w.writeAnnotation(1,-1,'witam')
# # w.close()
# pyedflib.EdfReader('edf_file.edf')
# print('oko')


# data =[]
# for i in range(32):
#     data.append([1,2,3,4,5,6,6])
#
# filename = 'example.edf'
# w = pyedflib.EdfWriter(filename, 32, file_type=pyedflib.FILETYPE_EDFPLUS)
# w.writeSamples(data)
#
# # file_name = pyedflib.data.get_generator_filename()
# # f = pyedflib.EdfReader(file_name)
# # n = f.signals_in_file
# # signal_labels = f.getSignalLabels()
# # sigbufs = np.zeros((n, f.getNSamples()[0]))
# # for i in np.arange(n):
# #         sigbufs[i, :] = f.readSignal(i)