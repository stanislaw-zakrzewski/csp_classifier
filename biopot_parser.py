import time
from pyedflib import highlevel

from config.config import Configurations
import numpy as np




def save_edf(signal, patient_name, recording_number):
    electrode_names = ['1', '2', '3', '4', '5', '6', '7', '8']
    sampling_frequency = 500


    filename = '{}_{}.edf'.format(patient_name, recording_number)

    sig_headers = highlevel.make_signal_headers(electrode_names, sample_rate=sampling_frequency,
                                                physical_max=1000000.0,
                                                physical_min=-1000000.0)



    header = highlevel.make_header(patientname=patient_name, equipment='biopot3', recording_additional=recording_number)

    highlevel.write_edf(filename, signal, sig_headers, header)

f = open("biopot3/Continuous_Data_14_28_47_ch_values.txt", "r")
lines = f.readlines()
parsed_data = [[],[],[],[],[],[],[],[]]
for line in lines:
    vls = [float(i) for i in line.strip().split(' ')]
    avg = np.average(vls)
    for i in range(8):
        parsed_data[i].append(vls[i] - avg)
save_edf(np.array(parsed_data), 'Bartłomiej Stasiak', 'Nagranie 1')
