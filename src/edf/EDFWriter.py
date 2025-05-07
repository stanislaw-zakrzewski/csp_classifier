import time
from pyedflib import highlevel
import numpy as np

from config.config import Configurations


class EDFWriter:
    def __init__(self):
        self.configurations = Configurations()
        self.electrode_names = self.configurations.read('general.all_electrodes')
        self.sampling_frequency = self.configurations.read('general.sampling_rate')

    def write(self, signal, start_date, queue, patient_name, gender):
        t = time.localtime()
        timestamp = time.strftime('%Y-%m-%dT%H-%M-%S', t)
        filename = 'data/{}.edf'.format(timestamp)

        sig_headers = highlevel.make_signal_headers(self.electrode_names, sample_rate=self.sampling_frequency,
                                                    physical_max=1000.0,
                                                    physical_min=-1000.0)

        annotations = []
        len_for_annot = 0
        for index, queue_element in enumerate(queue):
            if queue_element[0] != 'break':
                annotations.append([len_for_annot, queue_element[1], queue_element[0]])
            len_for_annot += queue_element[1]

        header = highlevel.make_header(patientname=patient_name, gender=gender,
                                       startdate=start_date)
        header.update({'annotations': annotations})
        clipped_signal = []
        for channel_signal in signal:
            clipped_signal.append(np.clip(channel_signal, -1000.0, 1000.0))
        highlevel.write_edf(filename, clipped_signal, sig_headers, header)
