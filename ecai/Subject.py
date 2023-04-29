import numpy as np
from mne import create_info, Annotations
from mne.channels import make_standard_montage
from mne.export import export_raw
from mne.io import concatenate_raws, RawArray
from pyedflib import highlevel
from scipy.io import loadmat
from pathlib import Path
import matplotlib.pyplot as plt

from ecai.config import eeg_data_path


def preprocess_subject(subject_paths):
    subject_count = len(subject_paths)
    Path("preprocessed_data/subjects").mkdir(parents=True, exist_ok=True)
    for index, subject_path in enumerate(subject_paths):
        print("Started processing {} of {} subjects".format(index + 1, subject_count))

        subject = loadmat(subject_path)
        data_left = subject['eeg'][0][0]['imagery_left'][0:64]
        data_right = subject['eeg'][0][0]['imagery_right'][0:64]

        averages_left = []
        averages_right = []
        for i in range(64):
            averages_left.append([])
            averages_right.append([])

        for i in range(data_left.shape[1]):
            # TODO fix this to make it faster
            # concatenate((averages_left, full(64, [average(data_left[:, i])])), axis=1)
            # concatenate((averages_right, full(64, [average(data_right[:, i])])), axis=1)
            average_left = np.average(data_left[:, i])
            average_rigt = np.average(data_right[:, i])
            for j in range(64):
                averages_left[j].append(average_left)
                averages_right[j].append(average_rigt)

        data_left -= averages_left
        data_right -= averages_right
        np.save('preprocessed_data/subjects/{}'.format(subject_path.split('.')[-2].split('/')[-1]),
                [data_left, data_right])
        print("Finished processing {} of {} subjects".format(index + 1, subject_count))



class Subject:
    def __init__(self, subject_name, filepath, with_rest=False, balanced=False):

        self.subject_mat = loadmat("{}/{}.mat".format(filepath, subject_name))
        self.subject_mat_sensor_locations = self.subject_mat['eeg'][0][0]['senloc']
        self.subject_mat_events = self.subject_mat['eeg'][0][0]['imagery_event']

        self.electrode_names = list(make_standard_montage('biosemi64').get_positions()['ch_pos'].keys())
        self.bad_trials = self.subject_mat['eeg'][0][0]['bad_trial_indices']
        self.bad_trials_voltage_left = np.asarray(self.subject_mat['eeg'][0][0]['bad_trial_indices'])[0][0][0][0][
            0].flatten()
        self.bad_trials_voltage_right = np.asarray(self.subject_mat['eeg'][0][0]['bad_trial_indices'])[0][0][0][0][
            1].flatten()
        self.bad_trials_mi_left = np.asarray(self.subject_mat['eeg'][0][0]['bad_trial_indices'])[0][0][1][0][
            0].flatten()
        self.bad_trials_mi_right = np.asarray(self.subject_mat['eeg'][0][0]['bad_trial_indices'])[0][0][1][0][
            1].flatten()
        mne_info = create_info(self.electrode_names, 512, 'eeg')

        self.events = self.generate_events(with_rest, balanced)
        data_left = self.subject_mat['eeg'][0][0]['imagery_left'][0:64]
        data_right = self.subject_mat['eeg'][0][0]['imagery_right'][0:64]


        #
        # try:
        #     data_left, data_right = np.load('preprocessed_data/subjects/{}.npy'.format(subject_name))
        # except FileNotFoundError:
        #     preprocess_subject(['{}/{}.mat'.format(eeg_data_path, subject_name)])
        #     data_left, data_right = np.load('preprocessed_data/subjects/{}.npy'.format(subject_name))

        raw1 = RawArray(data_left, mne_info)
        raw2 = RawArray(data_right, mne_info)

        montage = make_standard_montage('biosemi64')
        # for i in range(0, 64):
        #     montage.dig[i + 3].update(r=self.subject_mat_sensor_locations[i])

        raw1.set_montage(montage)
        raw2.set_montage(montage)
        oko = self.events
        self.raw = concatenate_raws([raw1, raw2])
        onset = []
        duration = []
        description = []
        annotations = []
        for event in self.events:
            # onset.append(int((event[0]+1)/512))
            # duration.append(5)
            if event[2] == 0:
                # description.append('left')
                annotations.append([(event[0]+1)/512, 5, 'left'])

            else:
                # description.append('right')
                annotations.append([(event[0]+1)/512, 5, 'right'])

        # annotations = Annotations(onset, duration, description)
        # self.raw.set_annotations(annotations)
        self.raw = self.raw.filter(l_freq=.5, h_freq=40, verbose='ERROR')
        print(self.raw.get_data().shape)
        header = highlevel.make_header(patientname=subject_name)
        header.update({'annotations': annotations})
        print(montage.ch_names)
        sig_headers = highlevel.make_signal_headers(montage.ch_names, sample_rate=512,
                                                    physical_max=2000000,
                                                    physical_min=-2000000)
        highlevel.write_edf('preprocessed_subjects/{}.edf'.format(subject_name), self.raw.get_data(), sig_headers, header)




        # raw_test = self.raw.crop(0, 7)
        # raw_test = raw_test.filter(l_freq=.5, h_freq=40, verbose='ERROR')
        # raw_test.plot(block=True, lowpass=40, highpass=.5, show_options=True)
        #
        # plt.show()

    def get_raw_copy(self):
        return self.raw.copy()

    def generate_events(self, with_rest=False, balanced=False):
        event_localizations = np.where(self.subject_mat_events == 1)[1]
        data = []
        breaker = False
        for index, value in enumerate(event_localizations):
            if index not in self.bad_trials_mi_left and index not in self.bad_trials_voltage_left:
                if with_rest:
                    if breaker or not balanced:
                        data.append([value - 1023, 0, 0])
                        breaker = False
                    else:
                        breaker = True
                    data.append([value + 512, 0, 1])
                else:
                    data.append([value, 0, 0])
        for index, value in enumerate(event_localizations):
            if index not in self.bad_trials_mi_right and index not in self.bad_trials_voltage_right:
                if with_rest:
                    if breaker or not balanced:
                        data.append([value + len(self.subject_mat_events[0]) - 1023, 0, 0])
                        breaker = False
                    else:
                        breaker = True
                    data.append([value + len(self.subject_mat_events[0]) + 512, 0, 2])
                else:
                    data.append([value + len(self.subject_mat_events[0]), 0, 1])
        return data

for i in range(9):
    print(i+1)
    Subject( 's0{}'.format(i+1), eeg_data_path)
for i in range(43):
    print(i+10)
    Subject( 's{}'.format(i+10), eeg_data_path)
