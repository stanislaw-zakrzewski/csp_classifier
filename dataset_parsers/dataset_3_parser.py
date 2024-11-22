import scipy.io
import numpy as np
import datetime
from mne.channels import make_standard_montage
from pyedflib import highlevel
from mne import create_info
from mne.io import RawArray
import os

event_dict_other = {
    1: 'left hand MI',
    2: 'right hand MI',
    3: 'passive state',
    4: 'left leg MI',
    5: 'tongue MI',
    6: 'right  leg MI',
    99: 'initial relaxation period',
    91: 'inter-session rest break period',
    92: 'experiment end',
}

event_dict_5F = {
    1: 'thumb MI',
    2: 'index finger MI',
    3: 'middle finger MI',
    4: 'ring finger MI',
    5: 'pinkie finger MI',
    99: 'initial relaxation period',
    91: 'inter-session rest break period',
    92: 'experiment end',
}

filepath = r"C:\Users\stz\Documents\Data\A_large_EEG_dataset_for_studying_cross-session_variability_in_motor_imagery_brain-computer_interface"
filepath2 = r"C:\Users\stz\Documents\Data\eeg_data"

# directory = os.fsencode(filepath)

recording_types = {
    '5F': 'FIVE FINGERS',
    'CLA': 'CLASSIC',
    'FREEFORM': 'FREE STYLE 5F',
    'HaLT': 'HAND LEG TONGUE',
    'NoMT': 'NO MOTOR'
}

subject_ids = {
    'SubjectA': {'gender': 'Male', 'age': '20-25'},
    'SubjectB': {'gender': 'Male', 'age': '20-25'},
    'SubjectC': {'gender': 'Male', 'age': '25-30'},
    'SubjectD': {'gender': 'Male', 'age': '25-30'},
    'SubjectE': {'gender': 'Female', 'age': '20-25'},
    'SubjectF': {'gender': 'Male', 'age': '30-35'},
    'SubjectG': {'gender': 'Male', 'age': '30-35'},
    'SubjectH': {'gender': 'Male', 'age': '20-25'},
    'SubjectI': {'gender': 'Female', 'age': '20-25'},
    'SubjectJ': {'gender': 'Female', 'age': '20-25'},
    'SubjectK': {'gender': 'Male', 'age': '20-25'},
    'SubjectL': {'gender': 'Female', 'age': '20-25'},
    'SubjectM': {'gender': 'Female', 'age': '20-25'},
}


def decode_part(subject_name, code_dict):
    for code_entry in code_dict:
        if subject_name.startswith(code_entry):
            return subject_name[len(code_entry):], code_dict[code_entry]


def decode_date(subject_name):
    date_encoded = subject_name[:6]
    date_decoded = datetime.datetime(2000 + int(date_encoded[:2]), int(date_encoded[2:4]), int(date_encoded[4:]))
    return subject_name[6:], date_decoded


def decode_subject_name(subject_name):
    subject_name_to_decode = subject_name.replace('-', '')
    subject_name_to_decode, recording_type = decode_part(subject_name_to_decode, recording_types)
    subject_name_to_decode, subject_data = decode_part(subject_name_to_decode, subject_ids)
    subject_name_to_decode, recording_date = decode_date(subject_name_to_decode)

    return recording_type, subject_data, recording_date


for file in os.listdir(filepath):
    filename = os.fsdecode(file)
    try:
        if filename.endswith(".mat"):
            subject_name = filename[:-4]
            recording_type, subject_data, recording_date = decode_subject_name(subject_name)

            subject_mat = scipy.io.loadmat("{}/{}.mat".format(filepath, subject_name))
            subject_mat = subject_mat['o']
            subject_mat = subject_mat[0][0]

            sfreq = int(subject_mat['sampFreq'][0][0])

            # self.subject_mat_sensor_locations = self.subject_mat['eeg'][0][0]['senloc']
            subject_mat_events = subject_mat['marker'].flatten()

            electrode_names = [ch[0] for ch in list(subject_mat['chnames'].flatten())]
            electrode_names2 = list(make_standard_montage('biosemi64').get_positions()['ch_pos'].keys())
            data = subject_mat['data']
            data = np.transpose(data, (1, 0))
            # self.bad_trials = self.subject_mat['eeg'][0][0]['bad_trial_indices']
            # self.bad_trials_voltage_left = np.asarray(self.subject_mat['eeg'][0][0]['bad_trial_indices'])[0][0][0][0][
            #     0].flatten()
            # self.bad_trials_voltage_right = np.asarray(self.subject_mat['eeg'][0][0]['bad_trial_indices'])[0][0][0][0][
            #     1].flatten()
            # self.bad_trials_mi_left = np.asarray(self.subject_mat['eeg'][0][0]['bad_trial_indices'])[0][0][1][0][
            #     0].flatten()
            # self.bad_trials_mi_right = np.asarray(self.subject_mat['eeg'][0][0]['bad_trial_indices'])[0][0][1][0][
            #     1].flatten()
            if data.shape[0] != len(electrode_names):
                electrode_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'A1', 'A2', 'F7', 'F8',
                                   'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz', 'X3']
            mne_info = create_info(electrode_names, sfreq, 'eeg')
            #
            # self.events = self.generate_events(with_rest, balanced)
            # original_data_left = self.subject_mat['eeg'][0][0]['imagery_left'][0:64]
            # original_data_right = self.subject_mat['eeg'][0][0]['imagery_right'][0:64]
            #
            # data_left, data_right = common_average_reference(original_data_left, original_data_right)
            #
            # #
            # # try:
            # #     data_left, data_right = np.load('preprocessed_data/subjects/{}.npy'.format(subject_name))
            # # except FileNotFoundError:
            # #     preprocess_subject(['{}/{}.mat'.format(eeg_data_path, subject_name)])
            # #     data_left, data_right = np.load('preprocessed_data/subjects/{}.npy'.format(subject_name))
            #
            raw = RawArray(data, mne_info, verbose='critical')
            # raw1 = RawArray(data_left, mne_info)
            # raw2 = RawArray(data_right, mne_info)
            #
            # montage = make_standard_montage('biosemi64')
            # # for i in range(0, 64):
            # #     montage.dig[i + 3].update(r=self.subject_mat_sensor_locations[i])
            #
            # raw1.set_montage(montage)
            # raw2.set_montage(montage)
            # oko = self.events
            # self.raw = concatenate_raws([raw1, raw2])
            # onset = []
            # duration = []
            # description = []
            annotations = []
            previous_event = 0
            current_len = 0
            for event in subject_mat_events:
                if event != 0:
                    event_name = None
                    if recording_type == recording_types['5F']:
                        if event in event_dict_5F:
                            event_name = event_dict_5F[event]
                    else:
                        if event in event_dict_other:
                            event_name = event_dict_other[event]
                    if event_name is not None:
                        if previous_event == event:
                            annotations[-1][1] += 1
                        else:
                            annotations.append([current_len, 1, event_name])
                    previous_event = event
                current_len += 1
            for annotation in annotations:
                annotation[0] /= sfreq
                annotation[1] /= sfreq

            #
            # # annotations = Annotations(onset, duration, description)
            # # self.raw.set_annotations(annotations)
            # self.raw = self.raw.filter(l_freq=.5, h_freq=40, verbose='ERROR')
            # print(self.raw.get_data().shape)
            header = highlevel.make_header(patientname=subject_name, sex=subject_data['gender'],
                                           startdate=recording_date)
            header.update({'annotations': annotations})
            # print(montage.ch_names)
            sig_headers = highlevel.make_signal_headers(electrode_names, sample_rate=sfreq,
                                                        physical_max=10000,
                                                        physical_min=-10000)
            highlevel.write_edf('preprocessed_dataset_3/{}.edf'.format(subject_name), raw.get_data(), sig_headers,
                                header)
            print("PARSED: {}".format(subject_name))
    except Exception as e:
        print('ERROR {}:'.format(subject_name), e)
print("DONE")
