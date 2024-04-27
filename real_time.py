import numpy as np
from mne.io import RawArray

import SenderLib
import pygds
from classifiers.flat import process
from config.config import Configurations
from config_old import channels2
from data_classes.subject import Subject
from playsound import playsound

SUBJECT_TO_TRAIN = 'data/2024-03-04T13-11-25.edf'

configurations = Configurations()
electrode_names = configurations.read('general.all_electrodes')
bandpass_filter_start_frequency = configurations.read('real_time.bandpass_filter_start_frequency')
bandpass_filter_end_frequency = configurations.read('real_time.bandpass_filter_end_frequency')
ipaddress = configurations.read('collect_data.ipaddress')
port = configurations.read('collect_data.port')


def calculate_recall(predictions, corrects, hand):
    numerator = 0
    denominator = 0
    for i in range(len(predictions)):
        if hand == corrects[i]:  # Actual condition: Positive
            if predictions[i] == corrects[i]:  # Predicted condition: Positive
                numerator += 1  # TP
            denominator += 1  # TP + FN
    if denominator == 0: return 0, 0
    return numerator, denominator


def calculate_precision(predictions, corrects, hand):
    numerator = 0
    denominator = 0
    for i in range(len(predictions)):
        if hand == predictions[i]:  # Predicted condition: Positive
            if predictions[i] == corrects[i]:  # Actual condition: Positive
                numerator += 1  # TP
            denominator += 1  # TP + FP
    if denominator == 0: return 0, 0
    return numerator, denominator


def main(bands, channels):
    precision_numerator = [0, 0]
    precision_denominator = [0, 0]
    recall_numerator = [0, 0]
    recall_denominator = [0, 0]
    subject = Subject(SUBJECT_TO_TRAIN)

    window_times, window_scores, csp_filters, epochs_info, predictions, corrects, classifier, mne_info = process(
        subject, bands,
        channels,
        n_splits=1)

    # update_predictions = []
    # for i in range(len(predictions)):
    #     update_predictions.append([])
    #     for j in range(len(predictions[i])):
    #         if predictions[i][j] == 0:
    #             update_predictions[i].append(1)
    #         else:
    #             update_predictions[i].append(0)
    # predictions = update_predictions

    for i in range(len(predictions)):
        for j in range(2):
            nprec, dprec = calculate_precision(predictions[i], corrects[i], j)
            nrec, drec = calculate_recall(predictions[i], corrects[i], j)
            precision_numerator[j] = precision_numerator[j] + nprec
            precision_denominator[j] = precision_denominator[j] + dprec
            recall_numerator[j] = recall_numerator[j] + nrec
            recall_denominator[j] = recall_denominator[j] + drec

    accuracy_nominator = 0
    accuracy_denumerator = 0
    for i in range(len(predictions)):
        for j in range(len(predictions[i])):
            accuracy_denumerator += 1
            if predictions[i][j] == corrects[i][j]:
                accuracy_nominator += 1

    print('Accuracy: {}'.format(accuracy_nominator / accuracy_denumerator))

    final_precision = [0, 0]
    final_recall = [0, 0]
    for i in range(2):
        if precision_denominator[i] == 0:
            final_precision[i] = 0
        else:
            final_precision[i] = precision_numerator[i] / precision_denominator[i]

    for i in range(2):
        if recall_denominator[i] == 0:
            final_recall[i] = 0
        else:
            final_recall[i] = recall_numerator[i] / recall_denominator[i]

    print('Combined precision for movement class: {}'.format(final_precision[0]))
    print('Combined precision for rest class: {}'.format(final_precision[1]))

    print('Combined recall for movement class: {}'.format(final_recall[0]))
    print('Combined recall for rest class: {}'.format(final_recall[1]))

    return csp_filters, classifier, mne_info


csp, lda, mne_info = main(
    [(bandpass_filter_start_frequency, bandpass_filter_end_frequency)],
    channels2
)
print(csp, lda)

print("Connecting to VR device...")

print(configurations.read('real_time.vr_audio_prompts'))

sender = SenderLib.Sender(ipaddress, port)
control = SenderLib.GameControl()
print("Successfully connected to VR device")

print("Inicjalizacja trochę trwa...")
d = pygds.GDS()
pygds.configure_demo(d)
supported_sensitivities = d.GetSupportedSensitivities()
sensitivity_id = 0
for ch in d.Channels:
    ch.Sensitivity = supported_sensitivities[0][sensitivity_id]
    ch.BandpassFilterIndex = 16  # 2-30Hz bandpass
d.SetConfiguration()
i = 0


def processCallback(samples):
    # samples = raw_samples[:,0:32]
    global i
    i += 1
    try:
        ret = []
        for _ in range(32):
            ret.append([])
        for sample in samples:
            for index, channel in enumerate(sample):
                if index < 32:
                    ret[index].append(channel)
        raw = RawArray(ret, mne_info, verbose='CRITICAL')

        for channel in electrode_names:
            if channel not in channels2:
                raw.drop_channels([channel])
        raw.filter(bandpass_filter_start_frequency, bandpass_filter_end_frequency, l_trans_bandwidth=2,
                   h_trans_bandwidth=2, filter_length=500, fir_design='firwin',
                   skip_by_annotation='edge', verbose='CRITICAL')
        flt = raw.get_data()
        res = lda.predict(csp.transform(np.array([flt])))
        print(res[0])
        if res[0] == 1:
            print('Movement')
            if configurations.read('real_time.vr_audio_prompts'):
                playsound('commands//sound_commands//ruch.wav')
            control.left = True
            control.right = True
            control.mode = configurations.read('collect_data.vr_mode')
            state = sender.send_data(control)
        else:
            print('Rest')
            if configurations.read('real_time.vr_audio_prompts'):
                playsound('commands//sound_commands//brak.wav')
            control.left = False
            control.right = False
            control.mode = configurations.read('collect_data.vr_mode')
            state = sender.send_data(control)

        # if send_to_vr:
        #     state = sender.send_data(control)

    except Exception as e:
        print(e)

    if i < 20: return True
    return False


a = d.GetData(d.SamplingRate * 2, processCallback)
d.Close()
del d
