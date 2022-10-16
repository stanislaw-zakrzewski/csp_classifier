import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
import seaborn as sns
from mne import Epochs, pick_types
from mne.decoding import CSP, UnsupervisedSpatialFilter
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from config import eeg_data_path
from data_classes.subject import Subject


def process_hierarchical(subject_name, bands, selected_channels, reg=None):
    tmin, tmax = 0., 2.
    event_1_id = dict(idle=0, imagery_movement=1)
    event_2_id = dict(left=1, right=2)
    subject = Subject()

    raw_signals = []
    for i in range(len(bands)):
        raw_signals.append(subject.get_raw_copy())

    if len(selected_channels) > 0:
        for raw_signal in raw_signals:
            for channel in subject.electrode_names:
                if channel not in selected_channels:
                    raw_signal.drop_channels([channel])

    filtered_raw_signals = []
    epochs_1 = []
    epochs_1_train = []
    epochs_1_data = []
    epochs_1_data_train = []
    epochs_2 = []
    epochs_2_train = []
    epochs_2_data = []
    epochs_2_data_train = []

    # Apply band-pass filter
    for index, band in enumerate(bands):
        filtered_raw_signals.append(
            raw_signals[index].filter(band[0], band[1], l_trans_bandwidth=.5, h_trans_bandwidth=.5, fir_design='firwin',
                                      skip_by_annotation='edge'))

    picks = pick_types(filtered_raw_signals[0].info, meg=False, eeg=True, stim=False, eog=False,
                       exclude='bads')

    all_labels = np.array([])
    for index, band in enumerate(bands):
        events_1 = []
        events_2 = []
        for i in subject.events:
            all_labels = np.append(all_labels, int(i[2]))
            if i[2] != 0:
                events_2.append(i)
            if i[2] == 2:
                events_1.append([i[0], i[1], 1])
            else:
                events_1.append(i)
        epochs_1.append(
            Epochs(filtered_raw_signals[index], events_1, event_1_id, tmin, tmax, proj=True, picks=picks,
                   baseline=None, preload=True))
        epochs_2.append(
            Epochs(filtered_raw_signals[index], events_2, event_2_id, tmin, tmax, proj=True, picks=picks,
                   baseline=None, preload=True))
        epochs_1_train.append(epochs_1[index].copy().crop(tmin=0., tmax=2.))
        epochs_2_train.append(epochs_2[index].copy().crop(tmin=0., tmax=2.))

        epochs_1_data.append(epochs_1[index].get_data())
        epochs_2_data.append(epochs_2[index].get_data())
        epochs_1_data_train.append(epochs_1_train[index].get_data())
        epochs_2_data_train.append(epochs_2_train[index].get_data())
    labels_1 = np.array(epochs_1[0].events[:, -1])
    labels_2 = np.array(epochs_2[0].events[:, -1])

    cv = ShuffleSplit(10, test_size=0.2, random_state=23)
    cv_split = cv.split(epochs_1_data_train[0])

    # Assemble a classifier
    # classifier = MLPClassifier(hidden_layer_sizes=(20, 20), random_state=1,
    #                            max_iter=1000)  # Originally: LinearDiscriminantAnalysis()
    classifier_1 = LinearDiscriminantAnalysis()
    classifier_2 = LinearDiscriminantAnalysis()
    csp_n_components = 10 if len(selected_channels) == 0 else min(len(selected_channels), 10)
    csp_1 = CSP(n_components=csp_n_components, reg=reg, log=True, norm_trace=False)
    csp_2 = CSP(n_components=csp_n_components, reg=reg, log=True, norm_trace=False)

    sfreq = raw_signals[0].info['sfreq']
    w_length = int(sfreq)  # running classifier: window length
    w_step = int(sfreq * 0.1)  # running classifier: window step size
    w_start = np.arange(0, epochs_1_data[0].shape[2] - w_length, w_step)

    scores_windows = []
    all_predictions = []
    all_correct = []
    for train_idx_1, test_idx_1 in cv_split:
        train_idx_2 = []
        test_idx_2 = []
        y_train_1, y_test_1 = labels_1[train_idx_1], labels_1[test_idx_1]

        for index, value in enumerate(train_idx_1):
            if y_train_1[index] != 0:
                train_idx_2.append(int((value - 1) / 2))
        for index, value in enumerate(test_idx_1):
            if y_test_1[index] != 0:
                test_idx_2.append(int((value - 1) / 2))
        y_train_2, y_test_2 = labels_2[train_idx_2], labels_2[test_idx_2]

        x_train_csp_1 = []
        x_test_csp_1 = []
        for edt in epochs_1_data_train:
            if len(x_train_csp_1) > 0:
                x_train_csp_1 = np.concatenate((x_train_csp_1, csp_1.fit_transform(edt[train_idx_1], y_train_1)),
                                               axis=1)
                x_test_csp_1 = np.concatenate((x_test_csp_1, csp_1.transform(edt[test_idx_1])), axis=1)
            else:
                x_train_csp_1 = csp_1.fit_transform(edt[train_idx_1], y_train_1)
                x_test_csp_1 = csp_1.transform(edt[test_idx_1])

        x_train_csp_2 = []
        x_test_csp_2 = []
        for edt in epochs_2_data_train:
            if len(x_train_csp_2) > 0:
                x_train_csp_2 = np.concatenate((x_train_csp_2, csp_2.fit_transform(edt[train_idx_2], y_train_2)),
                                               axis=1)
                x_test_csp_2 = np.concatenate((x_test_csp_2, csp_2.transform(edt[test_idx_2])), axis=1)
            else:
                x_train_csp_2 = csp_2.fit_transform(edt[train_idx_2], y_train_2)
                x_test_csp_2 = csp_2.transform(edt[test_idx_2])

        # fit classifier
        classifier_1.fit(x_train_csp_1, y_train_1)
        classifier_2.fit(x_train_csp_2, y_train_2)
        X_test_csp_1 = csp_1.transform(epochs_1_data[0][test_idx_1])
        predictions = classifier_1.predict(X_test_csp_1)
        pred = []
        for index, value in enumerate(test_idx_1):
            if predictions[index] == 0:
                pred.append(0)
            else:

                csp_value = csp_2.transform(epochs_1_data[0][[value]])
                classifier_value = classifier_2.predict(csp_value)
                pred.append(classifier_value[0])

        all_predictions.append(pred)
        all_correct.append(list(map(int, all_labels[test_idx_1])))

        # running classifier: test classifier on sliding window
        score_this_window = []
        for n in w_start:
            test_split_data = []
            for edt in epochs_1_data:
                if len(test_split_data) > 0:
                    test_split_data = np.concatenate(
                        (test_split_data, edt[test_idx_1][:, :, n:(n + w_length)]),
                        axis=1)
                else:
                    test_split_data = edt[test_idx_1][:, :, n:(n + w_length)]
            csp_1_test_res = csp_1.transform(test_split_data)

            step_1_predicions = classifier_1.predict(csp_1_test_res)
            step_predictions = []
            for index, value in enumerate(step_1_predicions):
                if value == 0:
                    step_predictions.append(value)
                else:
                    csp_2_value = csp_2.transform(np.array([test_split_data[index]]))
                    step_2_prediction = classifier_2.predict(csp_2_value)
                    step_predictions.append(step_2_prediction[0])
            score = accuracy_score(all_labels[test_idx_1], step_predictions)
            score_this_window.append(score)

    w_times = (w_start + w_length / 2.) / sfreq + epochs_1[0].tmin
    return w_times, scores_windows, csp_1, epochs_1[0].info, all_predictions, all_correct
