import mne
import numpy as np
from mne import Epochs, pick_types
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit
from sklearn.neural_network import MLPClassifier

from data_classes.subject import Subject

# import tensorflow as tf
#
# from tensorflow.keras import datasets, layers, models


def process(subject, bands, selected_channels, n_splits=10, reg=None, verbose='DEBUG', score_window_flag=False):
    tmin, tmax = 1., 3.

    raw_signals = []
    for i in range(len(bands)):
        raw_signals.append(subject.get_raw_copy())

    if len(selected_channels) > 0:
        for raw_signal in raw_signals:
            for channel in subject.electrode_names:
                if channel not in selected_channels:
                    raw_signal.drop_channels([channel])

    filtered_raw_signals = []
    epochs = []
    epochs_train = []
    epochs_data = []
    epochs_data_train = []

    # Apply band-pass filter
    for index, band in enumerate(bands):
        filtered_raw_signals.append(
            raw_signals[index].filter(band[0], band[1], l_trans_bandwidth=2, h_trans_bandwidth=2, filter_length=1024*2,
                                      fir_design='firwin',
                                      skip_by_annotation='edge', verbose=verbose))

    picks = pick_types(filtered_raw_signals[0].info, meg=False, eeg=True, stim=False, eog=False,
                       exclude='bads')

    for index, band in enumerate(bands):
        epochs.append(
            Epochs(filtered_raw_signals[index], subject.events, subject.id_dict, tmin, tmax, proj=True, picks=picks,
                   baseline=None, preload=True, verbose=verbose))
        epochs_train.append(epochs[index].copy().crop(tmin=tmin, tmax=tmax))

        epochs_data.append(epochs[index].get_data())
        epochs_data_train.append(epochs_train[index].get_data())
    labels = np.array(epochs[0].events[:, -1])


    cv = ShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=42)
    cv_split = cv.split(epochs_data_train[0])

    # Assemble a classifier
    classifier = MLPClassifier(hidden_layer_sizes=(100, 100), random_state=1,
                               max_iter=10000)  # Originally: LinearDiscriminantAnalysis()
    # classifier = LinearDiscriminantAnalysis()

    # model = models.Sequential()
    # model.add(layers.Conv2D(32, (1, 5), activation='relu', input_shape=(10, 1025, 1)))
    # model.add(layers.MaxPooling2D((1, 3)))
    # model.add(layers.Conv2D(64, (1, 5), activation='relu'))
    # model.add(layers.MaxPooling2D((1, 2)))
    # model.add(layers.Conv2D(64, (1, 3), activation='relu'))
    # model.add(layers.Flatten())
    # model.add(layers.Dense(64, activation='relu'))
    # model.add(layers.Dense(2))

    # model.compile(optimizer='adam',
    #               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #               metrics=['accuracy'])







    csp_n_components = 32 if len(selected_channels) == 0 else min(len(selected_channels), 32)
    mne.set_log_level('warning')
    csp = CSP(n_components=csp_n_components, reg=reg, log=None, norm_trace=False, transform_into='csp_space')

    sfreq = raw_signals[0].info['sfreq']
    w_length = int(sfreq)  # running classifier: window length
    w_step = int(sfreq * 0.1)  # running classifier: window step size
    w_start = np.arange(0, epochs_data[0].shape[2] - w_length, w_step)

    scores_windows = []
    all_predictions = []
    all_correct = []

    for train_idx, test_idx in cv_split:
        y_train, y_test = labels[train_idx], labels[test_idx]

        x_train_csp = []
        x_test_csp = []
        for edt in epochs_data_train:
            if len(x_train_csp) > 0:
                x_train_csp = np.concatenate((x_train_csp, csp.fit_transform(edt[train_idx], y_train)),
                                             axis=1)
                x_test_csp = np.concatenate((x_test_csp, csp.transform(edt[test_idx])), axis=1)
            else:
                x_train_csp = csp.fit_transform(edt[train_idx], y_train)
                x_test_csp = csp.transform(edt[test_idx])

        # new_x_train_csp = []
        # for i, e in enumerate(x_train_csp):
        #     new_x_train_csp.append(np.hstack(e))
        # x_train_csp = new_x_train_csp
        #
        # new_x_test_csp = []
        # for i, e in enumerate(x_test_csp):
        #     new_x_test_csp.append(np.hstack(e))
        # x_test_csp = new_x_test_csp
        x_train_shape = x_train_csp.shape
        x_train_csp = x_train_csp.reshape((x_train_shape[0], x_train_shape[1], x_train_shape[2], 1))
        y_train = y_train - 1
        x_test_shape = x_test_csp.shape
        x_test_csp = x_test_csp.reshape((x_test_shape[0], x_test_shape[1], x_test_shape[2], 1))
        y_test = y_test - 1

        # history = model.fit(x_train_csp, y_train, epochs=10,
        #                     validation_data=(x_test_csp, y_test))
        # results_model = model.predict(x_test_csp)
        # all_predictions.append(np.argmax(results_model, axis=1) + 1)


    #     classifier.fit(x_train_csp, y_train)
    #
    #
    #     predictions = classifier.predict(x_test_csp)
    #     all_predictions.append(predictions)
        all_correct.append(y_test + 1)
    #
    #     # running classifier: test classifier on sliding window
    #     if score_window_flag:
    #         score_this_window = []
    #         for n in w_start:
    #             x_test_csp = []
    #             for edt in epochs_data:
    #                 if len(x_test_csp) > 0:
    #                     x_test_csp = np.concatenate(
    #                         (x_test_csp, csp.transform(edt[test_idx][:, :, n:(n + w_length)])),
    #                         axis=1)
    #                 else:
    #                     x_test_csp = csp.transform(edt[test_idx][:, :, n:(n + w_length)])
    #             score_this_window.append(classifier.score(x_test_csp, y_test))
    #         scores_windows.append(score_this_window)
    #
    # w_times = []
    # if score_window_flag:
    #     w_times = (w_start + w_length / 2.) / sfreq + epochs[0].tmin
    # return w_times, scores_windows, csp, epochs[0].info, all_predictions, all_correct, classifier, subject.info
    return [], [], csp, epochs[0].info, all_predictions, all_correct, classifier, subject.info
