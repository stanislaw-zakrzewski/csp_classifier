import mne
import numpy as np
from mne import Epochs, pick_types
from mne.decoding import CSP
from scipy.fft import rfft
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit
from sklearn.neural_network import MLPClassifier

from data_classes.subject import Subject

import tensorflow as tf

from tensorflow.keras import datasets, layers, models, activations
from tf_keras_vis.activation_maximization import ActivationMaximization
import matplotlib.pyplot as plt
from scipy import signal
import seaborn as sns


def fft_transform_signle(single_signal):
    # # res = np.abs(rfft(single_signal[0:512])[0:50])
    # # res = np.concatenate((res, np.abs(rfft(single_signal[256:768])[0:50])))
    # # res = np.concatenate((res, np.abs(rfft(single_signal[512:])[0:50])))\

    # res = np.abs(rfft(single_signal[0:512])[0:50])
    # res = np.concatenate((res, np.abs(rfft(single_signal[128:640])[0:50])))
    # res = np.concatenate((res, np.abs(rfft(single_signal[256:768])[0:50])))
    # res = np.concatenate((res, np.abs(rfft(single_signal[384:896])[0:50])))
    # res = np.concatenate((res, np.abs(rfft(single_signal[512:])[0:50])))
    #
    # return res
    # return np.abs(rfft(single_signal)[0:150])
    fs=512
    frequencies = []
    for frequency in range(2,30):
        fl = frequency - .5
        fu = frequency + .5
        transwidth = .2
        wp = [fl, fu]
        ws = [fl - transwidth, fu + transwidth]
        gpass = 99
        gstop = 100
        n, wn = signal.cheb2ord(wp, ws, gpass, gstop, fs=fs)
        sos3 = signal.cheby2(n, 30, wn, 'bandpass', output='sos', fs=fs)
        filtered = signal.sosfilt(sos3, single_signal)
        frequencies.append(filtered)
        # sns.lineplot(single_signal, legend='brief', label='original')
        # sns.lineplot(filtered, legend='brief', label='{} frequency'.format(frequency))
        # plt.show()



    # SOS3 is best!
    # cycles = 14  # how many sine cycles
    # resolution = 1025  # how many datapoints to generate
    # length = np.pi * 2 * cycles
    # my_wave = np.sin(np.arange(0, length, length / resolution))
    #
    #
    # fs=512
    # fl = 6.5
    # fu = 7.5
    # transwidth = .2
    # wp = [fl, fu]
    # ws = [fl - transwidth, fu + transwidth]
    # gpass = 99
    # gstop = 100
    # n, wn = signal.buttord(wp, ws, gpass, gstop, fs=fs)
    # sos1 = signal.butter(n, wn, 'bandpass', output='sos', fs=fs)
    # n, wn = signal.cheb1ord(wp, ws, gpass, gstop, fs=fs)
    # sos2 = signal.cheby1(n, 3, wn, 'bandpass', output='sos', fs=fs)
    # n, wn = signal.cheb2ord(wp, ws, gpass, gstop, fs=fs)
    # sos3 = signal.cheby2(n, 30, wn, 'bandpass', output='sos', fs=fs)
    # n, wn = signal.ellipord(wp, ws, gpass, gstop, fs=fs)
    # sos4 = signal.ellip(n, 3, 30, wn, 'bandpass', output='sos', fs=fs)
    # sns.lineplot(my_wave, legend='brief', label='original')
    # sns.lineplot(signal.sosfilt(sos1, my_wave), legend='brief', label='sos1')
    # sns.lineplot(signal.sosfilt(sos2, my_wave), legend='brief', label='sos2')
    # sns.lineplot(signal.sosfilt(sos3, my_wave), legend='brief', label='sos3')
    # sns.lineplot(signal.sosfilt(sos4, my_wave), legend='brief', label='sos4')
    return frequencies
    # return np.abs(rfft(single_signal[256:768])[4:54])


def fft_transform(signal):
    return np.apply_along_axis(fft_transform_signle, 2, np.array(signal))


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
            raw_signals[index].filter(band[0], band[1], l_trans_bandwidth=2, h_trans_bandwidth=2,
                                      filter_length=1024 * 2,
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
    freq = np.fft.fftfreq(512, d=1. / 512)
    print('FREQUENCIES')
    print(freq[4:54])

    for train_idx, test_idx in cv_split:
        y_train, y_test = labels[train_idx], labels[test_idx]

        x_train_csp = []
        x_test_csp = []
        for edt in epochs_data_train:
            if len(x_train_csp) > 0:
                x_train_csp = np.concatenate((x_train_csp, fft_transform(edt[train_idx])),
                                             axis=1)
                x_test_csp = np.concatenate((x_test_csp, fft_transform(edt[test_idx])), axis=1)
            else:
                x_train_csp = fft_transform(edt[train_idx])
                x_test_csp = fft_transform(edt[test_idx])

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
        x_train_csp = x_train_csp.reshape((x_train_shape[0], x_train_shape[1], x_train_shape[2], x_train_shape[3], 1))
        y_train = y_train - 1
        x_test_shape = x_test_csp.shape
        x_test_csp = x_test_csp.reshape((x_test_shape[0], x_test_shape[1], x_test_shape[2], x_train_shape[3], 1))
        y_test = y_test - 1

        model = models.Sequential()
        model.add(layers.Conv3D(32, (1, 1, 16), activation='relu', input_shape=(x_train_shape[1], x_train_shape[2], x_train_shape[3], 1)))
        model.add(layers.MaxPooling3D((1, 1, 3)))
        model.add(layers.Conv3D(64, (1, 2, 8), activation='relu'))
        model.add(layers.MaxPooling3D((1, 1, 2)))
        model.add(layers.Conv3D(64, (1, 1, 4), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.Dense(2))

        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        history = model.fit(x_train_csp, y_train, epochs=100,
                            validation_data=(x_test_csp, y_test))
        results_model = model.predict(x_test_csp)
        all_predictions.append(np.argmax(results_model, axis=1) + 1)

        #     classifier.fit(x_train_csp, y_train)
        #
        #
        #     predictions = classifier.predict(x_test_csp)
        #     all_predictions.append(predictions)
        all_correct.append(y_test + 1)
        #     start of activation maximization
        # def model_modifier(m):
        #     m.layers[-1].activation = tf.keras.activations.linear
        #
        # visualize_activation = ActivationMaximization(model, model_modifier)
        # seed_input = tf.random.uniform((2, 2, 50, 1), 0, 1)
        #
        # def loss(output):
        #     return (output[0, 0], output[1, 1])
        #
        # activations = visualize_activation(loss, seed_input=seed_input, steps=512, input_range=(0, 1))
        # images = [activation.numpy().astype(np.float32) for activation in activations]
        #
        # classes = {
        #     0: 'left hand',
        #     1: 'right hand',
        # }
        #
        # # Visualize each image
        # f, axarr = plt.subplots(2, 1)
        # axarr[0].imshow(images[0])
        # axarr[1].imshow(images[1])
        # # for i in range(0, len(images)):
        # #     visualization = images[i]
        # #     plt.imshow(visualization, cmap='gray')
        # #     plt.title(f'CIFAR10 target = {classes[i]}')
        # plt.show()
    #     end of activation maximization


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
