import mne
import numpy as np
from mne import Epochs, pick_types
from mne.decoding import CSP
from scipy.fft import rfft
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit
from sklearn.neural_network import MLPClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from data_classes.subject import Subject


# import tensorflow as tf
#
# from tensorflow.keras import datasets, layers, models


def fft_transform_signle(single_signal):
    # res = np.abs(rfft(single_signal[0:512])[0:50])
    # res = np.concatenate((res, np.abs(rfft(single_signal[256:768])[0:50])))
    # res = np.concatenate((res, np.abs(rfft(single_signal[512:])[0:50])))\

    res = np.abs(rfft(single_signal[0:512])[0:50])
    res = np.concatenate((res, np.abs(rfft(single_signal[128:640])[0:50])))
    res = np.concatenate((res, np.abs(rfft(single_signal[256:768])[0:50])))
    res = np.concatenate((res, np.abs(rfft(single_signal[384:896])[0:50])))
    res = np.concatenate((res, np.abs(rfft(single_signal[512:])[0:50])))

    return res
    # return np.abs(rfft(single_signal)[0:50])


def fft_transform(signal):
    return np.apply_along_axis(fft_transform_signle, 2, np.array(signal))


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(10 * 5 * 50, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    # def __init__(self):
    #     super(Net, self).__init__()
    #     self.fc1 = nn.Linear(10 * 5 * 50, 64)  # 10 channels * 5 timeframes + 50 frequency bands
    #     self.fc2 = nn.Linear(64, 16)
    #     self.fc3 = nn.Linear(16, 2)
    #
    # def forward(self, x):
    #     x = torch.flatten(x, 1)
    #     x = F.relu(self.fc1(x))
    #     x = F.relu(self.fc2(x))
    #     x = self.fc3(x)
    #     return F.sigmoid(x)


def transform_y_for_learning(y, number_of_classes=2):
    new_y = []
    for value_y in y:
        nnn = []
        for i in range(number_of_classes):
            if i == value_y:
                nnn.append(1.0)
            else:
                nnn.append(.0)
        new_y.append(nnn)
    return torch.tensor(new_y, dtype=torch.float32)


def process(subject, bands, selected_channels, n_splits=3, reg=None, verbose='DEBUG', score_window_flag=False):
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

    # model = models.Sequential()
    # model.add(layers.Flatten(input_shape=(10, 250, 1)))
    # model.add(layers.Dense(256, activation='relu'))
    # model.add(layers.Dense(128, activation='relu'))
    # model.add(layers.Dense(64, activation='relu'))
    # model.add(layers.Dense(2))
    #
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
        # x_train_shape = x_train_csp.shape
        # x_train_csp = x_train_csp.reshape((x_train_shape[0], x_train_shape[1], x_train_shape[2], 1))
        y_train = y_train - 1
        # x_test_shape = x_test_csp.shape
        # x_test_csp = x_test_csp.reshape((x_test_shape[0], x_test_shape[1], x_test_shape[2], 1))
        y_test = y_test - 1

        X = torch.tensor(x_train_csp, dtype=torch.float32, requires_grad=True)
        y = torch.tensor(y_train, dtype=torch.float32)
        loss_fn = nn.CrossEntropyLoss()  # binary cross entropy
        net = Net()

        f1 = nn.Flatten()
        l1 = nn.Linear(10*5*50, 256)
        r1 = nn.ReLU()
        l2 = nn.Linear(256, 64)
        r2 = nn.ReLU()
        l3 = nn.Linear(64, 2)
        s3 = nn.Softmax()
        net = nn.Sequential(
            f1,
            l1,
            r1,
            l2,
            r2,
            l3,
            s3,
        )

        print(net)
        optimizer = optim.SGD(net.parameters(), lr=0.001)

        n_epochs = 100
        batch_size = 10

        for epoch in range(n_epochs):
            for i in range(0, len(X), batch_size):
                Xbatch = X[i:i + batch_size]
                y_pred = net(Xbatch)
                ybatch = y[i:i + batch_size]
                expectedy = transform_y_for_learning(ybatch)
                loss = loss_fn(y_pred, expectedy)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f'Finished epoch {epoch}, latest loss {loss}')

        # history = model.fit(x_train_csp, y_train, epochs=10,
        #                     validation_data=(x_test_csp, y_test))
        # results_model = model.predict(x_test_csp)
        # all_predictions.append(np.argmax(results_model, axis=1) + 1)

        #     classifier.fit(x_train_csp, y_train)
        #
        #
        predictions = torch.argmax(net(torch.tensor(x_test_csp, dtype=torch.float32, requires_grad=True)), dim=1)
        all_predictions.append(predictions + 1)
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
