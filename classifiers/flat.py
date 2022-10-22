import numpy as np
from mne import Epochs, pick_types
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit
from sklearn.pipeline import make_pipeline

from config import eeg_data_path
from data_classes.subject import Subject


def process(subject_name, bands, selected_channels, randomness, reg=None, balanced=True):
    tmin, tmax = 0., 1.
    event_id = dict(movement=0, rest=1)
    subject = Subject(randomness)

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
            raw_signals[index].filter(band[0], band[1], l_trans_bandwidth=.2, h_trans_bandwidth=.2, fir_design='firwin',
                                      skip_by_annotation='edge'))

    picks = pick_types(filtered_raw_signals[0].info, meg=False, eeg=True, stim=False, eog=False,
                       exclude='bads')

    for index, band in enumerate(bands):
        epochs.append(
            Epochs(filtered_raw_signals[index], subject.events, event_id, tmin, tmax, proj=True, picks=picks,
                   baseline=None, preload=True))
        epochs_train.append(epochs[index].copy().crop(tmin=0., tmax=1.))

        epochs_data.append(epochs[index].get_data())
        epochs_data_train.append(epochs_train[index].get_data())
    labels = np.array(epochs[0].events[:, -1])

    cv = ShuffleSplit(10, test_size=0.2, random_state=42)
    cv_split = cv.split(epochs_data_train[0])

    # Assemble a classifier
    # classifier = MLPClassifier(hidden_layer_sizes=(20, 20), random_state=1,
    #                            max_iter=1000)  # Originally: LinearDiscriminantAnalysis()
    classifier = LinearDiscriminantAnalysis()
    csp_n_components = 32 if len(selected_channels) == 0 else min(len(selected_channels), 32)
    csp = CSP(n_components=csp_n_components, reg=reg, log=True, norm_trace=False)
    clf = make_pipeline(
        # UnsupervisedSpatialFilter(PCA(csp_n_components), average=False),
        csp,
        # StandardScaler(),
        # SVC()
    )

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
                x_train_csp = np.concatenate((x_train_csp, clf.fit_transform(edt[train_idx], y_train)), axis=1)
                x_test_csp = np.concatenate((x_test_csp, clf.transform(edt[test_idx])), axis=1)
            else:
                x_train_csp = clf.fit_transform(edt[train_idx], y_train)
                x_test_csp = clf.transform(edt[test_idx])

        # fit classifier
        classifier.fit(x_train_csp, y_train)
        pred_train = classifier.predict(x_train_csp)

        X_test_csp = csp.transform(epochs_data[0][test_idx])
        predictions = classifier.predict(X_test_csp)
        all_predictions.append(predictions)
        all_correct.append(y_test)

        # running classifier: test classifier on sliding window
        score_this_window = []
        for n in w_start:
            x_test_csp = []
            for edt in epochs_data:
                if len(x_test_csp) > 0:
                    x_test_csp = np.concatenate(
                        (x_test_csp, clf.transform(edt[test_idx][:, :, n:(n + w_length)])),
                        axis=1)
                else:
                    x_test_csp = clf.transform(edt[test_idx][:, :, n:(n + w_length)])
            score_this_window.append(classifier.score(x_test_csp, y_test))
        scores_windows.append(score_this_window)

    w_times = (w_start + w_length / 2.) / sfreq + epochs[0].tmin
    return w_times, scores_windows, csp, epochs[0].info, all_predictions, all_correct
