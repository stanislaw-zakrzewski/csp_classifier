from yasa import irasa

from data_classes.subject import Subject
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from src.feature_extractors.CSP import CSP
from src.classifiers.LDA import LDA
from src.preprocessing.bandpass_filter import bandpass_filter
from src.preprocessing.channel_selector import select_channels
from mne import Epochs, pick_types, preprocessing
from filter_edf import main as create_tmp
from tkinter import filedialog as fd
import os
import traceback
from sklearn.cluster import KMeans
from scipy.fft import rfft, rfftfreq
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from time import time

# EEGNet-specific imports
from EEGModels import EEGNet
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
from skopt.space import Integer, Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize, dump, load

stage = 1

RANDOM_SEED = 23

FREQUENCY = 256  # Hz
FREQUENCY_EEGNET = 128  # Hz
LOWPASS_CUTOFF = 2  # Hz
HIGHPASS_CUTOFF = None  # Hz
EPOCHS = 30
BATCH_SIZE = 20
F1 = 8
D = 2
F2 = 16


def get_folds(labels, with_validation=False, seed=None):
    if seed is not None:
        np.random.seed(seed)

    trial_order = list(range(len(labels)))
    np.random.shuffle(trial_order)

    fifths = np.array_split(trial_order, 5)

    fold_splits = [
        [0, 1, 2, 3, 4],
        [1, 2, 3, 4, 0],
        [2, 3, 4, 0, 1],
        [3, 4, 0, 1, 2],
        [4, 0, 1, 2, 3],
    ]

    folds = []
    for fold_split in fold_splits:
        test_trials = fifths[fold_split[4]]
        if with_validation:
            train_trials = np.concatenate([fifths[fold_split[0]], fifths[fold_split[1]], fifths[fold_split[2]]])
            validate_trials = fifths[fold_split[3]]
            folds.append({
                'train': train_trials,
                'validate': validate_trials,
                'test': test_trials
            })
        else:
            train_trials = np.concatenate(
                [fifths[fold_split[0]], fifths[fold_split[1]], fifths[fold_split[2]], fifths[fold_split[3]]])
            folds.append({
                'train': train_trials,
                'test': test_trials
            })

    return folds


def csp_lda(subject, selected_channels, selected_event_ids, spectral_filter, t_min, t_max, seed=None, reg=None,
            verbose='CRITICAL'):
    sampling_frequency = subject.sampling_frequency

    raw_signal = subject.get_raw_copy()

    select_channels(raw_signal, selected_channels)

    filtered_raw_signal = bandpass_filter(raw_signal, spectral_filter[0], spectral_filter[1], sampling_frequency)

    events = subject.events
    if FREQUENCY > 0:
        filtered_raw_signal = filtered_raw_signal.resample(sfreq=FREQUENCY)  # resample
        s_freq = FREQUENCY
        events[:, 0] = np.round(events[:, 0] * (FREQUENCY / float(subject.sampling_frequency))).astype(
            int)

    picks = pick_types(filtered_raw_signal.info, meg=False, eeg=True, stim=False, eog=False,
                       exclude='bads')

    selected_events = [e for e in events if e[-1] in selected_event_ids.values()]

    epochs = Epochs(filtered_raw_signal, selected_events, selected_event_ids, t_min, t_max, proj=True, picks=picks,
                    baseline=None, preload=True, verbose=verbose)
    epochs_train = epochs.copy()  # .crop(tmin=tmin, tmax=tmax))

    epochs_data_train = epochs_train.get_data(copy=False)
    labels = np.array(epochs.events[:, -1])

    all_predictions = []
    all_correct = []

    folds = get_folds(labels, seed)

    for fold in folds:
        train_idx = fold['train']
        test_idx = fold['test']
        classifier = LDA()
        csp = CSP(selected_channels=selected_channels, reg=reg)
        y_train, y_test = labels[train_idx], labels[test_idx]

        x_train_csp = csp.fit_transform(epochs_data_train[train_idx], y_train)
        x_test_csp = csp.transform(epochs_data_train[test_idx])

        classifier.fit(x_train_csp, y_train)

        predictions = classifier.predict(x_test_csp)
        all_predictions.append(predictions)
        all_correct.append(y_test)

    return all_predictions, all_correct


def process_x(x, placements):
    new_x_vals = []
    for x_idx, x_val in enumerate(x):
        new_x_val = []
        for placement_key in placements:
            placement = placements[placement_key]
            tmp_val = x_val[placement_key, placement]
            new_x_val = np.concatenate((new_x_val, tmp_val))
        new_x_vals.append(new_x_val)
    return np.array(new_x_vals)


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def perform_fft(x, sampling_frequency, spectral_filter, channel_names):
    osc_epochs_data = []
    for x_val in x:
        _, _, psd_osc = irasa(x_val, sampling_frequency,
                              ch_names=channel_names, band=spectral_filter, win_sec=.5,
                              return_fit=False)
        osc_epochs_data.append(psd_osc)
    return np.array(osc_epochs_data)
    # x_fft = rfft(x)
    # xf_frequencies = rfftfreq(x.shape[-1], 1 / sampling_frequency)
    #
    # lim_min = find_nearest(xf_frequencies, spectral_filter[0])
    # lim_max = find_nearest(xf_frequencies, spectral_filter[1]) + 1
    # decomposition_frequencies = xf_frequencies[lim_min:lim_max]
    #
    # return np.abs(x_fft[:, :, lim_min:lim_max])

def filter_significant(filtered_raw_signal, differences_placement, selected_channels):
    channels_to_drop = []
    # all_placements = b_placements
    # for ap in a_placements:
    #     if ap in all_placements:
    #         all_placements = np.concatenate(all_placements[ap], a_placements[ap])
    #     else:
    #         all_placements[ap] = a_placements[ap]
    for sc_id, sc in enumerate(selected_channels):
        if sc_id not in differences_placement:
            channels_to_drop.append(sc)
    filtered_raw_signal = filtered_raw_signal.drop_channels(channels_to_drop)
    # for ap in all_placements:
    #     av = all_placements[ap]
    #     filtered_raw_signal = filtered_raw_signal.filter(min(av)-1, max(av)+1, picks=[selected_channels[ap]])

    return filtered_raw_signal


def eeg_net(subject, selected_channels, selected_event_ids, spectral_filter, t_min, t_max, frequencies, differences_placement, seed=None, significant=False,
          verbose='CRITICAL'):
    sampling_frequency = subject.sampling_frequency

    raw_signal = subject.get_raw_copy()
    original_sfreq = raw_signal.info['sfreq']
    raw_signal.resample(sfreq=FREQUENCY_EEGNET)  # resample

    events = subject.events
    events[:, 0] = np.round(events[:, 0] * (FREQUENCY_EEGNET / float(original_sfreq))).astype(
        int)  # adjust event placement to the resampling

    select_channels(raw_signal, selected_channels)

    filtered_raw_signal = bandpass_filter(raw_signal, spectral_filter[0], spectral_filter[1], FREQUENCY_EEGNET)
    # filtered_raw_signal.resample(sfreq=FREQUENCY_EEGNET)
    if significant:
        # F1 = 6
        # D = 2
        # F2 = 7
        filtered_raw_signal = filter_significant(filtered_raw_signal, differences_placement, selected_channels)
    else:
        # F1 = 4
        # D = 2
        # F2 = 16
        print('a')
    picks = pick_types(filtered_raw_signal.info, meg=False, eeg=True, stim=False, eog=False,
                       exclude='bads')

    selected_events = [e for e in events if e[-1] in selected_event_ids.values()]

    epochs = Epochs(filtered_raw_signal, selected_events, selected_event_ids, t_min, t_max, proj=True, picks=picks,
                    baseline=None, preload=True, verbose=verbose)
    epochs_train = epochs.copy()

    epochs_data_train = epochs_train.get_data(copy=False)
    labels = np.array(epochs.events[:, -1])

    all_predictions = []
    all_correct = []

    folds = get_folds(labels, seed)

    acc_combined = []

    # space = [Integer(2, 8, name='F1'), Integer(2, 16, name='F2'), Categorical([20,40,60,80,100], name='EPOCHS')]

    # @use_named_args(space)
    # def objective(**params):
    acc_combined = []
    for fold in folds:
        # F1 = params['F1']
        # F2 = params['F2']
        # EPOCHS = params['EPOCHS']
        train_idx = fold['train']
        validate_idx = fold['validate']
        test_idx = fold['test']
        # classifier = LDA()
        # csp = CSP(selected_channels=selected_channels, reg=reg)
        y_train, y_validate, y_test = labels[train_idx], labels[validate_idx], labels[test_idx]
        #
        x_train = epochs_data_train[train_idx]
        x_validate = epochs_data_train[validate_idx]
        x_test = epochs_data_train[test_idx]
        #
        # classifier.fit(x_train_csp, y_train)
        #
        # predictions = classifier.predict(x_test_csp)
        # all_predictions.append(predictions)
        # all_correct.append(y_test)

        # convert labels to one-hot encodings.
        Y_train = np_utils.to_categorical(y_train - 1)
        Y_validate = np_utils.to_categorical(y_validate - 1)
        Y_test = np_utils.to_categorical(y_test - 1)

        kernels, chans, samples = 1, x_train.shape[1], x_train.shape[2]
        # convert data to NHWC (trials, channels, samples, kernels) format. Data
        # contains 60 channels and 151 time-points. Set the number of kernels to 1.
        X_train = x_train.reshape(x_train.shape[0], chans, samples, kernels)
        X_validate = x_validate.reshape(x_validate.shape[0], chans, samples, kernels)
        X_test = x_test.reshape(x_test.shape[0], chans, samples, kernels)

        # configure the EEGNet-8,2,16 model with kernel length of 32 samples (other
        # model configurations may do better, but this is a good starting point)
        model = EEGNet(nb_classes=2, Chans=chans, Samples=samples,
                       dropoutRate=0.5, kernLength=128, F1=F1, D=D, F2=F2,
                       dropoutType='Dropout')

        # compile the model and set the optimizers
        model.compile(loss='categorical_crossentropy', optimizer='adam',
                      metrics=['accuracy'])

        # set a valid path for your system to record model checkpoints
        # checkpointer = ModelCheckpoint(filepath='/tmp/checkpoint.h5', verbose=1,
        #                                save_best_only=True)

        ###############################################################################
        # if the classification task was imbalanced (significantly more trials in one
        # class versus the others) you can assign a weight to each class during
        # optimization to balance it out. This data is approximately balanced so we
        # don't need to do this, but is shown here for illustration/completeness.
        ###############################################################################

        # the syntax is {class_1:weight_1, class_2:weight_2,...}. Here just setting
        # the weights all to be 1
        class_weights = {0: 1, 1: 1, 2: 1, 3: 1}

        ################################################################################
        # fit the model. Due to very small sample sizes this can get
        # pretty noisy run-to-run, but most runs should be comparable to xDAWN +
        # Riemannian geometry classification (below)
        ################################################################################
        hist = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=EPOCHS,
                         verbose=2, validation_data=(X_validate, Y_validate),
                         # callbacks=[checkpointer],
                         class_weight=class_weights)
        sns.lineplot(x=hist.epoch, y=hist.history['accuracy'], label='Train Accuracy')
        sns.lineplot(x=hist.epoch, y=hist.history['val_accuracy'], label="Validation Accuracy")
        plt.show()

        # load optimal weights
        # model.load_weights('/tmp/checkpoint.h5')

        probs = model.predict(X_test)

        preds = probs.argmax(axis=-1)
        acc = np.mean(preds == Y_test.argmax(axis=-1))

        # cm = metrics.confusion_matrix(Y_test.argmax(axis=-1), preds)
        # return acc, cm
        acc_combined.append(acc)
    # return 1-np.average(acc_combined)

    # res_gp = gp_minimize(objective, space, n_calls=40, random_state=0)
    # data_to_save = {
    #     'x': res_gp['x'],
    #     'x_iters': res_gp['x_iters'],
    #     'func_vals': res_gp['func_vals']
    # }
    # if significant:
    #     np.save( 'significant_optimizer.npy', data_to_save, allow_pickle=True)
    # else:
    #     np.save( 'all_optimizer.npy', data_to_save, allow_pickle=True)
    # return 1
    return np.average(acc_combined)


def level(subject, selected_channels, selected_event_ids, spectral_filter, t_min, t_max, frequencies,
          all_placement, differences_placement, montage, seed=None, reg=None,
          verbose='CRITICAL'):
    start = time()
    s_freq = subject.sampling_frequency
    sampling_frequency = subject.sampling_frequency

    raw_signal = subject.get_raw_copy()
    if montage == 'standard_1020':
        raw_signal = raw_signal.drop_channels(['X3'], on_missing='ignore')
        raw_signal = raw_signal.drop_channels(['X5'], on_missing='ignore')

    filtered_raw_signal = bandpass_filter(raw_signal, spectral_filter[0], spectral_filter[1], sampling_frequency)

    events = subject.events
    if FREQUENCY > 0:
        filtered_raw_signal = filtered_raw_signal.resample(sfreq=FREQUENCY)  # resample
        events[:, 0] = np.round(events[:, 0] * (FREQUENCY / float(s_freq))).astype(
            int)
        s_freq = FREQUENCY

    filtered_raw_signal.set_montage(montage)

    filtered_raw_signal = preprocessing.compute_current_source_density(filtered_raw_signal)

    # picks = pick_types(filtered_raw_signal.info, meg=False, eeg=True, stim=False, eog=False,
    #                    exclude='bads')

    select_channels(filtered_raw_signal, selected_channels)

    picks = list(range(len(filtered_raw_signal.info['ch_names'])))

    selected_events = [e for e in events if e[-1] in selected_event_ids.values()]

    epochs = Epochs(filtered_raw_signal, selected_events, selected_event_ids, t_min, t_max, proj=True, picks=picks,
                    baseline=None, preload=True, verbose=verbose)
    epochs_train = epochs.copy()  # .crop(tmin=tmin, tmax=tmax))

    epochs_data_train = epochs_train.get_data(copy=False)
    labels = np.array(epochs.events[:, -1])

    all_predictions_a_knn = []
    all_predictions_s_knn = []
    all_predictions_a_rf = []
    all_predictions_s_rf = []
    all_correct = []

    folds = get_folds(labels, seed)
    end = time()
    time_knn = end - start
    time_knn_s = end - start
    time_rf = end - start
    time_rf_s = end - start

    for fold in folds:
        start = time()

        train_idx = fold['train']
        test_idx = fold['test']
        y_train, y_test = labels[train_idx], labels[test_idx]
        all_correct.append(y_test)

        x_train = perform_fft(epochs_data_train[train_idx], s_freq, spectral_filter,
                              selected_channels)
        x_test = perform_fft(epochs_data_train[test_idx], s_freq, spectral_filter,
                             selected_channels)

        x_train_a = process_x(x_train, all_placement)
        x_train_s = process_x(x_train, differences_placement)
        x_test_a = process_x(x_test, all_placement)
        x_test_s = process_x(x_test, differences_placement)
        end = time()
        time_knn += end - start
        time_knn_s += end - start
        time_rf += end - start
        time_rf_s += end - start

        start = time()
        knn_a = KNeighborsClassifier(n_neighbors=10)
        knn_a.fit(x_train_a, y_train)
        predictions_a_knn = knn_a.predict(x_test_a)
        all_predictions_a_knn.append(predictions_a_knn)
        time_knn += time() - start

        start = time()
        knn_s = KNeighborsClassifier(n_neighbors=10)
        knn_s.fit(x_train_s, y_train)
        predictions_s_knn = knn_s.predict(x_test_s)
        all_predictions_s_knn.append(predictions_s_knn)
        time_knn_s += time() - start

        start = time()
        rf_a = RandomForestClassifier(max_depth=7, random_state=0)
        rf_a.fit(x_train_a, y_train)
        predictions_a_rf = rf_a.predict(x_test_a)
        all_predictions_a_rf.append(predictions_a_rf)
        time_rf += time() - start

        start = time()
        rf_s = RandomForestClassifier(max_depth=7, random_state=0)
        rf_s.fit(x_train_s, y_train)
        predictions_s_rf = rf_s.predict(x_test_s)
        all_predictions_s_rf.append(predictions_s_rf)
        time_rf_s += time() - start


    return all_correct, all_predictions_a_knn, all_predictions_s_knn, all_predictions_a_rf, all_predictions_s_rf, time_knn, time_knn_s, time_rf, time_rf_s


def get_significant_placement(heatmap_data, cluster_count=1):
    if cluster_count == 1:
        res = {}
        for channel_idx, channel_data in enumerate(heatmap_data):
            res[channel_idx] = np.array(list(range(len(channel_data))))
        return res, 100

    kmeans = KMeans(n_clusters=cluster_count, n_init='auto')
    kmeans.fit(heatmap_data.flatten().reshape(-1, 1))

    significant_cluster = 0
    for cluster_id, cluster_center in enumerate(kmeans.cluster_centers_):
        if kmeans.cluster_centers_[significant_cluster] < cluster_center:
            significant_cluster = cluster_id

    res = {}
    all_count = 0
    selected_count = 0
    for channel_idx, channel_data in enumerate(heatmap_data):
        all_count += len(channel_data)
        preds = kmeans.predict(channel_data.reshape(-1, 1))
        locs = np.where(preds == significant_cluster)[0]
        if len(locs) > 0:
            selected_count += len(locs)
            res[channel_idx] = locs
    return res, selected_count / all_count * 100


def test_subject(subject_path, subject_heatmap_path, seed=None):
    print('Processing:', subject_path)
    subject = Subject(subject_path)
    heatmap = np.load(subject_heatmap_path, allow_pickle=True).item()
    differences = np.abs(heatmap['heatmap_a'] - heatmap['heatmap_b'])
    differences_all_placement, percentage_differences_all = get_significant_placement(differences)
    differences_significant_placement, percentage_differences_significant = get_significant_placement(differences,
                                                                                                      cluster_count=2)

    # a_all_placements, percentage_a = get_significant_placement(heatmap['heatmap_a'])
    # b_all_placements, percentage_b = get_significant_placement(heatmap['heatmap_b'])
    # a_significant_placements, percentage_a_significant = get_significant_placement(heatmap['heatmap_a'], 2)
    # b_significant_placements, percentage_b_significant = get_significant_placement(heatmap['heatmap_b'], 2)
    # significant_frequencies_count, frequencies_count, significant_channels_count, channels_count = create_tmp(heatmap,
    #                                                                                                         subject)
    print('Percentage significant:', percentage_differences_significant)
    # rest_frequencies_count, _, rest_channels_count, _ = create_tmp(heatmap, subject, inverse=True)
    # remaining_data_significant = significant_frequencies_count / frequencies_count
    # remaining_data_significant = "%0.2f" % (remaining_data_significant * 100)
    # remaining_data_rest = rest_frequencies_count / frequencies_count
    # remaining_data_rest = "%0.2f" % (remaining_data_rest * 100)
    # subject_filtered = Subject('filtered.edf')
    # subject_filtered_inverse = Subject('filtered_inverse.edf')
    all_channels = subject.electrode_names
    selected_channels = heatmap['channels']
    a_label_name = heatmap['a_label_name']
    b_label_name = heatmap['b_label_name']
    t_min = heatmap['t_min']
    t_max = heatmap['t_max']
    montage = heatmap['montage']
    selected_event_ids = {
        a_label_name: subject.id_dict[a_label_name],
        b_label_name: subject.id_dict[b_label_name]
    }
    frequencies = heatmap['frequencies']
    l_freq = frequencies[0]
    h_freq = frequencies[-1]
    # start = time()
    # accuracy_eegnet = eeg_net(subject, selected_channels, selected_event_ids,
    #                                                [l_freq, h_freq], t_min, t_max, frequencies, differences_all_placement, seed)
    # time_eegnet = time() - start
    # start = time()
    # accuracy_eegnet_significant = eeg_net(subject, selected_channels, selected_event_ids,
    #                                                [l_freq, h_freq], t_min, t_max, frequencies, differences_significant_placement, seed, True)
    # time_eegnet_s = time() - start
    y_true, y_all_knn, y_s_knn, y_a_rf, y_s_rf, time_knn, time_knn_s, time_rf, time_rf_s = level(subject, selected_channels, selected_event_ids,
                                                       [l_freq, h_freq], t_min, t_max, frequencies,
                                                       differences_all_placement, differences_significant_placement, montage, seed)
    # y_pred_knn_s, y_true_significant_s, y_pred_random_forest_s = level(subject, selected_channels, selected_event_ids,
    #                                                                    [l_freq, h_freq], t_min, t_max, frequencies,
    #                                                                    differences_significant_placement, seed)

    start = time()
    y_pred_baseline, y_true_baseline = csp_lda(subject, selected_channels, selected_event_ids, [l_freq, h_freq], t_min,
                                               t_max, seed)
    time_baseline = time() - start

    # _, y_pred_rest, y_true_rest, _, _ = csp_lda(subject_filtered_inverse, selected_channels, selected_event_ids,
    #                                             [l_freq, h_freq], t_min, t_max, seed)


    def get_accuracy_data(y_pred, y_true):
        denominator = 0
        nominator = 0
        for fold_id in range(len(y_pred)):
            fold_data = y_pred[fold_id]
            for trial_id in range(len(fold_data)):
                denominator += 1
                if fold_data[trial_id] == y_true[fold_id][trial_id]:
                    nominator += 1
        return nominator, denominator

    accuracy_baseline, denominator_baseline = get_accuracy_data(y_pred_baseline, y_true_baseline)
    accuracy_knn, denominator_knn = get_accuracy_data(y_all_knn, y_true)
    accuracy_random_forest, denominator_random_forest = get_accuracy_data(y_a_rf, y_true)
    accuracy_knn_s, denominator_knn_s = get_accuracy_data(y_s_knn, y_true)
    accuracy_random_forest_s, denominator_random_forest_s = get_accuracy_data(y_s_rf, y_true)
    accuracy_eegnet, accuracy_eegnet_significant = 1,1
    time_eegnet, time_eegnet_s = 1,1
    return accuracy_baseline / denominator_baseline, accuracy_knn / denominator_knn, accuracy_random_forest / denominator_random_forest, accuracy_knn_s / denominator_knn_s, accuracy_random_forest_s / denominator_random_forest_s, accuracy_eegnet, accuracy_eegnet_significant, percentage_differences_all, percentage_differences_significant, time_baseline, time_knn, time_knn_s, time_rf, time_rf_s, time_eegnet, time_eegnet_s


accuracies = {'subject': [], 'method': [], 'accuracy': [], 'percentage_of_data_used': [], 'time': []}
folder_path = fd.askdirectory()
# folder_path = 'dataset_temp'
for file in os.listdir(folder_path):
    file_name = os.fsdecode(file)
    file_name = file_name[:-4]
    parsed_index = file_name

    try:
        if stage == 1:
            heatmap_path = 'parafac_analysis/significant_heatmaps/{}.npy'.format(parsed_index)
        else:
            heatmap_path = 'parafac_analysis/significant_heatmaps_stage_2/{}.npy'.format(parsed_index)
        acc_b, acc_knn, acc_rf, acc_knn_s, acc_rf_s, acc_en, acc_en_s, p_a, p_s, t_knn, t_knn_s, t_rf, t_rf_s, t_b, t_en, t_en_s = test_subject(
            '{}/{}.edf'.format(folder_path, parsed_index), heatmap_path, seed=42)
        accuracies['subject'].append('{}'.format(parsed_index))
        accuracies['method'].append('baseline')
        accuracies['accuracy'].append(acc_b)
        accuracies['percentage_of_data_used'].append(p_a)
        accuracies['time'].append(t_b)

        accuracies['subject'].append('{}'.format(parsed_index))
        accuracies['method'].append('knn')
        accuracies['accuracy'].append(acc_knn)
        accuracies['percentage_of_data_used'].append(p_a)
        accuracies['time'].append(t_knn)

        accuracies['subject'].append('{}'.format(parsed_index))
        accuracies['method'].append('random forest')
        accuracies['accuracy'].append(acc_rf)
        accuracies['percentage_of_data_used'].append(p_a)
        accuracies['time'].append(t_rf)

        accuracies['subject'].append('{}'.format(parsed_index))
        accuracies['method'].append('knn significant')
        accuracies['accuracy'].append(acc_knn_s)
        accuracies['percentage_of_data_used'].append(p_s)
        accuracies['time'].append(t_knn_s)

        accuracies['subject'].append('{}'.format(parsed_index))
        accuracies['method'].append('random forest significant')
        accuracies['accuracy'].append(acc_rf_s)
        accuracies['percentage_of_data_used'].append(p_s)
        accuracies['time'].append(t_rf_s)

        accuracies['subject'].append('{}'.format(parsed_index))
        accuracies['method'].append('EEGNet')
        accuracies['accuracy'].append(acc_en)
        accuracies['percentage_of_data_used'].append(p_a)
        accuracies['time'].append(t_en)

        accuracies['subject'].append('{}'.format(parsed_index))
        accuracies['method'].append('EEGNet significant')
        accuracies['accuracy'].append(acc_en_s)
        accuracies['percentage_of_data_used'].append(p_s)
        accuracies['time'].append(t_en_s)
    except Exception as e:
        print('ERROR with subject {}'.format(parsed_index))
        print(e)
        print(traceback.format_exc())

df = pd.DataFrame(data=accuracies)
df.to_csv('{}_accuracies.csv'.format(folder_path.split('/')[-1]))

sns.set_theme(rc={'figure.figsize': (25, 10)})
ax = sns.barplot(x='subject', y='accuracy', hue='method', data=df)
for i in ax.containers:
    ax.bar_label(i, )
plt.show()
