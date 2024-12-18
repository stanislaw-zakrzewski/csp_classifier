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

stage = 1


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

    picks = pick_types(filtered_raw_signal.info, meg=False, eeg=True, stim=False, eog=False,
                       exclude='bads')

    selected_events = [e for e in subject.events if e[-1] in selected_event_ids.values()]

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
                              ch_names=channel_names, band=spectral_filter, win_sec=1,
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


def level(subject, selected_channels, selected_event_ids, spectral_filter, t_min, t_max, frequencies,
          all_placement, differences_placement, montage, seed=None, reg=None,
          verbose='CRITICAL'):
    sampling_frequency = subject.sampling_frequency

    raw_signal = subject.get_raw_copy()
    if montage == 'standard_1020':
        raw_signal = raw_signal.drop_channels(['X3'], on_missing='ignore')
        raw_signal = raw_signal.drop_channels(['X5'], on_missing='ignore')

    filtered_raw_signal = bandpass_filter(raw_signal, spectral_filter[0], spectral_filter[1], sampling_frequency)

    filtered_raw_signal.set_montage(montage)

    filtered_raw_signal = preprocessing.compute_current_source_density(filtered_raw_signal)

    # picks = pick_types(filtered_raw_signal.info, meg=False, eeg=True, stim=False, eog=False,
    #                    exclude='bads')

    select_channels(filtered_raw_signal, selected_channels)

    picks = list(range(len(filtered_raw_signal.info['ch_names'])))

    selected_events = [e for e in subject.events if e[-1] in selected_event_ids.values()]

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

    for fold in folds:
        train_idx = fold['train']
        test_idx = fold['test']
        y_train, y_test = labels[train_idx], labels[test_idx]

        x_train = perform_fft(epochs_data_train[train_idx], subject.sampling_frequency, spectral_filter,
                              selected_channels)
        x_test = perform_fft(epochs_data_train[test_idx], subject.sampling_frequency, spectral_filter,
                             selected_channels)

        x_train_a = process_x(x_train, all_placement)
        x_train_s = process_x(x_train, differences_placement)
        x_test_a = process_x(x_test, all_placement)
        x_test_s = process_x(x_test, differences_placement)

        knn_a = KNeighborsClassifier(n_neighbors=10)
        knn_s = KNeighborsClassifier(n_neighbors=10)
        knn_a.fit(x_train_a, y_train)
        knn_s.fit(x_train_s, y_train)

        rf_a = RandomForestClassifier(max_depth=7, random_state=0)
        rf_s = RandomForestClassifier(max_depth=7, random_state=0)
        rf_a.fit(x_train_a, y_train)
        rf_s.fit(x_train_s, y_train)

        predictions_a_knn = knn_a.predict(x_test_a)
        predictions_s_knn = knn_s.predict(x_test_s)
        predictions_a_rf = rf_a.predict(x_test_a)
        predictions_s_rf = rf_s.predict(x_test_s)

        all_predictions_a_knn.append(predictions_a_knn)
        all_predictions_s_knn.append(predictions_s_knn)
        all_predictions_a_rf.append(predictions_a_rf)
        all_predictions_s_rf.append(predictions_s_rf)
        all_correct.append(y_test)

    return all_correct, all_predictions_a_knn, all_predictions_s_knn, all_predictions_a_rf, all_predictions_s_rf


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
    y_true, y_all_knn, y_s_knn, y_a_rf, y_s_rf = level(subject, selected_channels, selected_event_ids,
                                                       [l_freq, h_freq], t_min, t_max, frequencies,
                                                       differences_all_placement, differences_significant_placement, montage, seed)
    # y_pred_knn_s, y_true_significant_s, y_pred_random_forest_s = level(subject, selected_channels, selected_event_ids,
    #                                                                    [l_freq, h_freq], t_min, t_max, frequencies,
    #                                                                    differences_significant_placement, seed)
    y_pred_baseline, y_true_baseline = csp_lda(subject, selected_channels, selected_event_ids, [l_freq, h_freq], t_min,
                                               t_max, seed)

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

    return accuracy_baseline / denominator_baseline, accuracy_knn / denominator_knn, accuracy_random_forest / denominator_random_forest, accuracy_knn_s / denominator_knn_s, accuracy_random_forest_s / denominator_random_forest_s, percentage_differences_all, percentage_differences_significant


accuracies = {'subject': [], 'method': [], 'accuracy': [], 'percentage_of_data_used': []}
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
        acc_b, acc_knn, acc_rf, acc_knn_s, acc_rf_s, p_a, p_s = test_subject(
            '{}/{}.edf'.format(folder_path, parsed_index), heatmap_path, seed=42)
        accuracies['subject'].append('{}'.format(parsed_index))
        accuracies['method'].append('baseline')
        accuracies['accuracy'].append(acc_b)
        accuracies['percentage_of_data_used'].append(100)
        accuracies['subject'].append('{}'.format(parsed_index))
        accuracies['method'].append('knn')
        accuracies['accuracy'].append(acc_knn)
        accuracies['percentage_of_data_used'].append(p_a)
        accuracies['subject'].append('{}'.format(parsed_index))
        accuracies['method'].append('random forest')
        accuracies['accuracy'].append(acc_rf)
        accuracies['percentage_of_data_used'].append(p_a)
        accuracies['subject'].append('{}'.format(parsed_index))
        accuracies['method'].append('knn significant')
        accuracies['accuracy'].append(acc_knn_s)
        accuracies['percentage_of_data_used'].append(p_s)
        accuracies['subject'].append('{}'.format(parsed_index))
        accuracies['method'].append('random forest significant')
        accuracies['accuracy'].append(acc_rf_s)
        accuracies['percentage_of_data_used'].append(p_s)
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
