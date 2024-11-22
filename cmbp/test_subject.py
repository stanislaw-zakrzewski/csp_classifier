from data_classes.subject import Subject
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from src.feature_extractors.CSP import CSP
from src.classifiers.LDA import LDA
from src.preprocessing.bandpass_filter import bandpass_filter
from src.preprocessing.channel_selector import select_channels
from mne import Epochs, pick_types
from filter_edf import main as create_tmp
from tkinter import filedialog as fd
import os
import traceback

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

    return csp, all_predictions, all_correct, classifier, subject.info


def test_subject(subject_path, subject_heatmap_path, seed=None):
    print('Processing:', subject_path)
    subject = Subject(subject_path)
    heatmap = np.load(subject_heatmap_path, allow_pickle=True).item()
    significant_frequencies_count, frequencies_count, significant_channels_count, channels_count = create_tmp(heatmap,
                                                                                                              subject)
    rest_frequencies_count, _, rest_channels_count, _ = create_tmp(heatmap, subject, inverse=True)
    remaining_data_significant = significant_frequencies_count / frequencies_count
    remaining_data_significant = "%0.2f" % (remaining_data_significant * 100)
    remaining_data_rest = rest_frequencies_count / frequencies_count
    remaining_data_rest = "%0.2f" % (remaining_data_rest * 100)
    subject_filtered = Subject('filtered.edf')
    subject_filtered_inverse = Subject('filtered_inverse.edf')
    all_channels = subject.electrode_names
    selected_channels = heatmap['channels']
    a_label_name = heatmap['a_label_name']
    b_label_name = heatmap['b_label_name']
    t_min = heatmap['t_min']
    t_max = heatmap['t_max']
    selected_event_ids = {
        a_label_name: subject.id_dict[a_label_name],
        b_label_name: subject.id_dict[b_label_name]
    }
    l_freq = heatmap['frequencies'][0]
    h_freq = heatmap['frequencies'][-1]

    _, y_pred_baseline, y_true_baseline, _, _ = csp_lda(subject, selected_channels, selected_event_ids,
                                                        [l_freq, h_freq], t_min, t_max, seed)
    _, y_pred_significant, y_true_significant, _, _ = csp_lda(subject_filtered, selected_channels, selected_event_ids,
                                                              [l_freq, h_freq], t_min, t_max, seed)
    _, y_pred_rest, y_true_rest, _, _ = csp_lda(subject_filtered_inverse, selected_channels, selected_event_ids,
                                                [l_freq, h_freq], t_min, t_max, seed)

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
    accuracy_significant, denominator_significant = get_accuracy_data(y_pred_significant, y_true_significant)
    accuracy_rest, denominator_rest = get_accuracy_data(y_pred_rest, y_true_rest)

    return accuracy_baseline / denominator_baseline, accuracy_significant / denominator_significant, accuracy_rest / denominator_rest, remaining_data_significant, remaining_data_rest


accuracies = {'subject': [], 'method': [], 'accuracy': [], 'remaining_data': []}
folder_path = fd.askdirectory()
for file in os.listdir(folder_path):
    file_name = os.fsdecode(file)
    file_name = file_name[:-4]
    parsed_index = file_name

    try:
        if stage == 1:
            heatmap_path = 'parafac_analysis/significant_heatmaps/{}.npy'.format(parsed_index)
        else:
            heatmap_path = 'parafac_analysis/significant_heatmaps_stage_2/{}.npy'.format(parsed_index)
        acc_b, acc_f, acc_r, remaining_data_significant, remaining_data_rest = test_subject(
            '{}/{}.edf'.format(folder_path, parsed_index),
            heatmap_path, seed=42)
        accuracies['subject'].append('s{}'.format(parsed_index))
        accuracies['method'].append('baseline')
        accuracies['remaining_data'].append(100)
        accuracies['accuracy'].append(acc_b)
        accuracies['subject'].append('s{}'.format(parsed_index))
        accuracies['method'].append('significant')
        accuracies['remaining_data'].append(remaining_data_significant)
        accuracies['accuracy'].append(acc_f)
        accuracies['subject'].append('s{}'.format(parsed_index))
        accuracies['method'].append('rest')
        accuracies['remaining_data'].append(remaining_data_rest)
        accuracies['accuracy'].append(acc_r)
    except Exception as e:
        print('ERROR with subject {}'.format(parsed_index))
        print(e)
        print(traceback.format_exc())

df = pd.DataFrame(data=accuracies)
df.to_csv('{}_csp_accuracies.csv'.format(folder_path.split('/')[-1]))

sns.set_theme(rc={'figure.figsize': (25, 10)})
ax = sns.barplot(x='subject', y='accuracy', hue='method', data=df)
for i in ax.containers:
    ax.bar_label(i, )
plt.show()
