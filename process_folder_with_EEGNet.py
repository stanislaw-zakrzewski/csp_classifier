from time import time

from skopt import gp_minimize

from data_classes.subject import Subject
from tkinter import filedialog as fd
from itertools import permutations
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn import metrics
import numpy as np
import mne
from sklearn.preprocessing import MinMaxScaler
from EEGModels import EEGNet
import seaborn as sns
import matplotlib.pyplot as plt
import os
import traceback
import pandas as pd
from sklearn.cluster import KMeans
from skopt.space import Integer, Categorical
from skopt.utils import use_named_args

RANDOM_SEED = 23
SELECTED_LABELS = []

FREQUENCY = 128  # Hz
LOWPASS_CUTOFF = 2  # Hz
HIGHPASS_CUTOFF = None  # Hz
TIME_WINDOW = (.0, 1.0)  # seconds after start of the event
EPOCHS = 200
BATCH_SIZE = 20
F1 = 8
D = 2
F2 = 16


def read_data(data_path):
    raw = mne.io.read_raw_edf(data_path, preload=True)
    events, labels = mne.events_from_annotations(raw)
    return raw, events, labels


def load_dataset(data_path, selected_labels, bads):
    raw, events, label_names = read_data(data_path)
    original_sfreq = raw.info['sfreq']
    raw.filter(LOWPASS_CUTOFF, HIGHPASS_CUTOFF, method='iir')  # replace baselining with high-pass
    raw.resample(sfreq=FREQUENCY)  # resample
    events[:, 0] = np.round(events[:, 0] * (FREQUENCY / float(original_sfreq))).astype(
        int)  # adjust event placement to the resampling

    raw.info['bads'] = bads
    picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                           exclude='bads')

    # Read epochs
    epochs = mne.Epochs(raw, events, selected_labels, TIME_WINDOW[0], TIME_WINDOW[1], proj=False,
                        picks=picks, baseline=None, preload=True, verbose=False)
    labels = epochs.events[:, -1]

    # extract raw data. scale by 1000 due to scaling sensitivity in deep learning
    X = epochs.get_data()  # * 100  # format is in (trials, channels, samples)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    X = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(
        X.shape)  # 3D array cannot be scaled using this scaler so we turn it into 3d array

    y = labels

    return X, y


def split_subject_data(data_path, selected_labels, bads):
    x, y = load_dataset(data_path, selected_labels, bads)

    permutation = np.random.RandomState(seed=RANDOM_SEED).permutation(len(x))
    x = x[permutation]
    y = y[permutation]

    # take 60/20/20 percent of the data to train/validate/test
    fifth = int(len(x) / 5)
    x_train = x[0:3 * fifth, ]
    y_train = y[0:3 * fifth]
    x_validate = x[3 * fifth:4 * fifth, ]
    y_validate = y[3 * fifth:4 * fifth]
    x_test = x[4 * fifth:, ]
    y_test = y[4 * fifth:]

    return x_train, y_train, x_validate, y_validate, x_test, y_test

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

def fix_labels(y):
    orig = y
    first = min(y)
    if first != 0:
        y = y - first
    y = np.clip(y, 0, 1)
    return y


def run_classification(data_path, selected_labels, bads, seed, significant):
    x, y = load_dataset(data_path, selected_labels, bads)
    y = fix_labels(y)
    folds = get_folds(y, seed)
    acc_all = []
    space = [Integer(2, 8, name='F1'), Integer(2, 16, name='F2'), Categorical([50,100,150,200], name='EPOCHS')]
    # if significant:
    #     F1 = 6
    #     D = 14
    #     F2 = 200
    # else:
    #     F1 = 7
    #     D = 9
    #     F2 = 200

    @use_named_args(space)
    def objective(**params):
        F1 = params['F1']
        F2 = params['F2']
        EPOCHS = params['EPOCHS']
        acc_all = []
        for fold in folds:
            X_train = x[fold['train']]
            X_validate = x[fold['validate']]
            X_test = x[fold['test']]
            Y_train = y[fold['train']]
            Y_validate = y[fold['validate']]
            Y_test = y[fold['test']]

            # convert labels to one-hot encodings.
            Y_train = np_utils.to_categorical(Y_train)
            Y_validate = np_utils.to_categorical(Y_validate)
            Y_test = np_utils.to_categorical(Y_test)

            kernels, chans, samples = 1, X_train.shape[1], X_train.shape[2]
            # convert data to NHWC (trials, channels, samples, kernels) format. Data
            # contains 60 channels and 151 time-points. Set the number of kernels to 1.
            X_train = X_train.reshape(X_train.shape[0], chans, samples, kernels)
            X_validate = X_validate.reshape(X_validate.shape[0], chans, samples, kernels)
            X_test = X_test.reshape(X_test.shape[0], chans, samples, kernels)

            # configure the EEGNet-8,2,16 model with kernel length of 32 samples (other
            # model configurations may do better, but this is a good starting point)
            model = EEGNet(nb_classes=len(selected_labels), Chans=chans, Samples=samples,
                           dropoutRate=0.5, kernLength=32, F1=F1, D=D, F2=F2,
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
            class_weights = {0: 1, 1: 1}

            ################################################################################
            # fit the model. Due to very small sample sizes this can get
            # pretty noisy run-to-run, but most runs should be comparable to xDAWN +
            # Riemannian geometry classification (below)
            ################################################################################
            hist = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=EPOCHS,
                      verbose=2, validation_data=(X_validate, Y_validate),
                      # callbacks=[checkpointer],
                             class_weight=class_weights)

            # load optimal weights
            # model.load_weights('/tmp/checkpoint.h5')

            probs = model.predict(X_test)
            # sns.lineplot(x=hist.epoch,y=hist.history['accuracy'],label='Train Accuracy')
            # sns.lineplot(x=hist.epoch,y=hist.history['val_accuracy'], label="Validation Accuracy")
            # plt.show()
            preds = probs.argmax(axis=-1)
            acc = np.mean(preds == Y_test.argmax(axis=-1))
            acc_all.append(acc)

            return 1 - np.average(acc_all)

    res_gp = gp_minimize(objective, space, n_calls=40, random_state=0)
    # data_to_save = {
    #     'x': res_gp['x'],
    #     'x_iters': res_gp['x_iters'],
    #     'func_vals': res_gp['func_vals']
    # }
    # if significant:
    #     np.save( 'significant_optimizer.npy', data_to_save, allow_pickle=True)
    # else:
    #     np.save( 'all_optimizer.npy', data_to_save, allow_pickle=True)
    return 1
    return np.average(acc_all)


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



def main(data_path, heatmap_path, seed):
    global SELECTED_LABELS
    # data_path = fd.askopenfilename(filetypes=[("European Data Format files", "*.edf")])
    subject = Subject(data_path)
    heatmap = np.load(heatmap_path, allow_pickle=True).item()
    label_names = {}
    for label_name in subject.id_dict:
        label_names[subject.id_dict[label_name]] = label_name
    print('Available event types: ')
    unique_events, event_counts = np.unique(subject.events[:, 2], return_counts=True)
    cardinalities = dict(zip(unique_events, event_counts))
    for label_name in label_names:
        print('\t* {}: {} (cardinality: {})'.format(label_name, label_names[label_name], cardinalities[label_name]))
    if len(SELECTED_LABELS) == 0:
        SELECTED_LABELS = [int(selected_label) for selected_label in (input("Select labels (comma separated):").split(','))]
        slected_labels = {k: v for k, v in subject.id_dict.items() if v in SELECTED_LABELS}
        SELECTED_LABELS = slected_labels.keys()
    else:
        slected_labels = {k: v for k, v in subject.id_dict.items() if k in SELECTED_LABELS}

    differences = np.abs(heatmap['heatmap_a'] - heatmap['heatmap_b'])
    differences_all_placement, percentage_differences_all = get_significant_placement(differences)
    differences_significant_placement, percentage_differences_significant = get_significant_placement(differences,
                                                                                                      cluster_count=2)
    good_a = list(np.array(heatmap['channels'])[list(differences_all_placement.keys())])
    good_s = list(np.array(heatmap['channels'])[list(differences_significant_placement.keys())])
    bads_a = [chn for chn in subject.raw.ch_names if chn not in good_a]
    bads_s = [chn for chn in subject.raw.ch_names if chn not in good_s]

    start = time()
    eeg_accuracy = run_classification(data_path, slected_labels, bads_a, seed, False)
    time_eegnet = time() - start
    start = time()
    eeg_significant_accuracy = run_classification(data_path, slected_labels, bads_s, seed, True)
    time_eegnet_s = time() - start
    return eeg_accuracy, eeg_significant_accuracy, percentage_differences_all, percentage_differences_significant,time_eegnet,time_eegnet_s


accuracies = {'subject': [], 'method': [], 'accuracy': [], 'percentage_of_data_used': [], 'time': []}
# folder_path = fd.askdirectory()

folder_path = 'dataset_temp'
start = time()
for file in os.listdir(folder_path):
    file_name = os.fsdecode(file)
    file_name = file_name[:-4]
    parsed_index = file_name

    try:
        heatmap_path = 'parafac_analysis/significant_heatmaps/{}.npy'.format(parsed_index)

        acc_en, acc_en_s, p_a, p_s, t_en, t_en_s = main('{}/{}.edf'.format(folder_path, parsed_index), heatmap_path, RANDOM_SEED)
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
df.to_csv('{}_accuracies_eegnet.csv'.format(folder_path.split('/')[-1]))
print('RUNTIME:', time() - start)