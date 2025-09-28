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

RANDOM_SEED = 23

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


def load_dataset(data_path, selected_labels):
    raw, events, label_names = read_data(data_path)
    original_sfreq = raw.info['sfreq']
    raw.filter(LOWPASS_CUTOFF, HIGHPASS_CUTOFF, method='iir')  # replace baselining with high-pass
    raw.resample(sfreq=FREQUENCY)  # resample
    events[:, 0] = np.round(events[:, 0] * (FREQUENCY / float(original_sfreq))).astype(
        int)  # adjust event placement to the resampling

    raw.info['bads'] = ['Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5',
                        'FC1', 'T7', 'TP7', 'CP5', 'CP1',
                        'P1', 'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'Iz', 'Oz',
                        'POz', 'Pz', 'CPz', 'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz', 'F2',
                        'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC2', 'FCz',
                        'T8', 'TP8', 'CP6', 'CP2', 'P2', 'P4', 'P6', 'P8', 'P10',
                        'PO8', 'PO4', 'O2']
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


def split_subject_data(data_path, selected_labels):
    x, y = load_dataset(data_path, selected_labels)

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


def run_classification(data_path, selected_labels):
    X_train, Y_train, X_validate, Y_validate, X_test, Y_test = split_subject_data(data_path, selected_labels)

    # convert labels to one-hot encodings.
    Y_train = np_utils.to_categorical(Y_train - 1)
    Y_validate = np_utils.to_categorical(Y_validate - 1)
    Y_test = np_utils.to_categorical(Y_test - 1)

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
    checkpointer = ModelCheckpoint(filepath='/tmp/checkpoint.h5', verbose=1,
                                   save_best_only=True)

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
              callbacks=[checkpointer], class_weight=class_weights)

    # load optimal weights
    # model.load_weights('/tmp/checkpoint.h5')

    probs = model.predict(X_test)
    sns.lineplot(x=hist.epoch,y=hist.history['accuracy'],label='Train Accuracy')
    sns.lineplot(x=hist.epoch,y=hist.history['val_accuracy'], label="Validation Accuracy")
    plt.show()
    preds = probs.argmax(axis=-1)
    acc = np.mean(preds == Y_test.argmax(axis=-1))

    cm = metrics.confusion_matrix(Y_test.argmax(axis=-1), preds)
    return acc, cm


def main():
    data_path = fd.askopenfilename(filetypes=[("European Data Format files", "*.edf")])
    subject = Subject(data_path)
    label_names = {}
    for label_name in subject.id_dict:
        label_names[subject.id_dict[label_name]] = label_name
    print('Available event types: ')
    unique_events, event_counts = np.unique(subject.events[:, 2], return_counts=True)
    cardinalities = dict(zip(unique_events, event_counts))
    for label_name in label_names:
        print('\t* {}: {} (cardinality: {})'.format(label_name, label_names[label_name], cardinalities[label_name]))
    selected_labels = [int(selected_label) for selected_label in (input("Select labels (comma separated):").split(','))]
    selected_labels = {k: v for k, v in subject.id_dict.items() if v in selected_labels}
    final_accuracy, confusion_matrix = run_classification(data_path, selected_labels)
    print(final_accuracy, confusion_matrix)
main()