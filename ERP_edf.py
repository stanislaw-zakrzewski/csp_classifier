"""
 Sample script using EEGNet to classify Event-Related Potential (ERP) EEG data
 from a four-class classification task, using the sample dataset provided in
 the MNE [1, 2] package:
     https://martinos.org/mne/stable/manual/sample_dataset.html#ch-sample-data

 The four classes used from this dataset are:
     LA: Left-ear auditory stimulation
     RA: Right-ear auditory stimulation
     LV: Left visual field stimulation
     RV: Right visual field stimulation

 The code to process, filter and epoch the data are originally from Alexandre
 Barachant's PyRiemann [3] package, released under the BSD 3-clause. A copy of
 the BSD 3-clause license has been provided together with this software to
 comply with software licensing requirements.

 When you first run this script, MNE will download the dataset and prompt you
 to confirm the download location (defaults to ~/mne_data). Follow the prompts
 to continue. The dataset size is approx. 1.5GB download.

 For comparative purposes you can also compare EEGNet performance to using
 Riemannian geometric approaches with xDAWN spatial filtering [4-8] using
 PyRiemann (code provided below).

 [1] A. Gramfort, M. Luessi, E. Larson, D. Engemann, D. Strohmeier, C. Brodbeck,
     L. Parkkonen, M. Hämäläinen, MNE software for processing MEG and EEG data,
     NeuroImage, Volume 86, 1 February 2014, Pages 446-460, ISSN 1053-8119.

 [2] A. Gramfort, M. Luessi, E. Larson, D. Engemann, D. Strohmeier, C. Brodbeck,
     R. Goj, M. Jas, T. Brooks, L. Parkkonen, M. Hämäläinen, MEG and EEG data
     analysis with MNE-Python, Frontiers in Neuroscience, Volume 7, 2013.

 [3] https://github.com/alexandrebarachant/pyRiemann.

 [4] A. Barachant, M. Congedo ,"A Plug&Play P300 BCI Using Information Geometry"
     arXiv:1409.0107. link

 [5] M. Congedo, A. Barachant, A. Andreev ,"A New generation of Brain-Computer
     Interface Based on Riemannian Geometry", arXiv: 1310.8115.

 [6] A. Barachant and S. Bonnet, "Channel selection procedure using riemannian
     distance for BCI applications," in 2011 5th International IEEE/EMBS
     Conference on Neural Engineering (NER), 2011, 348-351.

 [7] A. Barachant, S. Bonnet, M. Congedo and C. Jutten, “Multiclass
     Brain-Computer Interface Classification by Riemannian Geometry,” in IEEE
     Transactions on Biomedical Engineering, vol. 59, no. 4, p. 920-928, 2012.

 [8] A. Barachant, S. Bonnet, M. Congedo and C. Jutten, “Classification of
     covariance matrices using a Riemannian-based kernel for BCI applications“,
     in NeuroComputing, vol. 112, p. 172-178, 2013.


 Portions of this project are works of the United States Government and are not
 subject to domestic copyright protection under 17 USC Sec. 105.  Those
 portions are released world-wide under the terms of the Creative Commons Zero
 1.0 (CC0) license.

 Other portions of this project are subject to domestic copyright protection
 under 17 USC Sec. 105.  Those portions are licensed under the Apache 2.0
 license.  The complete text of the license governing this material is in
 the file labeled LICENSE.TXT that is a part of this project's official
 distribution.
"""

import numpy as np

# mne imports
import mne
from mne import io
import os
from pathlib import Path
from mne.datasets import sample
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# EEGNet-specific imports
from EEGModels import EEGNet
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K

# PyRiemann imports
from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace
# from pyriemann.utils.viz import plot_confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# tools for plotting confusion matrices
from matplotlib import pyplot as plt

from data_classes.subject import Subject

FREQUENCY = 128  # Hz
LOWPASS_CUTOFF = 2  # Hz
HIGHPASS_CUTOFF = None  # Hz
EVENT_IDS = dict(left_movement=1, right_movement=2)
TIME_WINDOW = (.5, 2.5)  # seconds after start of the event

# while the default tensorflow ordering is 'channels_last' we set it here
# to be explicit in case if the user has changed the default ordering
K.set_image_data_format('channels_last')

dataset = [
    'preprocessed_subjects_car\\s01.edf',
    'preprocessed_subjects_car\\s03.edf',
    'preprocessed_subjects_car\\s04.edf',
    'preprocessed_subjects_car\\s06.edf',
    'preprocessed_subjects_car\\s14.edf',
    'preprocessed_subjects_car\\s23.edf',
    'preprocessed_subjects_car\\s35.edf',
    'preprocessed_subjects_car\\s41.edf',
    'preprocessed_subjects_car\\s43.edf',
    'preprocessed_subjects_car\\s50.edf',
]

def read_data(data_paths):
    raw = mne.io.read_raw_edf(data_paths[0], preload=True)
    events = mne.events_from_annotations(raw)[0]

    for data_path in data_paths[1:]:
        new_raw = mne.io.read_raw_edf(data_path, preload=True)
        new_events = mne.events_from_annotations(new_raw)[0]
        events = np.concatenate([events, new_events + [len(raw), 0, 0]], axis=0)
        raw.append(new_raw)
    all = []
    for i in events:
        if i[0] in all:
            print('=======', i[0])
        all.append(i[0])

    return raw, events


##################### Process, filter and epoch the data ######################
raw, events = read_data(dataset)
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
epochs = mne.Epochs(raw, events, EVENT_IDS, TIME_WINDOW[0], TIME_WINDOW[1], proj=False,
                    picks=picks, baseline=None, preload=True, verbose=False)
labels = epochs.events[:, -1]

# extract raw data. scale by 1000 due to scaling sensitivity in deep learning
X = epochs.get_data()  # * 100  # format is in (trials, channels, samples)

scaler = MinMaxScaler(feature_range=(-1, 1))
X = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(
    X.shape)  # 3D array cannot be scaled using this scaler so we turn it into 3d array

y = labels

permutation = np.random.RandomState(seed=23).permutation(len(X))
X = X[permutation]
y = y[permutation]

kernels, chans, samples = 1, 11, 257

# take 50/25/25 percent of the data to train/validate/test
quarter = int(len(X) / 4)
X_train = X[0:2 * quarter, ]
Y_train = y[0:2 * quarter]
X_validate = X[2 * quarter:3 * quarter, ]
Y_validate = y[2 * quarter:3 * quarter]
X_test = X[3 * quarter:, ]
Y_test = y[3 * quarter:]

############################# EEGNet portion ##################################

# convert labels to one-hot encodings.
Y_train = np_utils.to_categorical(Y_train - 1)
Y_validate = np_utils.to_categorical(Y_validate - 1)
Y_test = np_utils.to_categorical(Y_test - 1)

# convert data to NHWC (trials, channels, samples, kernels) format. Data
# contains 60 channels and 151 time-points. Set the number of kernels to 1.
X_train = X_train.reshape(X_train.shape[0], chans, samples, kernels)
X_validate = X_validate.reshape(X_validate.shape[0], chans, samples, kernels)
X_test = X_test.reshape(X_test.shape[0], chans, samples, kernels)

# configure the EEGNet-8,2,16 model with kernel length of 32 samples (other
# model configurations may do better, but this is a good starting point)
model = EEGNet(nb_classes=2, Chans=chans, Samples=samples,
               dropoutRate=0.5, kernLength=32, F1=4, D=4, F2=8,
               dropoutType='Dropout')

# compile the model and set the optimizers
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

# count number of parameters in the model
numParams = model.count_params()

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
fittedModel = model.fit(X_train, Y_train, batch_size=100, epochs=1000,
                        verbose=2, validation_data=(X_validate, Y_validate),
                        callbacks=[checkpointer], class_weight=class_weights)

plt.plot(fittedModel.history['accuracy'], label='accuracy')
plt.plot(fittedModel.history['val_accuracy'], label='val_accuracy')
plt.legend()
plt.show()
plt.plot(fittedModel.history['loss'], label='loss')
plt.plot(fittedModel.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

# load optimal weights
model.load_weights('/tmp/checkpoint.h5')

###############################################################################
# can alternatively used the weights provided in the repo. If so it should get
# you 93% accuracy. Change the WEIGHTS_PATH variable to wherever it is on your
# system.
###############################################################################

# WEIGHTS_PATH = /path/to/EEGNet-8-2-weights.h5
# model.load_weights(WEIGHTS_PATH)

###############################################################################
# make prediction on test set.
###############################################################################

probs = model.predict(X_test)
preds = probs.argmax(axis=-1)
acc = np.mean(preds == Y_test.argmax(axis=-1))
print("Classification accuracy: %f " % (acc))

# ############################# PyRiemann Portion ##############################
#
# # code is taken from PyRiemann's ERP sample script, which is decoding in
# # the tangent space with a logistic regression
#
# n_components = 2  # pick some components
#
# # set up sklearn pipeline
# clf = make_pipeline(XdawnCovariances(n_components),
#                     TangentSpace(metric='riemann'),
#                     LogisticRegression())
#
# preds_rg = np.zeros(len(Y_test))
#
# # reshape back to (trials, channels, samples)
# X_train = X_train.reshape(X_train.shape[0], chans, samples)
# X_test = X_test.reshape(X_test.shape[0], chans, samples)
#
# # train a classifier with xDAWN spatial filtering + Riemannian Geometry (RG)
# # labels need to be back in single-column format
# clf.fit(X_train, Y_train.argmax(axis=-1))
# preds_rg = clf.predict(X_test)
#
# # Printing the results
# acc2 = np.mean(preds_rg == Y_test.argmax(axis=-1))
# print("Classification accuracy: %f " % (acc2))

# plot the confusion matrices for both classifiers
names = ['left', 'right']
plt.figure(0)
# plot_confusion_matrix(preds, Y_test.argmax(axis=-1), names, title='EEGNet-8,2')
cm = confusion_matrix(Y_test.argmax(axis=-1), preds)
ConfusionMatrixDisplay(cm, display_labels=names).plot()
# plt.figure(1)
# # plot_confusion_matrix(preds_rg, Y_test.argmax(axis=-1), names, title='xDAWN + RG')
# cm = confusion_matrix(Y_test.argmax(axis=-1), preds_rg)
# ConfusionMatrixDisplay(cm, display_labels=names).plot()
plt.show()
