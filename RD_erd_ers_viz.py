from tkinter import filedialog as fd
from src.preprocessing.bandpass_filter import bandpass_filter
from mne import Epochs, pick_types, preprocessing
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from data_classes.subject import Subject


def moving_average(x, w):
    return list(np.convolve(x, np.ones(w), 'valid') / w)

T_MIN = -1.5
T_MAX = 4.5

# subject_path = fd.askopenfilename(filetypes=[("EDF EEG signal file", "*.edf")])
subject_path = 'preprocessed_subjects/s43.edf'
subject = Subject(subject_path)

MOVING_AVERAGE_STEP = 100
verbose = 'CRITICAL'
ALPHA_BAND = [8, 12]
BETA_BAND = [15, 25]
selected_event_ids = [1, 2]
events = subject.events
alpha_raw_signal = subject.get_raw_copy()
beta_raw_signal = subject.get_raw_copy()
sampling_frequency = subject.sampling_frequency

alpha_raw_signal = bandpass_filter(alpha_raw_signal, ALPHA_BAND[0], ALPHA_BAND[1], sampling_frequency)
beta_raw_signal = bandpass_filter(beta_raw_signal, BETA_BAND[0], BETA_BAND[1], sampling_frequency)

selected_events = [e for e in events if e[-1] in selected_event_ids]
picks = pick_types(alpha_raw_signal.info, meg=False, eeg=True, stim=False, eog=False,
                   exclude='bads')

alpha_epochs = Epochs(alpha_raw_signal, selected_events, selected_event_ids, T_MIN, T_MAX, proj=True, picks=picks,
                      baseline=None, preload=True, verbose=verbose)
beta_epochs = Epochs(beta_raw_signal, selected_events, selected_event_ids, T_MIN, T_MAX, proj=True, picks=picks,
                     baseline=None, preload=True, verbose=verbose)
epochs_data = alpha_epochs.get_data(copy=False)
beta_epochs_data = beta_epochs.get_data(copy=False)
labels = np.array(alpha_epochs.events[:, -1])

c3_left = []
c4_left = []
c3_right = []
c4_right = []

beta_c3_left = []
beta_c4_left = []
beta_c3_right = []
beta_c4_right = []

c3_idx = alpha_epochs.ch_names.index('C3')
c4_idx = alpha_epochs.ch_names.index('C4')

for trial_idx, label in enumerate(labels):
    trial = epochs_data[trial_idx]
    trial_c3 = trial[c3_idx] ** 2
    trial_c4 = trial[c4_idx] ** 2

    beta_trial = beta_epochs_data[trial_idx]
    beta_trial_c3 = beta_trial[c3_idx] ** 2
    beta_trial_c4 = beta_trial[c4_idx] ** 2
    if label == 1:
        c3_left.append(trial_c3)
        c4_left.append(trial_c4)

        beta_c3_left.append(beta_trial_c3)
        beta_c4_left.append(beta_trial_c4)
    else:
        c3_right.append(trial_c3)
        c4_right.append(trial_c4)

        beta_c3_right.append(beta_trial_c3)
        beta_c4_right.append(beta_trial_c4)

c3_left = moving_average(np.mean(c3_left, axis=0), MOVING_AVERAGE_STEP)
c4_left = moving_average(np.mean(c4_left, axis=0), MOVING_AVERAGE_STEP)
c3_right = moving_average(np.mean(c3_right, axis=0), MOVING_AVERAGE_STEP)
c4_right = moving_average(np.mean(c4_right, axis=0), MOVING_AVERAGE_STEP)
beta_c3_left = moving_average(np.mean(beta_c3_left, axis=0), MOVING_AVERAGE_STEP)
beta_c4_left = moving_average(np.mean(beta_c4_left, axis=0), MOVING_AVERAGE_STEP)
beta_c3_right = moving_average(np.mean(beta_c3_right, axis=0), MOVING_AVERAGE_STEP)
beta_c4_right = moving_average(np.mean(beta_c4_right, axis=0), MOVING_AVERAGE_STEP)
timepoints = np.linspace(T_MIN, T_MAX, len(c3_left))

c3_c4_comparison_data = {
    'left': [*c3_left, *c4_left],
    'electrode': [*['C3'] * len(c3_left), *['C4'] * len(c4_left)],
    'right': [*c3_right, *c4_right],
    'timepoints': [*timepoints, *timepoints],
}
c3_c4_comparison = pd.DataFrame(data=c3_c4_comparison_data)

fig, ax = plt.subplots(1,1)
sns.lineplot(c3_c4_comparison, x='timepoints', y='left', hue='electrode')
plt.axvline(0, c='black', ls='--', label='cue start')
plt.axvline(3, c='black', ls='--', label='cue end')
plt.text(3, 0.99, 'median', color='r', ha='right', va='top', rotation=90,
        transform=ax.get_xaxis_transform())
plt.show()
# sns.lineplot(c3_c4_comparison, x='timepoints', y='right', hue='electrode')
# plt.axvline(0, c='black', ls='-', label='cue start')
# plt.axvline(3, c='black', ls='-', label='cue end')
# plt.show()
#
# alpha_beta_comparison_data = {
#     'left': [*c4_left, *beta_c4_left],
#     'band': [*(['alfa'] * len(c3_left)), *(['beta'] * len(beta_c4_left))],
#     'right': [*c3_right, *beta_c3_right],
#     'timepoints': [*timepoints, *timepoints],
# }
# alpha_beta_comparison = pd.DataFrame(data=alpha_beta_comparison_data)
#
# sns.lineplot(alpha_beta_comparison, x='timepoints', y='left', hue='band')
# plt.show()
# sns.lineplot(alpha_beta_comparison, x='timepoints', y='right', hue='band')
# plt.show()
