import matplotlib.pyplot as plt
import numpy as np
from tkinter import filedialog as fd
from matplotlib.colors import TwoSlopeNorm
from data_classes.subject import Subject

import mne
from mne.io import read_raw_edf
from mne.stats import permutation_cluster_1samp_test as pcluster_test

subject_edf_path = fd.askopenfilename(filetypes=[("European Data Format files", "*.edf")])
# subject_edf_path = 'preprocessed_subjects/s43.edf'

subject = Subject(subject_edf_path)
raw = subject.get_raw_copy()

selected_frequency_band = input('Selected frequency band (default: "2,36"): ')
if selected_frequency_band:
    selected_frequency_band = selected_frequency_band.split(',')
    selected_frequency_band = (int(selected_frequency_band[0]), int(selected_frequency_band[1]))
else:
    selected_frequency_band = (2, 36)

t_min = input('Time from cue to start cut (default: -1): ')
t_min = float(t_min) if t_min != '' else -1.

t_max = input('Time from cue to end cut (default: 4): ')
t_max = float(t_max) if t_max != '' else 4.0

print('Available channels: {}'.format(', '.join(subject.electrode_names)))
selected_channels = input('Select 3 channels (default: "C3,Cz,C4"): ')
if selected_channels:
    selected_channels = selected_channels.split(',')
else:
    selected_channels = ['C3', 'Cz', 'C4']

label_names = {}
for label_name in subject.id_dict:
    label_names[subject.id_dict[label_name]] = label_name

print("Available labels:")
unique_events, event_counts = np.unique(subject.events[:, 2], return_counts=True)
cardinalities = dict(zip(unique_events, event_counts))

for label_name in label_names:
    print('\t* {}: {} (cardinality: {})'.format(label_name, label_names[label_name], cardinalities[label_name]))
selected_label_1 = int(input("Select first label:"))
selected_label_2 = int(input("Select second label:"))
selected_label_1_name = label_names[selected_label_1]
selected_label_2_name = label_names[selected_label_2]
selected_labels = {
    selected_label_1_name: selected_label_1,
    selected_label_2_name: selected_label_2
}

# event_ids = dict(left=2, right=3)  # map event IDs to tasks

epochs = mne.Epochs(
    raw,
    event_id=["left", "right"],
    tmin=t_min - 0.5,
    tmax=t_max + 0.5,
    picks=("C3", "Cz", "C4"),
    baseline=None,
    preload=True,
)

freqs = np.arange(selected_frequency_band[0], selected_frequency_band[1])  # frequencies from 2-35Hz
vmin, vmax = -1, 1.5  # set min and max ERDS values in plot
baseline = (-1, 0)  # baseline interval (in s)
cnorm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)  # min, center & max ERDS

kwargs = dict(
    n_permutations=100, step_down_p=0.05, seed=1, buffer_size=None, out_type="mask"
)  # for cluster test

tfr = epochs.compute_tfr(
    method="multitaper",
    freqs=freqs,
    n_cycles=freqs,
    use_fft=True,
    return_itc=False,
    average=False,
    decim=2,
)
tfr.crop(t_min, t_max).apply_baseline(baseline, mode="percent")

for event in selected_labels:
    # select desired epochs for visualization
    tfr_ev = tfr[event]
    fig, axes = plt.subplots(
        1, 4, figsize=(12, 4), gridspec_kw={"width_ratios": [10, 10, 10, 1]}
    )
    for ch, ax in enumerate(axes[:-1]):  # for each channel
        # positive clusters
        _, c1, p1, _ = pcluster_test(tfr_ev.data[:, ch], tail=1, **kwargs)
        # negative clusters
        _, c2, p2, _ = pcluster_test(tfr_ev.data[:, ch], tail=-1, **kwargs)

        # note that we keep clusters with p <= 0.05 from the combined clusters
        # of two independent tests; in this example, we do not correct for
        # these two comparisons
        c = np.stack(c1 + c2, axis=2)  # combined clusters
        p = np.concatenate((p1, p2))  # combined p-values
        mask = c[..., p <= 0.05].any(axis=-1)

        # plot TFR (ERDS map with masking)
        tfr_ev.average().plot(
            [ch],
            cmap="RdBu",
            cnorm=cnorm,
            axes=ax,
            colorbar=False,
            show=False,
            mask=mask,
            mask_style="mask",
        )

        ax.set_title(epochs.ch_names[ch], fontsize=10)
        ax.axvline(0, linewidth=1, color="black", linestyle=":")  # event
        if ch != 0:
            ax.set_ylabel("")
            ax.set_yticklabels("")
    fig.colorbar(axes[0].images[-1], cax=axes[-1]).ax.set_yscale("linear")
    fig.suptitle(f"ERDS ({event})")
    plt.show()