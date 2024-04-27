import numpy as np
from mne.channels import make_standard_montage
from mne.io import read_raw_edf
from mne.datasets import eegbci
import mne.viz
from mne.time_frequency import tfr_stockwell
import matplotlib.pyplot as plt


raw = read_raw_edf('C:/Users/I72/Documents/GitHub/csp_classifier/data/2024-02-22T11-02-49.edf', preload=True)

event_id = dict(rest=1, movement=2)

print(raw.info)
eegbci.standardize(raw)
montage = make_standard_montage("standard_1005")
raw.set_montage(montage)
raw.pick(["FC3", "FC4", "C1", "C2", "C3", "C4", "C5", "C6", "Cz", "CP4", "CP3"])
raw.filter(4.0, 28.0)

# mne.viz.plot_raw(raw, block=True)

events = mne.events_from_annotations(raw)
epochs = mne.Epochs(raw, events[0], event_id=event_id, tmin=-1, tmax=7, preload=True)

epochs["rest"][0].compute_psd().plot()
plt.show()


epochs["movement"][0].compute_psd().plot()
plt.show()

power = tfr_stockwell(epochs['movement'][4], fmin=4, fmax=28, width=0.2)
power.plot(picks=('C3', 'C4'), mode='mean')
plt.show()