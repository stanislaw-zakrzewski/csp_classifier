import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from mne.time_frequency import psd_array_multitaper

irasa_data = np.load('irasa_data.npy', allow_pickle=True)
irasa_data = irasa_data.item()


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

frac = irasa_data['frac'][1]
osc = irasa_data['osc'][1]
freq = irasa_data['freq']
a = psd_array_multitaper(irasa_data['original'], 512,6,26)
original = fft(irasa_data['original'][1])
original = np.abs(original)
print(len(original))

original_freq = fftfreq(len(original), 1.0/512)
orig = []
for fr in freq:
    orig.append(a[0][1][find_nearest(a[1], fr)])
orig = np.array(orig)
orig /= 500

f, axes = plt.subplots(2, 1)
sns.lineplot(x=freq, y=orig, label='Original signal', ax=axes[0])
sns.lineplot(x=freq, y=frac, label='Aperiodic activity', ax=axes[1])
sns.lineplot(x=freq, y=osc, label='Oscillatory activity', ax=axes[1])
axes[0].set_ylabel('Power (V^2/Hz)')
axes[1].set_ylabel('Power (V^2/Hz)')
axes[1].set_xlabel('Frequency (Hz)')
plt.show()