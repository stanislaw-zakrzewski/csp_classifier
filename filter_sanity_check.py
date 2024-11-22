import matplotlib.pyplot as plt
from mne.filter import filter_data, notch_filter
import numpy as np
from scipy.fft import rfft, rfftfreq

# Desired bands and frequency lowpass cutoff
alpha_band = [9.5,10.5]
beta_band = [19, 25]
max_frequency = 30

# Whitenoise data
sampling_rate = 512
length_in_seconds = 10
mean = 0
std = .1
num_samples = sampling_rate * length_in_seconds

# Filter data
l_trans_bandwidth = .5
h_trans_bandwidth = .5
filter_length = 7 * sampling_rate

# Generate whitenoise data
samples = np.random.normal(mean, std, size=num_samples)

# Test adding filtering
adding_filtered_signals = notch_filter(samples, sampling_rate, 10, trans_bandwidth=1)
# adding_filtered_signals = [
#     filter_data(samples, sampling_rate, alpha_band[0], alpha_band[1], filter_length=filter_length,
#                 l_trans_bandwidth=l_trans_bandwidth, h_trans_bandwidth=h_trans_bandwidth),
#     filter_data(samples, sampling_rate, beta_band[0], beta_band[1], filter_length=filter_length,
#                 l_trans_bandwidth=l_trans_bandwidth, h_trans_bandwidth=h_trans_bandwidth)]
# adding_filtered_signals = np.sum(adding_filtered_signals, axis=0)

# Test band stop filtering
band_stop_filtered_signals = notch_filter(samples, sampling_rate, 10)
# band_stop_filtered_signals = filter_data(samples, sampling_rate, alpha_band[0], beta_band[1],
#                                          filter_length=filter_length, l_trans_bandwidth=l_trans_bandwidth,
#                                          h_trans_bandwidth=h_trans_bandwidth)
# band_stop_filtered_signals = filter_data(band_stop_filtered_signals, sampling_rate, beta_band[0], alpha_band[1],
#                                          filter_length=filter_length, l_trans_bandwidth=l_trans_bandwidth,
#                                          h_trans_bandwidth=h_trans_bandwidth)

# Get list of frequencies for fft output
frequencies = rfftfreq(len(samples), 1 / sampling_rate)

# Get cutoff index for lowpass cutoff
max_frequency_index = len(list(filter(lambda x: x < max_frequency, frequencies)))

# Limit list of frequencies to max_frequency
frequencies = frequencies[:max_frequency_index]

# Perform fft on all three signals
yf_original = rfft(samples)
yf_adding_filtered_signals = rfft(adding_filtered_signals)
yf_band_stop_filtered_signals = rfft(band_stop_filtered_signals)

# Take only real part of filtered data and limit to max frequency
original_spectral = np.abs(yf_original)[:max_frequency_index]
adding_filtered_signals_spectral = np.abs(yf_adding_filtered_signals)[:max_frequency_index]
band_stop_filtered_signals_spectral = np.abs(yf_band_stop_filtered_signals)[:max_frequency_index]

# Plot the results
fig, ax = plt.subplots()
fig.set_figheight(8)
fig.set_figwidth(15)
ax.plot(frequencies, original_spectral, label='Original')
ax.plot(frequencies, adding_filtered_signals_spectral, label='Adding')
ax.plot(frequencies, band_stop_filtered_signals_spectral, '--', label='Band Stop')
ax.axvspan(alpha_band[0], alpha_band[1], alpha=0.2, color='gray')
ax.axvspan(beta_band[0], beta_band[1], alpha=0.2, color='blue')
plt.legend()
plt.ylabel('Amplitude')
plt.xlabel('Frequency (Hz)')
plt.show()
