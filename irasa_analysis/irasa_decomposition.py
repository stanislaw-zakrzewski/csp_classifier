import numpy as np
from neurodsp.sim import sim_combined
from neurodsp.spectral import compute_spectrum, trim_spectrum
from neurodsp.plts import plot_power_spectra

# Import IRASA related functions
from neurodsp.aperiodic import compute_irasa, fit_irasa