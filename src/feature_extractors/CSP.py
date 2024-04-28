from mne.decoding import CSP as MNE_CSP

from config.config import Configurations


class CSP:
    def __init__(self, n_components, reg=None):
        self.n_components = n_components
        self.configurations = Configurations()
        self.verbose = self.configurations.read('general.verbose')
        self.csp = MNE_CSP(n_components=n_components, reg=reg, log=True, norm_trace=False)

    def fit_transform(self, x, y):
        return self.csp.fit_transform(x, y)

    def transform(self, x):
        return self.csp.transform(x)
