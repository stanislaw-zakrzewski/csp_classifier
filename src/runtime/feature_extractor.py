from config.config import Configurations
from src.feature_extractors.CSP import CSP


class FeatureExtraction:
    def __init__(self):
        self.configurations = Configurations()
        self.csp = CSP(10)

    def fit_transform(self, x, y):
        return self.csp.fit_transform(x, y)

    def transform(self, x):
        return self.csp.transform(x)
