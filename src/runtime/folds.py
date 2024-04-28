from sklearn.model_selection import ShuffleSplit

from config.config import Configurations


def folds_wrapper():
    configurations = Configurations()
    n_splits = configurations.read('analyze_data.n_splits')
    test_size = configurations.read('analyze_data.test_size')
    random_state = configurations.read('analyze_data.random_state')
    cv = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
