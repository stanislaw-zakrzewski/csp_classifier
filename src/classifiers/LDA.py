from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class LDA:
    def __init__(self):
        self.classifier = LinearDiscriminantAnalysis()

    def fit(self, x, y):
        return self.classifier.fit(x, y)

    def predict(self, x):
        return self.classifier.predict(x)

    def predict_proba(self, x):
        return self.classifier.predict_proba(x)

    def score(self, x, y):
        return self.classifier.score(x, y)
