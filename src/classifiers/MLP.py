from sklearn.neural_network import MLPClassifier


class MLP:
    def __init__(self, hidden_layer_sizes=(10, 10), random_state=1, max_iter=1000):
        self.classifier = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, random_state=random_state,
                                        max_iter=max_iter)

    def fit(self, x, y):
        return self.classifier.fit(x, y)

    def predict(self, x):
        return self.classifier.predict(x)

    def predict_proba(self, x):
        return self.classifier.predict_proba(x)

    def score(self, x, y):
        return self.classifier.score(x, y)
