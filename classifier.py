from sklearn.linear_model import LogisticRegression


class Classifier:
    def __init__(self):
        self.classifier = LogisticRegression()

    def fit(self, X, y):
        return self.classifier.fit(X, y)

    def predict(self, X):
        return self.classifier.predict(X)
