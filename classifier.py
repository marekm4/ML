import math

import numpy as np


class Classifier:
    def __init__(self, n_neighbors=20, gamma=0.9):
        self.n_neighbors = n_neighbors
        self.gamma = gamma
        self.X = None
        self.y = None
        self.n_classes = None

    def fit(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)

    def distance(self, A, B):
        return math.sqrt(math.pow(A[0] - B[0], 2) + math.pow(A[1] - B[1], 2))

    def predict(self, X):
        predictions = []
        for x in X:
            distances = np.array([self.distance(x, d) for d in self.X])
            closest = distances.argsort()[: self.n_neighbors]
            results = {}
            for c in closest:
                value = self.y[c]
                if value not in results:
                    results[value] = []
                results[value].append(distances[c])
            for key, value in results.items():
                results[key] = np.mean(value) * math.pow(self.gamma, len(value))
            predictions.append(min(results, key=results.get))
        return predictions
