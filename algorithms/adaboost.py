import math
import numpy as np
from sklearn.tree import DecisionTreeClassifier


class AdaBoost:
    """ Implemented as per the description in
    https://machinelearningmastery.com/boosting-and-adaboost-for-machine-learning/

    AdaBoost is a binary classifier as such the y labels are expected to be -1 or +1 values only
    """
    def __init__(self):
        self.classifiers = []
        self.classifier_weights = []

    def fit(self, X, y, max_clf_count=10):
        weights = np.ones(len(y)) / len(y)

        for _ in range(max_clf_count):
            clf = DecisionTreeClassifier(max_depth=1)
            clf.fit(X, y, weights)
            element_wise_error = np.abs(np.sign(y - clf.predict(X)))
            model_error = self.compute_model_error(element_wise_error, weights)
            stage = math.log((1 - model_error) / model_error)

            self.classifiers.append(clf)
            self.classifier_weights.append(stage)

            # update the weights
            weights = weights * np.exp(stage * element_wise_error)

    def predict(self, X):
        y = np.transpose(np.array([clf.predict(X) for clf in self.classifiers]))
        return np.sign(np.sum(self.classifier_weights * y, axis=1))

    @staticmethod
    def compute_model_error(element_wise_error, weights) -> float:
        return np.sum(element_wise_error * weights) / np.sum(weights)

