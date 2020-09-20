import numpy as np
import pandas as pd
import random

class GDLinearRegressionModel():

    def __init__(self, learning_rate = 0.01, iterations = 2000):
        self.eta = learning_rate
        self.iterations = iterations


    def computeCostLinear(self, x, y, w, b, m):
        sum = 0
        for i, j in zip(x, y):
            temp = i * w + b - j
            sum += temp ** 2
        return 1/(2*m) * sum

    def computeCostMulti(self, X, y, w, b, m):
        J = 1/(2*m) * np.sum((X.dot(w) + b - y)**2)
        return J

    def get_wb_linear(self, X, y, m):
        self.history = []
        w, b = 0, 0

        for i in range(self.iterations):
            temp = b - self.eta * 1/m * np.sum(((w * X + b) - y))
            a = (X.T.dot((w * X + b) - y))
            w = w - self.eta * 1/m * (X.T.dot((w * X + b) - y))
            b = temp
            self.history.append(self.computeCostLinear(X, y, w, b, m))
        self.intercept, self.slope = b, w
        return


    def get_wb_multi(self, X, y, m):
        b = np.array([0])
        w = np.array([0] * len(X.T))
        self.history = []
        for i in range(self.iterations):
            temp = b - self.eta * 1/m * np.sum(((X.dot(w) + b) - y))
            w = w - (self.eta * 1/m * X.T.dot((X.dot(w) + b) - y))
            b = temp
            self.history.append(self.computeCostMulti(X, y, w, b, m))
        self.intercept, self.slope = b, w
        return

    def fit(self, X, y):
        m = len(y)
        try:
            lx = len(X[0])
            try:
                self.get_wb_multi(X.T, y, m)
            except:
                print("error in multi")
        except:
            self.get_wb_linear(X, y, m)
        return

    def predict(self, X, w = None, b = None):
        if w == None:
            w = self.slope
            b = self.intercept

        return X.dot(w) + b
