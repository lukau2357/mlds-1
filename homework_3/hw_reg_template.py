import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from numpy.linalg import inv, norm
from numpy.random import default_rng
from scipy.optimize import minimize

class RidgeReg:
    def __init__(self, l):
        self.l = l
        
    def fit(self, X, y):
        """
        Closed form solution for ridge regression, given l = lambda > 0. The commented
        part represents a solution without the intercept term, assuming X and y are standardized
        (or centered at the very least).
        """
        # self.beta = inv(X.T.dot(X) + self.l * np.identity(X.shape[1])).dot(X.T).dot(y)
        X = np.c_[np.ones(X.shape[0]), X]
        t = self.l * np.identity(X.shape[1])
        # Do not penalize the intercept!
        t[0, 0] = 0
        self.beta = inv(X.T.dot(X) + t).dot(X.T).dot(y)

    def predict(self, X):
        """
        Commented part refers to the implementation of ridge regression without the
        intercept.
        """
        # return np.dot(X, self.beta)
        return np.dot(X, self.beta[1:]) + self.beta[0]

def lasso_objective(beta, Xp, y, l):
     """
     Objective function for lasso regularization
     Xp stands for data matrix X padded with ones for convenient implementation.
     """
     return norm(Xp.dot(beta) - y) ** 2 + l * np.sum(np.abs(beta[1:])) 

class LassoReg:
    def __init__(self, l, seed = 41):
        """
        We draw initial value for our weights from 0 mean unit variance Laplace distribution.
        We opted for this distribution since Laplace is the prior distribution in the Bayesian
        interpretation of L1 regularization. Seed is just a number used for reproducibility.
        """
        self.l = l
        self.seed = seed

    def fit(self, X, y):
        rnd = default_rng(self.seed)
        self.beta = rnd.laplace(0, 1, X.shape[1] + 1)
        Xp = np.c_[np.ones(X.shape[0]), X]
        self.beta = (minimize(lasso_objective, self.beta, args = (Xp, y, self.l), method = "Powell")).x

    def predict(self, X):
        return X.dot(self.beta[1:]) + self.beta[0]

class RegularizationTest(unittest.TestCase):
    def test_ridge_simple(self):
        X = np.array([[1],
                      [10],
                      [100]])
        y = 10 + 2 * X[:,0]
        model = RidgeReg(1)
        model.fit(X, y)
        y = model.predict(np.array([[10],
                           [20]]))
        self.assertAlmostEqual(y[0], 30, delta = 0.1)
        self.assertAlmostEqual(y[1], 50, delta = 0.1)

    def test_lasso_simple(self):
        X = np.array([[1],
                      [10],
                      [100]])
        y = 10 + 2 * X[:,0]
        model = LassoReg(1)
        model.fit(X, y)
        y = model.predict(np.array([[10],
                           [20]]))
        self.assertAlmostEqual(y[0], 30, delta = 0.1)
        self.assertAlmostEqual(y[1], 50, delta = 0.1)
    # ... add your tests

def load(fname):
    df = pd.read_csv(fname)
    target = "critical_temp"
    
    X = df.loc[:, df.columns != target].to_numpy()
    y = df[target].to_numpy()

    X_train, y_train = X[:200, :], y[:200]
    X_test, y_test = X[200:, :], y[200:]

    return df.columns, X_train, y_train, X_test, y_test

def superconductor(X_train, y_train, X_test, y_test):
    lambdas = np.arange(0, 2, step = 0.01)
    rmse_test = []

    mu_x, std_x = X_train.mean(), X_train.std()
    mu_y, std_y = y_train.mean(), y_train.std()

    X_train = (X_train - mu_x) / std_x
    y_train = (y_train - mu_y) / std_y

    # Standardize the test data using Mu and Sigma of the training data!
    X_test = (X_test - mu_x) / std_x
    y_test = (y_test - mu_y) / std_y

    for i in range(len(lambdas)):
        model = RidgeReg(lambdas[i])
        model.fit(X_train, y_train)
        rmse_test.append(rmse(model, X_test, y_test))

    print("Best RMSE {:f} obtained for regularization rate of {:f}".format(min(rmse_test), lambdas[np.argmin(rmse_test)]))
    plt.style.use("ggplot")
    fig, ax = plt.subplots()
    ax.plot(lambdas, rmse_test, label = "RMSE test")
    plt.show()

def rmse(model, X, y):
    preds = model.predict(X)
    N = X.shape[0]
    return np.math.sqrt((1 / N) * np.sum((preds - y) ** 2))

if __name__ == "__main__":
    unittest.main()