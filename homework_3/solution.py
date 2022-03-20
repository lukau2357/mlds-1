import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from numpy.linalg import inv, norm
from numpy.random import default_rng
from scipy.optimize import minimize
from sklearn.model_selection import LeaveOneOut

from sklearn.linear_model import Ridge

class RidgeReg:
    def __init__(self, l):
        self.l = l
        
    def fit(self, X, y):
        """
        Closed form solution for ridge regression, given l = lambda > 0. The commented
        part represents a solution without the intercept term, assuming X and y are standardized
        (or centered at the very least).
        """
        self.intercept = y.mean()
        self.beta = inv(X.T.dot(X) + self.l * np.identity(X.shape[1])).dot(X.T).dot(y - self.intercept)

        '''
        alternative implementation of Ridge regression, not using the mean as intercept
        no assumption about X being centered/standardized.
        X = np.c_[np.ones(X.shape[0]), X]
        t = self.l * np.identity(X.shape[1])
        t[0, 0] = 0
        self.beta = inv(X.T.dot(X) + t).dot(X.T).dot(y)
        '''

    def predict(self, X):
        return np.dot(X, self.beta) + self.intercept
        # return np.dot(X, self.beta[1:]) + self.beta[0] # for alternative implementation

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
        interpretation of L1 regularization. Seed is just a number that is used for 
        reproducibility.
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
        X = np.arange(0, 100, 5, dtype = "float64")
        X += np.random.normal(X.shape[0])
        X = X.reshape(X.shape[0], 1)

        y = 10 + 2 * X[:,0]

        X = X - X.mean()

        model = RidgeReg(1)
        model.fit(X, y)
        y_pred = model.predict(X)

        np.testing.assert_array_almost_equal(y_pred, y, decimal = 2)

    def test_lasso_simple(self):
        X = np.arange(0, 100, 5, dtype = "float64")
        X += np.random.normal(X.shape[0])
        X = X.reshape(X.shape[0], 1)

        y = 10 + 2 * X[:,0]

        model = LassoReg(1)
        model.fit(X, y)
        y_pred = model.predict(X)

        np.testing.assert_array_almost_equal(y_pred, y, decimal = 2)

def load(fname):
    df = pd.read_csv(fname)
    target = "critical_temp"
    
    X = df.loc[:, df.columns != target].to_numpy()
    y = df[target].to_numpy()

    X_train, y_train = X[:200, :], y[:200]
    X_test, y_test = X[200:, :], y[200:]

    return df.columns, X_train, y_train, X_test, y_test

def superconductor(X_train, y_train, X_test, y_test):
    rmse_loo = []
    rmse_test = []

    mu_x, std_x = X_train.mean(), X_train.std()
    # mu_y, std_y = y_train.mean(), y_train.std()

    X_train = (X_train - mu_x) / std_x
    # y_train = (y_train - mu_y) / std_y

    # Standardize the test data using Mu and Sigma of the training data!
    X_test = (X_test - mu_x) / std_x
    # y_test = (y_test - mu_y) / std_y

    plt.style.use("ggplot")
    fig, ax = plt.subplots()

    reg_weights = np.arange(0.01, 2, 0.01)

    for w in reg_weights:
        model = RidgeReg(w)
        rmse_loo.append(loocv_score(model, X_train, y_train))
        # Fit the model using the full training set to compute RMSE on the test set
        model.fit(X_train, y_train)
        rmse_test.append(rmse(model, X_train, y_train))

    ax.plot(reg_weights, rmse_loo, label = "Mean RMSE after LOOCV")
    ax.plot(reg_weights, rmse_test, label = "RMSE on the test set")
    ax.axvline(reg_weights[np.argmin(rmse_loo)], linestyle = "--")
    ax.set_xlabel("Regularization weight")
    ax.set_ylabel("Model score")
    ax.legend()
    plt.show()

    print("Minimum LOOCV score {:f} obtained for regularization weight of {:f}".format(min(rmse_loo), reg_weights[np.argmin(rmse_loo)]))
    print("Test set RMSE for this regularization weight: {:f}".format(rmse_test[np.argmin(rmse_loo)]))

def rmse(model, X, y, mu_y = None, std_y = None):
    preds = model.predict(X)

    if mu_y is not None:
        preds = preds * std_y + mu_y
        y = y * std_y + mu_y
    
    N = X.shape[0]
    return np.math.sqrt((1 / N) * np.sum((preds - y) ** 2))

def loocv_score(model, X_train, y_train, mu_y = None, std_y = None):
    """
    LOOCV score for the given model, using RMSE as the evaluation metric.
    """
    scores = []

    loo = LeaveOneOut()
    for train_index, test_index in loo.split(X_train):
        model.fit(X_train[train_index], y_train[train_index])
        scores.append(rmse(model, X_train[test_index], y_train[test_index]))
    
    scores = np.array([scores])
    return scores.mean()

if __name__ == "__main__":
    # superconductor(X_train, y_train, X_test, y_test)
    unittest.main()