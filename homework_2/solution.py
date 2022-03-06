import pandas as pd
import numpy as np

from scipy.optimize import fmin_l_bfgs_b
from scipy.special import softmax
from sklearn.model_selection import train_test_split

"""
TODO:
    - Interpret MLR coefficients. First quantify uncertainty of computed coefficients.
    One approach is to generate bootstrap confidence intervals (less difficult), and
    another approach is to use the asymptotic normality of MLE to construct CI using
    Fisher information (more difficult).
    - Compe up with an artificial dataset where OLR performs better and MLR
"""

def preprocessing(hold_columns, dummy_columns, y_encoding):
    df = pd.read_csv("./dataset.csv", sep = ";")
    # Data standardization since regression is sensitive to large values
    dummy_encoded = pd.get_dummies(df[dummy_columns], drop_first = True)

    df = df[hold_columns].copy()
    df[["Angle", "Distance"]] =  (df[["Angle", "Distance"]] - df[["Angle", "Distance"]].mean()) / df[["Angle", "Distance"]].std()

    for column in dummy_encoded.columns:
        df = df.join(dummy_encoded[column])
    
    df["ShotType"].replace(list(y_encoding.keys()), list(y_encoding.values()), inplace = True)
    X = df.loc[:, df.columns != "ShotType"].to_numpy()
    y = df["ShotType"].to_numpy()

    return X, y

def softmax_reg_log_likelihood(Beta, X, y):
    # Scipy optimization methods work with vectors, while we perform all our calculations
    # with weight matrix
    N = X.shape[1]
    m = np.max(y)

    parameter_matrix = Beta.reshape((m, N + 1))
    Beta_matrix = parameter_matrix[:, 1:]
    Bias_vector = parameter_matrix[:, 0]

    N = X.shape[0]
    res = 0

    for i in range(X.shape[0]):
        X_i = X[i, :]
        
        if y[i] == m:
            u = 1
        
        else:
            u = np.math.exp(np.dot(Beta_matrix[y[i]], X_i) + Bias_vector[y[i]])

        v = np.sum(np.exp(np.matmul(Beta_matrix, X_i) + Bias_vector))
        res += np.math.log(u / (v + 1))

    # When dealing with MLE we are maximizing log-likelihood, however scipy optimization
    # methods deal with minimization problems, so we invert the log-likelihood
    return -res

def inverse_logit(z):
    return 1 / (1 + np.exp(-z))

def ordinal_reg_log_likelihood(point, X, y):
    m = X.shape[1]
    # Extract model parameters
    bias = point[0]
    beta = point[1:(m + 1)]
    delta = point[(m + 1):]

    # Fully utilize numpy!
    t = np.hstack((np.array([-np.inf, 0]), np.cumsum(delta), np.array([np.inf])))
    u = np.dot(X, beta) + bias
    res_matrix = np.log(inverse_logit(t[y + 1] - u) - inverse_logit(t[y] - u))
    return -np.sum(res_matrix)

class MultinomialLogReg():
    # Empty constructor as demanded by homework instructions
    def __init__(self):
        pass
    
    def build(self, X, y, cache = False, label = "model_weights"):
        # Number of target categories
        m = max(y)
        N = X.shape[1]

        # Beta[i][0] - bias for i-th category
        # m - 1 rows because m-th category is taken as reference
        self.Beta = np.zeros((m, N + 1))
        opt, _, __ = fmin_l_bfgs_b(softmax_reg_log_likelihood, self.Beta.flatten(), 
            args = (X, y), approx_grad = True, iprint = 99)
        
        self.Beta = opt.reshape(m, N + 1)
        # Since optimization does take some time, we cache the weights locally
        if cache:
            np.save("{}.npy".format(label), self.Beta)

        # Since unit tests are made to accept an object, we return the entire model
        return self

    def predict(self, X):
        N = X.shape[0]
        m = self.Beta.shape[0] + 1
        preds = np.zeros((N, m))

        for i in range(N):
            t = np.dot(self.Beta[:, 1:], X[i, :]) + self.Beta[:, 0]
            t = np.append(t, 0)
            t = softmax(t)
            preds[i, :] = t.copy()
        
        return preds

class OrdinalLogReg:
    def __init__(self):
        pass

    def build(self, X, y, cache = False, label = "label", eps = 1e-9):
        """
        Important assumption: continuous columns of X are standardized, and values
        of y are ordered!
        """

        m = max(y) + 1
        N = X.shape[1]

        # Initial weight/delta values
        # Thresholds are derived from delta because optimization constraints are more manageable
        # We also have the intercept, hence the N + 1
        self.Beta = np.zeros(N + 1)
        self.delta = np.full(m - 2, 0.5)

        constraint_beta = [(None, None) for i in range(N + 1)]
        constraint_delta = [(eps, None) for i in range(m - 2)]
        x0 = np.hstack((self.Beta, self.delta))

        opt, _, __ = fmin_l_bfgs_b(
            ordinal_reg_log_likelihood, 
            x0, 
            args = (X, y), 
            approx_grad = True,
            iprint = 99, 
            bounds = constraint_beta + constraint_delta)
        
        # Extract the optimal values from LBFGS
        self.Beta = opt[:(N + 1)]
        self.delta = opt[(N + 1):]

        # Since optimization does take some time, we cache the weights locally
        if cache:
            np.save("{}.npy".format(label), self.Beta)

        # Since unit tests are made to accept an object, we return the entire model
        return self

    def predict(self, X):
        N = X.shape[0]
        M = self.delta.shape[0] + 2

        predictions = np.zeros((N, M))
        t = np.hstack((np.array([-np.inf, 0]), np.cumsum(self.delta), np.array([np.inf])))

        for i in range(N):
            X_i = X[i, :]
            u_i = np.dot(self.Beta[1:], X_i) + self.Beta[0]
            for j in range(M):
                predictions[i, j] = inverse_logit(t[j + 1] - u_i) - inverse_logit(t[j] - u_i)
        
        return predictions

def naive_evaluation(X, y, model, cache = False, label = "model_weights"):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

    model.build(X_train, y_train, cache = cache, label = label)
    predictions = model.predict(X_test)
    argmax_pred = []

    for i in range(predictions.shape[0]):
        argmax_pred.append(np.argmax(predictions[i]))
    
    print(sum([argmax_pred[i] == y_test[i] for i in range(len(y_test))]) / len(argmax_pred))

if __name__ == "__main__":
    hold_columns = ["ShotType", "Transition", "TwoLegged", "Angle", "Distance"]
    dummy_columns = ["PlayerType", "Movement", "Competition"]

    y_encoding = {
        "above head": 0,
        "layup": 1,
        "other": 2,
        "hook shot": 3,
        "dunk": 4,
        "tip-in": 5
    }

    X, y = preprocessing(hold_columns, dummy_columns, y_encoding)
    l = MultinomialLogReg()
    naive_evaluation(X, y, l, cache = True, label = "softmax_reg_weights_intercept")