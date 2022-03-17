import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import random

from scipy.optimize import fmin_l_bfgs_b
from scipy.special import softmax
from sklearn.model_selection import train_test_split
from numpy.linalg import norm

MBOG_TRAIN = 100

def preprocessing(hold_columns, dummy_columns, y_encoding, standardize = True):
    df = pd.read_csv("./dataset.csv", sep = ";")
    # Data standardization since regression is sensitive to large values
    dummy_encoded = pd.get_dummies(df[dummy_columns], drop_first = True)

    df = df[hold_columns].copy()
    if standardize:
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

    N = X.shape[0]
    res = 0

    # We make use of fast matrix multiplication in numpy to compute the log-likelihood as fast as possible
    softmax_matrix = np.matmul(parameter_matrix[:, 1:], X.T)
    for i in range(m):
        softmax_matrix[i] += parameter_matrix[i, 0]
    
    softmax_matrix = np.vstack((softmax_matrix, np.zeros(softmax_matrix.shape[1])))
    softmax_matrix = softmax(softmax_matrix, axis = 0)

    for i in range(X.shape[0]):
        res += np.log(softmax_matrix[y[i], i])

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
    
    def build(self, X, y):
        # Number of target categories
        m = max(y)
        N = X.shape[1]

        self.Beta = np.ones((m, N + 1))

        opt, _, __ = fmin_l_bfgs_b(softmax_reg_log_likelihood, self.Beta.flatten(), 
            args = (X, y), approx_grad = True)

        self.Beta = opt.reshape(m, N + 1)
                
        # Since unit tests are made to accept an object, we return the object itself
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

    def build(self, X, y, eps = 1e-9):
        """
        The assumption is that numerical labels of y follow the ordering of levels of y. A model would still
        be trainable if that was not the case, however one would heavily lose in interpretability and surely
        the results would not be natural.

        eps - initial value for thresholds (we are using the 'stick-breaking' parametrization of thresholds)
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
            bounds = constraint_beta + constraint_delta)
        
        # Extract the optimal values from LBFGS
        self.Beta = opt[:(N + 1)]
        self.delta = opt[(N + 1):]

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

def naive_evaluation(X, y, model):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 41)

    start = time.time()

    model.build(X_train, y_train)

    end = time.time()

    print("Time required for training the model: {:.2f}".format(end - start))

    predictions = model.predict(X_test)
    argmax_pred = []

    for i in range(predictions.shape[0]):
        argmax_pred.append(np.argmax(predictions[i]))
    
    print(sum([argmax_pred[i] == y_test[i] for i in range(len(y_test))]) / len(argmax_pred))

def coefficient_interpretation(X, y, boot_samples, x_labels, y_encoding, seed = 41):
    N = X.shape[1]
    rnd = np.random.RandomState(seed)
    n = X.shape[0]
    m = np.max(y)

    weights = {(i, j): [] for i in range(max(y)) for j in range(N + 1)}

    for k in range(boot_samples):
        sample = rnd.choice(list(range(n)), size = n, replace = True)

        X_boot = X[sample, :].copy()
        y_boot = y[sample].copy()

        while len(np.unique(y_boot)) < 6:
            print("Non-saturated bootstrap sample ocurred.")
            sample = rnd.choice(list(range(n)), size = n, replace = True)
            X_boot = X[sample, :].copy()
            y_boot = y[sample].copy()

        # Standardize numerical columns
        X_boot[:, 2] = (X_boot[:, 2] - np.mean(X_boot[:, 2])) / np.std(X_boot[:, 2])
        X_boot[:, 3] = (X_boot[:, 3] - np.mean(X_boot[:, 3])) / np.std(X_boot[:, 3])

        start = time.time()
        model = MultinomialLogReg()
        print("Started training model {:d}".format(k))
        model.build(X_boot, y_boot)
        end = time.time()
        print("Finished training model {:d} in {:.2f} seconds".format(k, end - start))

        for i in range(m):
            for j in range(N + 1):
                weights[(i, j)].append(model.Beta[i, j])

    plt.style.use("ggplot")

    for i in range(m):
        fig, ax = plt.subplots()
        data = [np.mean(weights[(i, j)]) for j in range(N + 1)]
        pos = np.arange(N + 1) + 1
        ax.barh(pos, data, linewidth = 0.5, edgecolor = "black")

        # Bootstrap-percentile confidence intervals
        for j in range(N + 1):
            left = np.quantile(weights[(i, j)], 0.025)
            right = np.quantile(weights[(i, j)], 0.975)
            ax.plot((left, right), (j + 1, j + 1), color = "black", linewidth = 2)

        ax.axvline(0, color = "black", linewidth = 3, linestyle = "--")
        ax.set_title(y_encoding[i])
        ax.set_yticks(pos)
        ax.set_yticklabels(x_labels)
        ax.set_xlabel("Mean after bootstrap")

    plt.show()

def multinomial_bad_ordinal_good(size, rand):
    l = int(np.math.floor(size / 4))
    diff = size - 4 * l

    c0 = np.array([[rand.normalvariate(0, 1), 0] for i in range(l)])
    c1 = np.array([[rand.normalvariate(1, 1), 1] for i in range(l)])
    c2 = np.array([[rand.normalvariate(5, 1), 2] for i in range(l)])
    c3 = np.array([[rand.normalvariate(12, 1), 3] for i in range(l + diff)])

    res = np.concatenate((c0, c1, c2, c3))
    X = res[:, :1]
    y = res[:, 1].astype(int)

    return X, y

def accuracy(X, y, model):
    preds = model.predict(X)
    preds = [np.argmax(preds[i]) for i in range(preds.shape[0])]
    return sum([y[i] == preds[i] for i in range(y.shape[0])]) / len(preds)

def log_loss(X, y, model):
    preds = model.predict(X)
    res = 0
    for i in range(preds.shape[0]):
        res += np.log(preds[i, y[i]])
    
    return -(1 / preds.shape[0] * res)

def ml_vs_ol_demo():
    rnd = random.Random(41)
    X, y = multinomial_bad_ordinal_good(MBOG_TRAIN + 1000, rnd)
    
    ml = MultinomialLogReg()
    ol = OrdinalLogReg()

    s = rnd.sample(list(range(X.shape[0])), MBOG_TRAIN)

    while len(np.unique(y[s])) < 3:
        s = rnd.sample(list(range(X.shape[0])), MBOG_TRAIN)

    X_train, y_train = X[s], y[s]
    mask = [True if i not in s else False for i in range(X.shape[0])]
    X_test, y_test = X[mask], y[mask]

    start_ml = time.time()
    print("Started training ML")
    ml.build(X_train, y_train)
    print("Finished training ML in: {:f}".format(time.time() - start_ml))
    print("Started training OL")
    start_ol = time.time()
    ol.build(X_train, y_train)
    print("Finished training OL in: {:f}".format(time.time() - start_ol))

    ml_ac = accuracy(X_test, y_test, ml)
    ml_ll = log_loss(X_test, y_test, ml)
    ol_ac = accuracy(X_test, y_test, ol)
    ol_ll = log_loss(X_test, y_test, ol)

    plt.style.use("ggplot")
    fig, ax = plt.subplots()
    X = np.arange(2)
    ax.bar(X - 0.2, [ml_ac, ol_ac], color = ["#ec7063"], label = "Accuracy", width = 0.4)
    ax.bar(X + 0.2, [ml_ll, ol_ll], color = ["#82e0aa"], label = "Mean log-loss", width = 0.4)
    ax.set_title("Model evaluation")
    ax.set_xticks([0, 1], ["Multinomial Regression", "Ordinal Regression"])
    ax.legend()
    plt.show()

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

    y_encoding_reversed = {
        0: "Above Head",
        1: "Layup",
        2: "Other",
        3: "Hook shot",
        4: "Dunk",
        5: "Tip-in"
    }

    ml_vs_ol_demo()