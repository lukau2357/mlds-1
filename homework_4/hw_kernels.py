import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cvxopt

from sklearn.kernel_ridge import KernelRidge

"""
TODO:
    - Housing dataset, grid search for best hyperparameters
    - Fully utilize numpy for quadratic form of SVR dual problem
"""

class Linear:
    """
    Linear kernel implementation
    """

    def __init__(self):
        pass

    def __call__(self, A, B):
        return A.dot(B.T)

class Polynomial:
    """
    Polynomial kernel implementation
    """
    def __init__(self, M = 1):
        self.M = M
    
    def __call__(self, A, B):
        return (1 + A.dot(B.T)) ** self.M

class RBF:
    """
    RBF kernel implementation
    """
    def __init__(self, sigma = 1):
        self.sigma = sigma
    
    def __call__(self, A, B):
        if len(A.shape) == 1:
            A = A.reshape(1, A.shape[0])
        
        if len(B.shape) == 1:
            B = B.reshape(1, B.shape[0])

        anorms = np.linalg.norm(A, axis = 1) ** 2
        bnorms = np.linalg.norm(B, axis = 1) ** 2
        gram = A.dot(B.T)
        res = anorms.reshape(gram.shape[0], 1) - 2 * gram + bnorms
        res = np.exp(-1 / (2 * (self.sigma ** 2)) * res)

        return res[0, 0] if (res.shape == (1, 1)) else res.flatten() if (res.shape[0] == 1 or res.shape[1] == 1) else res

class KernelizedRidgeRegression:
    def __init__(self, kernel, lambda_):
        self.kernel = kernel
        self.lambda_ = lambda_
        self.alpha = None

    def fit(self, X, y):
        self.alpha = np.linalg.inv(self.kernel(X, X) + np.identity(X.shape[0]) * self.lambda_).dot(y)
        self.X_observed = X.copy()
        return self

    def predict(self, X):
        return np.dot(self.kernel(X, self.X_observed), self.alpha)

class SVR:
    """
    Support vector regression implementation
    """
    def __init__(self, kernel, lambda_, eps, error = 1e-6):
        """
        CVXOPT approximates solution for the dual SVR quadratic program, that is why
        we need an error threshold for our results
        """
        self.kernel = kernel
        self.lambda_ = lambda_ # regularization parameter
        self.eps = eps # Vapnik loss parameter
        self.error = error
    
    def fit(self, X, y):
        n = X.shape[0]
        even_ind_mask = np.array([i % 2 == 0 for i in range(2 * n)])
        odd_ind_mask = ~even_ind_mask
        kernel_matrix = self.kernel(X, X)

        self.X_obs = X

        # TODO: Try to utilize numpy fully for this part!
        # Quadratic term
        X1 = np.zeros((n * 2, n * 2))
        for i in range(n):
            for j in range(n):
                X1[2 * i, 2 * j] = kernel_matrix[i, j]

        X2 = np.zeros((n * 2, n * 2))
        for i in range(n):
            for j in range(n):
                X2[2 * i, 2 * j + 1] = kernel_matrix[i, j]

        X3 = np.zeros((n * 2, n * 2))
        for i in range(n):
            for j in range(n):
                X3[2 * i + 1, 2 * j] = kernel_matrix[i, j]

        X4 = np.zeros((n * 2, n * 2))
        for i in range(n):
            for j in range(n):
                X4[2 * i + 1, 2 * j + 1] = kernel_matrix[i, j]

        # Quadratic form for the dual problem
        P2 = cvxopt.matrix((X1 - X2 - X3 + X4).astype("float"))

        # Linear term
        q1 = np.full(2 * n, self.eps)
        q2 = np.zeros(2 * n)
        q2[odd_ind_mask] = -y.copy()
        q2[even_ind_mask] = y.copy()
        q = cvxopt.matrix((q1 - q2).astype("float"))

        # Box constraints
        G1 = np.identity(2 * n) * (-1)
        G2 = np.identity(2 * n)
        h1 = np.zeros(2 * n)
        h2 = np.full(2 * n, 1 / self.lambda_)

        G = cvxopt.matrix(np.vstack((G1, G2)).astype("float"))
        h = cvxopt.matrix(np.hstack((h1, h2)).astype("float"))

        # Linear constraints
        A = np.ones(2 * n)
        A[odd_ind_mask] = -1
        A = cvxopt.matrix(A.reshape(1, A.shape[0]).astype("float"))
        b = cvxopt.matrix(0.0)

        # Turn off solver verbose mode
        # cvxopt.solvers.options["show_progress"] = False
        solution = np.array(cvxopt.solvers.qp(P2, q, G, h, A, b)["x"]).flatten()

        # Dual variables used for prediction and intercept computation
        self.alpha = solution[::2]
        self.alpha_star = solution[1::2]

        # Computing the intercept
        min_ind = (self.alpha > self.error) | (self.alpha_star < 1 / self.lambda_ - self.error)
        max_ind = (self.alpha_star > self.error) | (self.alpha < 1 / self.lambda_ - self.error)
        W = kernel_matrix.dot(self.alpha - self.alpha_star)

        b_l =  max(-self.eps + y[min_ind] - W[min_ind])
        b_u = min(self.eps + y[max_ind] - W[max_ind])
        self.b = (b_l + b_u) / 2

        self.support_vectors = np.abs(self.alpha - self.alpha_star) > self.error
    
    def predict(self, X):
        return (self.alpha - self.alpha_star).T.dot(self.kernel(X, self.X_obs)) + self.b
        
def read_sine():
    df = pd.read_csv("./sine.csv")
    X = df["x"].to_numpy().reshape(df.shape[0], 1)
    y = df["y"].to_numpy()

    x_mu, x_std = X.mean(), X.std()
    X = (X - x_mu) / x_std

    return X, y, x_mu, x_std

def sine_test_poly(M_values, colors):
    X, y, mu, std = read_sine()

    plt.style.use("ggplot")
    fig, ax = plt.subplots()
    ax.scatter(X * std + mu, y)

    X_f = X.copy().flatten()
    X_argsort = np.argsort(X_f)
    X_s = np.sort(X_f)

    for m, c in zip(M_values, colors):
        model = KernelizedRidgeRegression(Polynomial(M = m), 0.01)
        model.fit(X, y)
        preds = model.predict(X)
        preds = preds[X_argsort]
        ax.plot(X_s * std + mu, preds, label = "{}".format(str(m)), color = c)
    
    ax.set_title("Sine dataset - Kernelized Ridge Regression with Polynomial kernel")
    ax.legend(title = "M")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.show()

def sine_test_rbf(sigmas, colors):
    X, y, mu, std = read_sine()

    plt.style.use("ggplot")
    fig, ax = plt.subplots()
    ax.scatter(X * std + mu, y)

    X_f = X.copy().flatten()
    X_argsort = np.argsort(X_f)
    X_s = np.sort(X_f)

    for s, c in zip(sigmas, colors):
        model = KernelizedRidgeRegression(RBF(sigma = s), 0.01)
        model.fit(X, y)
        preds = model.predict(X)
        preds = preds[X_argsort]
        ax.plot(X_s * std + mu, preds, label = "{}".format(str(s)), color = c)
    
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Sine dataset - Kernelized Ridge Regression with RBF kernel")
    ax.legend(title = r"\sigma")
    plt.show()

def sine_test_svr_poly(M_values, colors):
    X, y, mu, std = read_sine()

    plt.style.use("ggplot")
    fig, ax = plt.subplots()
    ax.scatter(X * std + mu, y)

    X_f = X.copy().flatten()
    X_argsort = np.argsort(X_f)
    X_s = np.sort(X_f)

    for m, c in zip(M_values, colors):
        model = SVR(Polynomial(M = m), 0.01, 0.6)
        model.fit(X, y)
        preds = model.predict(X)
        preds = preds[X_argsort]
        ax.plot(X_s * std + mu, preds, label = "{}".format(str(m)), color = c)
        ax.scatter(X_f[model.support_vectors] * std + mu, y[model.support_vectors], label = "Support vectors".format(str(m)), color = c)
    
    ax.set_title("Sine dataset - Support Vector Regression with Polynomial kernel")
    ax.legend(title = "M")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.show()

    ax.scatter(X[model.support_vectors] * std + mu, y[model.support_vectors])
    plt.show()

def sine_test_svr_rbf(sigmas, colors):
    X, y, mu, std = read_sine()

    plt.style.use("ggplot")
    fig, ax = plt.subplots()
    ax.scatter(X * std + mu, y)

    X_f = X.copy().flatten()
    X_argsort = np.argsort(X_f)
    X_s = np.sort(X_f)

    for s, c in zip(sigmas, colors):
        model = SVR(RBF(sigma = s), 0.01, 0.6)
        model.fit(X, y)
        preds = model.predict(X)
        preds = preds[X_argsort]
        ax.plot(X_s * std + mu, preds, label = "{}".format(str(s)), color = c)
        ax.scatter(X_f[model.support_vectors] * std + mu, y[model.support_vectors], label = "Support vectors".format(str(s)), color = c)
    
    ax.set_title("Sine dataset - Support Vector Regression with RBF kernel")
    ax.legend(title = r"$\sigma$")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.show()

    ax.scatter(X[model.support_vectors] * std + mu, y[model.support_vectors])
    plt.show()

if __name__ == "__main__":
    # sine_test_poly([2, 3, 5, 10, 15, 20], ["#C77CFF", "#1536CA", "#15CA15", "#0CAFDF", "#AF1439", "#818890"])
    # sine_test_rbf([0.01, 0.1, 0.5, 1, 5], ["#C77CFF", "#1536CA", "#15CA15", "#0CAFDF", "#AF1439"])
    # sine_test_svr_poly([10, 20], ["#C77CFF", "#1536CA"])
    sine_test_svr_rbf([0.1, 0.5], ["#C77CFF", "#1536CA"])