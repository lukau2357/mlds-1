import numpy as np
import cvxopt
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs

color_map = {1: "#F8766D", -1: "#00BFC4"}
cvxopt.solvers.options["show_progress"] = False

"""
TODO:
    - Add kernel support.
"""

class SVM_HardMargin:
    def __init__(self, sv_threshold = 1e-8):
        """
        It follows from Karush-Kuhn-Tucker conditions that for the corresponding Lagrange
        multiplier for a given observation is 0. However, due to numerical precision issues
        our solver returns small values that are close to 0. We set a threshold to differentiate
        between support vectors given a value of the Lagrange multiplier. 
        """
        self.sv_threshold = sv_threshold

    def fit(self, X, y):
        K = cvxopt.matrix(X.dot(X.T))
        P = cvxopt.matrix(K * np.outer(y, y))
        q = cvxopt.matrix(np.full(X.shape[0], -1).astype("float"))
        G = cvxopt.matrix(np.identity(X.shape[0]).astype("float") * (-1))
        A = cvxopt.matrix(y.reshape(1, X.shape[0]).astype("float"))
        h = cvxopt.matrix(np.zeros(X.shape[0]).astype("float"))
        b = cvxopt.matrix(0.0)

        solution = np.array(cvxopt.solvers.qp(P, q, G, h, A, b)["x"]).flatten()
        self.sv = solution > self.sv_threshold
        self.w = np.sum(X[self.sv].T * (solution[self.sv]* y[self.sv]), axis = 1)
        self.w0 = (y[self.sv].astype("float") - X[self.sv].dot(self.w)).mean()

class SVM_SoftMargin():
    def __init__(self, sv_threshold = 1e-5, C = 1, C_error = 1e-9):
        """
        C - regularization hyperparameter for soft margin SVM
        """
        self.sv_threshold = sv_threshold
        self.C = C
        self.C_error = C_error

    def fit(self, X, y):
        K = cvxopt.matrix(X.dot(X.T))
        P = cvxopt.matrix(K * np.outer(y, y))
        q = cvxopt.matrix(np.full(X.shape[0], -1).astype("float"))
        A = cvxopt.matrix(y.reshape(1, X.shape[0]).astype("float"))
        b = cvxopt.matrix(0.0)

        G_1 = (np.identity(X.shape[0]) * (-1))
        G_2 = np.identity(X.shape[0])
        h1 = np.zeros(X.shape[0])
        h2 = np.full(X.shape[0], self.C)

        G = cvxopt.matrix(np.vstack((G_1, G_2)).astype("float"))
        h = cvxopt.matrix(np.hstack((h1, h2)).astype("float"))

        solution = np.array(cvxopt.solvers.qp(P, q, G, h, A, b)["x"]).flatten()
        self.sv = solution > self.sv_threshold

        self.w = np.sum((X[self.sv].T * (solution[self.sv] * y[self.sv])), axis = 1)
        # Slightly different that the hard margin case, for the intercept we average over 
        # support vectors for which \xi_{i} = 0.
        b = self.sv & (self.C - solution > self.C_error)
        self.w0 = (y[b].astype("float") - X[b].dot(self.w)).mean()

def generate_dataset(n_samples = 200, std = 1):
    X, y =  make_blobs(n_samples = n_samples, centers = 2, cluster_std = std, random_state = 3)
    y[y == 0] = -1
    return X, y

def get_y(model, x, c = 0):
    # c = 0 corresponds to decision boundary
    # c = 1 corresponds to positive margin
    # c = -1 corresponds to negative margin
    return -(model.w0 + model.w[0] * x - c) / model.w[1]

def hard_margin_demo():
    X, y = generate_dataset()
    model = SVM_HardMargin()
    model.fit(X, y)

    plt.style.use("ggplot")
    fig, ax = plt.subplots()
    pos_class = y == 1
    neg_class = y == -1

    ax.scatter(X[pos_class, 0], X[pos_class, 1], c = color_map[1])
    ax.scatter(X[neg_class, 0], X[neg_class, 1], c = color_map[-1])

    x_min = X[:, 0].min()
    x_max = X[:, 0].max()

    xrange = np.arange(x_min, x_max, step = 0.1)
    db_y = [get_y(model, x) for x in xrange]
    pos_margin_y = [get_y(model, x, c = 1) for x in xrange]
    neg_margin_y = [get_y(model, x, c = -1) for x in xrange]
    
    ax.plot(xrange, db_y)
    ax.plot(xrange, pos_margin_y, linestyle = "--")
    ax.plot(xrange, neg_margin_y, linestyle = "--")
    ax.set_title("Hard margin SVM - linearly separable case")
    plt.show()

def soft_margin_demo(C_values):
    # C_values - different regularization weights for SVM soft margin models
    # For the purposes of this demonstration we consider 4 different C values,
    # and hence supblots is 2x2

    # 1.5 std within a cluster to achieve linear inseparability
    X, y = generate_dataset(std = 1.5)

    plt.style.use("ggplot")
    fig, ax = plt.subplots(nrows = 2, ncols = 2)
    fig.suptitle("Soft margin SVM demonstration")

    pos_class = y == 1
    neg_class = y == -1
    x_min = X[:, 0].min()
    x_max = X[:, 0].max()
    xrange = np.arange(x_min, x_max, step = 0.1)

    for i, C in enumerate(C_values):
        model = SVM_SoftMargin(C = C)
        model.fit(X, y)

        row = 0 if i < 2 else 1
        col = i % 2

        ax[row, col].scatter(X[pos_class, 0], X[pos_class, 1], c = color_map[1])
        ax[row, col].scatter(X[neg_class, 0], X[neg_class, 1], c = color_map[-1])

        db_y = [get_y(model, x) for x in xrange]
        pos_margin_y = [get_y(model, x, c = 1) for x in xrange]
        neg_margin_y = [get_y(model, x, c = -1) for x in xrange]

        ax[row, col].plot(xrange, db_y)
        ax[row, col].plot(xrange, pos_margin_y, linestyle = "--")
        ax[row, col].plot(xrange, neg_margin_y, linestyle = "--")
        ax[row, col].set_title("C = {:.2f}".format(C))
    plt.show()

if __name__ == "__main__":
    # hard_margin_demo()
    soft_margin_demo([0.01, 0.1, 1, 10])