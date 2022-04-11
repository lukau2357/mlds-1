import numpy as np
import cvxopt
import matplotlib.pyplot as plt

from sklearn import datasets

color_map = {1: "#F8766D", -1: "#00BFC4"}
cvxopt.solvers.options["show_progress"] = False

# Kernel support added

class Linear:
    """
    Linear kernel.
    """
    def __init__(self):
        pass

    def __call__(self, A, B):
        return A.dot(B.T)

class Polynomial:
    """
    Polynomial kernel
    """
    def __init__(self, M = 1):
        self.M = M
    
    def __call__(self, A, B):
        return (1 + A.dot(B.T)) ** self.M

class RBF:
    """
    RBF kernel
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

class SVM_HardMargin:
    def __init__(self, sv_threshold = 1e-8, kernel = Linear()):
        """
        It follows from Karush-Kuhn-Tucker conditions that for the corresponding Lagrange
        multiplier for a given observation is 0. However, due to numerical precision issues
        our solver returns small values that are close to 0. We set a threshold to differentiate
        between support vectors given a value of the Lagrange multiplier. 
        """
        self.sv_threshold = sv_threshold
        self.kernel = kernel

    def fit(self, X, y):
        kernel_matrix = self.kernel(X, X)
        self.X_train = X
        self.y_train = y

        K = cvxopt.matrix(kernel_matrix)
        P = cvxopt.matrix(K * np.outer(y, y))
        q = cvxopt.matrix(np.full(X.shape[0], -1).astype("float"))
        G = cvxopt.matrix(np.identity(X.shape[0]).astype("float") * (-1))
        A = cvxopt.matrix(y.reshape(1, X.shape[0]).astype("float"))
        h = cvxopt.matrix(np.zeros(X.shape[0]).astype("float"))
        b = cvxopt.matrix(0.0)

        self.alpha = np.array(cvxopt.solvers.qp(P, q, G, h, A, b)["x"]).flatten()
        self.sv = self.alpha > self.sv_threshold

        if isinstance(self.kernel, Linear):
            self.w = (X[self.sv].T.dot(self.alpha[self.sv] * y[self.sv]))
        
        self.w0 = (y[self.sv].astype("float") - self.kernel(X[self.sv], X).dot(self.alpha * y)).mean()

    def predict(self, X):
        return np.sign(self.kernel(X, self.X_train).dot(self.alpha * self.y_train) + self.w0)

class SVM_SoftMargin():
    def __init__(self, C = 1, error = 1e-8, kernel = Linear()):
        """
        C - regularization hyperparameter for soft margin SVM. Error term kept for the same
        reasons as with hard margin case.
        """
        self.C = C
        self.error = error
        self.kernel = kernel

    def fit(self, X, y):
        kernel_matrix = self.kernel(X, X)
        self.X_train = X
        self.y_train = y

        K = cvxopt.matrix(kernel_matrix)
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

        self.alpha = np.array(cvxopt.solvers.qp(P, q, G, h, A, b)["x"]).flatten()
        self.sv = self.alpha > self.error

        if isinstance(self.kernel, Linear):
            self.w = (X[self.sv].T.dot(self.alpha[self.sv] * y[self.sv]))

        b = self.sv & (self.alpha < self.C - self.error)
        self.w0 = (y[b].astype("float") - self.kernel(X[b], X).dot(self.alpha * y)).mean()

    def predict(self, X):
        return np.sign(self.kernel(X, self.X_train).dot(self.alpha * self.y_train) + self.w0)

def generate_dataset(n_samples = 200, std = 0.5):
    X, y =  datasets.make_blobs(n_samples = n_samples, centers = 2, cluster_std = std, random_state = 3)
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