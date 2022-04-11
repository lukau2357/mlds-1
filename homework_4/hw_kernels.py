import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cvxopt

from sklearn.model_selection import KFold

"""
TODO:
    - Quantify uncertainty of RMSE?
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
        self.X_observed = None

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
    def __init__(self, kernel = Linear(), lambda_ = 0.01, epsilon = 0.01, error = 1e-6):
        """
        CVXOPT approximates solution for the dual SVR quadratic program, that is why
        we need an error threshold for our results in the form of error variable
        """
        self.kernel = kernel
        self.lambda_ = lambda_ # regularization parameter
        self.epsilon = epsilon # Vapnik loss parameter
        self.error = error
    
    def get_alpha(self):
        return self.alpha_

    def get_b(self):
        return self.b

    def fit(self, X, y):
        n = X.shape[0]
        even_ind_mask = np.array([i % 2 == 0 for i in range(2 * n)])
        odd_ind_mask = ~even_ind_mask
        kernel_matrix = self.kernel(X, X)

        self.X_obs = X
        c = np.tile(np.array([[1, -1], [-1, 1]]), kernel_matrix.shape)
        P = cvxopt.matrix(np.repeat(np.repeat(kernel_matrix, 2, 0), 2, 1) * c)

        '''
        Old, naive way of computing the quadratic form of interest!
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
        P = cvxopt.matrix((X1 - X2 - X3 + X4).astype("float"))
        '''

        # Linear term
        q1 = np.full(2 * n, self.epsilon)
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
        cvxopt.solvers.options["show_progress"] = False
        solution = np.array(cvxopt.solvers.qp(P, q, G, h, A, b)["x"]).flatten()

        # Alpha vector to satisfy test cases
        self.alpha_ = solution.reshape(X.shape[0], 2)

        # Dual variables used for prediction and intercept computation
        self.alpha = solution[::2]
        self.alpha_star = solution[1::2]

        # Computing the intercept
        min_ind = (self.alpha_star > self.error) | (self.alpha < 1 / self.lambda_ - self.error)
        max_ind = (self.alpha > self.error) | (self.alpha_star < 1 / self.lambda_ - self.error)
        W = kernel_matrix.dot(self.alpha - self.alpha_star)
        b_l =  max(-self.epsilon + y[min_ind] - W[min_ind])
        b_u = min(self.epsilon + y[max_ind] - W[max_ind])
        self.b = (b_l + b_u) / 2

        self.support_vectors = np.abs(self.alpha - self.alpha_star) > self.error
        return self

    def predict(self, X):
        return (self.kernel(X, self.X_obs)).dot(self.alpha - self.alpha_star) + self.b
        
def read_sine():
    df = pd.read_csv("./sine.csv")
    X = df["x"].to_numpy().reshape(df.shape[0], 1)
    y = df["y"].to_numpy()

    x_mu, x_std = X.mean(), X.std()
    X = (X - x_mu) / x_std

    return X, y, x_mu, x_std

def sine_test_poly(M_values, colors):
    """
    Kernelized ridge regression with polynomial kernel on the sine dataset.
    """
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
    """
    Kernelized ridge regression with RBF kernel on the sine dataset.
    """
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
    """
    Support vector regression with polynomial kernel on the sine dataset.
    """
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
    """
    Support vector regression with RBF kernel on the sine dataset.
    """
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

def read_house():
    df = pd.read_csv("./housing2r.csv", index_col = None)
    # Randomly shuffle the dataset
    df = df.sample(frac = 1, random_state = 42)
    X = df.loc[:, df.columns != "y"].to_numpy()
    y = df["y"].to_numpy()

    X_mu, X_std = X.mean(), X.std()
    X = (X - X_mu) / X_std

    y_mu, y_std = y.mean(), y.std()
    # y = (y - y_mu) / y_std

    return X, y, X_mu, X_std, y_mu, y_std

def rmse(p, y, y_mu, y_std, rescale = False):
    if rescale:
        p = p * y_std + y_mu
        y = y * y_std + y_mu
    
    return np.sqrt(((p - y) ** 2).sum() / y.shape[0])

def cv_krr_lambda(cv, kernel, lambdas, X_train, y_train, y_mu, y_std):
    """
    Cross-validation for kernelized ridge regression
    """
    rmse_l = np.zeros(len(lambdas))
    kfold = KFold(n_splits = cv)

    for i, l in enumerate(lambdas):
        preds = np.zeros(X_train.shape[0])

        for train, test in kfold.split(X_train):
            current_model = KernelizedRidgeRegression(kernel, l)
            current_model.fit(X_train[train], y_train[train])
            preds[test] = current_model.predict(X_train[test])

        rmse_l[i] = rmse(preds, y_train, y_mu, y_std)
    return lambdas[np.argmin(rmse_l)]

def housing_krr_poly(lambdas, M_values = (list(range(1, 11)))):
    """
    Hyperparameter tuning for kernelized ridge regression with polynomial kernel
    on the housing dataset.
    """
    X, y, X_mu, X_std, y_mu, y_std = read_house()
    n_threshold = int(X.shape[0] * 0.8)
    X_train, y_train = X[:n_threshold], y[:n_threshold]
    X_test, y_test = X[n_threshold:], y[n_threshold:]

    rmse_1, rmse_best, lambda_choices = [], [], []

    for m in M_values:
        kernel = Polynomial(M = m)
        model_1 = KernelizedRidgeRegression(kernel, 1)
        model_1.fit(X_train, y_train)
        preds_1 = model_1.predict(X_test)
        rmse_1.append(rmse(preds_1, y_test, y_mu, y_std))

        best_lambda = cv_krr_lambda(10, kernel, lambdas, X_train, y_train, y_mu, y_std)
        lambda_choices.append(best_lambda)
        model_2 = KernelizedRidgeRegression(kernel, best_lambda)
        model_2.fit(X_train, y_train)
        preds_2 = model_2.predict(X_test)
        rmse_best.append(rmse(preds_2, y_test, y_mu, y_std))

    plt.style.use("ggplot")
    fix, ax = plt.subplots()
    ax.plot(M_values, rmse_1, label = r"$\lambda = 1$")
    ax.plot(M_values, rmse_best, label = r"CV $\lambda$")
    ax.set_xticks(M_values)
    ax.set_xlabel("Degree")
    ax.set_ylabel("RMSE on validation set")
    ax.legend()
    ax.set_title("Housing dataset - Kernelized Ridge Regression with Polynomial kernel")
    plt.show()

    print("Best parameter for lambda = 1 and RMSE: {} {}".format(M_values[np.argmin(rmse_1)], min(rmse_1)))
    print("Best CV lambda, parameter and RMSE: {} {} {}".format(lambda_choices[np.argmin(rmse_best)], M_values[np.argmin(rmse_best)], min(rmse_best)))

def housing_krr_rbf(lambdas, sigma_values):
    """
    Hyperparameter tuning for kernelized ridge regression with RBF kernel
    on the housing dataset.
    """
    X, y, X_mu, X_std, y_mu, y_std = read_house()
    n_threshold = int(X.shape[0] * 0.8)
    X_train, y_train = X[:n_threshold], y[:n_threshold]
    X_test, y_test = X[n_threshold:], y[n_threshold:]

    rmse_1, rmse_best, lambda_choices = [], [], []

    for sigma in sigma_values:
        kernel = RBF(sigma = sigma)
        model_1 = KernelizedRidgeRegression(kernel = kernel, lambda_ = 1)
        model_1.fit(X_train, y_train)
        preds_1 = model_1.predict(X_test)
        rmse_1.append(rmse(preds_1, y_test, y_mu, y_std))

        best_lambda = cv_krr_lambda(10, kernel, lambdas, X_train, y_train, y_mu, y_std)
        lambda_choices.append(best_lambda)
        model_2 = KernelizedRidgeRegression(kernel = kernel, lambda_ = best_lambda)
        model_2.fit(X_train, y_train)
        preds_2 = model_2.predict(X_test)
        rmse_best.append(rmse(preds_2, y_test, y_mu, y_std))

    plt.style.use("ggplot")
    fix, ax = plt.subplots()
    ax.plot(sigma_values, rmse_1, label = r"$\lambda = 1$")
    ax.plot(sigma_values, rmse_best, label = r"CV $\lambda$")
    # ax.set_xticks(sigma_values)
    ax.set_xlabel(r"$\sigma$")
    ax.set_ylabel("RMSE on validation set")
    ax.legend()
    ax.set_title("Housing dataset - Kernelized Ridge Regression with RBF kernel")
    plt.show()

    print("Best parameter for lambda = 1 and RMSE: {} {}".format(sigma_values[np.argmin(rmse_1)], min(rmse_1)))
    print("Best CV lambda, parameter and RMSE: {} {} {}".format(lambda_choices[np.argmin(rmse_best)], sigma_values[np.argmin(rmse_best)], min(rmse_best)))

def cv_svr_lambda(cv, kernel, lambdas, X_train, y_train, y_mu, y_std, epsilon):
    """
    Cross validation for support vector regression
    """
    rmse_l = np.zeros(len(lambdas))
    kfold = KFold(n_splits = cv)

    for i, l in enumerate(lambdas):
        print("Current lambda: {}".format(l))
        preds = np.zeros(X_train.shape[0])

        for train, test in kfold.split(X_train):
            current_model = SVR(kernel = kernel, lambda_ = l, epsilon = epsilon)
            current_model.fit(X_train[train], y_train[train])
            preds[test] = current_model.predict(X_train[test])

        rmse_l[i] = rmse(preds, y_train, y_mu, y_std)

    return lambdas[np.argmin(rmse_l)]

def housing_svr_poly(lambdas, M_values = list(range(1, 11)), epsilon = 5):
    """
    Hyperparameter tuning for support vector regression with polynomial kernel
    on the housing dataset.
    """
    X, y, X_mu, X_std, y_mu, y_std = read_house()
    n_threshold = int(X.shape[0] * 0.8)
    X_train, y_train = X[:n_threshold], y[:n_threshold]
    X_test, y_test = X[n_threshold:], y[n_threshold:]

    rmse_1, rmse_best, sv_1, sv_best, lambda_choices = [], [], [], [], []

    for m in M_values:
        kernel = Polynomial(M = m)
        model_1 = SVR(kernel = kernel, lambda_ = 1, epsilon = epsilon)
        model_1.fit(X_train, y_train)
        preds_1 = model_1.predict(X_test)
        rmse_1.append(rmse(preds_1, y_test, y_mu, y_std))
        sv_1.append(model_1.support_vectors.sum() / X_train.shape[0])

        print("Cross validating for degree: {}".format(str(m)))
        best_lambda = cv_svr_lambda(10, kernel, lambdas, X_train, y_train, y_mu, y_std, epsilon)
        lambda_choices.append(best_lambda)
        model_2 = SVR(kernel = kernel, lambda_ = best_lambda, epsilon = epsilon)
        model_2.fit(X_train, y_train)
        preds_2 = model_2.predict(X_test)
        rmse_best.append(rmse(preds_2, y_test, y_mu, y_std))
        sv_best.append(model_2.support_vectors.sum() / X_train.shape[0])

    plt.style.use("ggplot")
    fig, ax = plt.subplots(nrows = 1, ncols = 2)
    ax[0].plot(M_values, rmse_1, label = r"$\lambda = 1$")
    ax[0].plot(M_values, rmse_best, label = r"CV $\lambda$")
    ax[0].set_xticks(M_values)
    ax[0].set_xlabel("M")
    ax[0].set_ylabel("RMSE on validation set")
    ax[0].legend()

    ax[1].plot(M_values, sv_1, label = r"$\lambda = 1$")
    ax[1].plot(M_values, sv_best, label = r"CV $\lambda$")
    ax[1].set_xticks(M_values)
    ax[1].set_xlabel("M")
    ax[1].set_ylabel("Proportion of support vectors")
    ax[1].legend()

    fig.suptitle("Housing dataset - Support Vector Regression with Polynomial kernel")
    plt.show()

    print("Best degree for lambda = 1 RMSE and proportion of support vectors: {} {} {}".format(M_values[np.argmin(rmse_1)], min(rmse_1), sv_1[np.argmin(rmse_1)]))
    print("Best CV lambda, degree RMSE and proportion of support vectors: {} {} {} {}".format(lambda_choices[np.argmin(rmse_best)], M_values[np.argmin(rmse_best)], min(rmse_best), sv_best[np.argmin(rmse_best)]))

def housing_svr_rbf(lambdas, sigmas, epsilon = 5):
    """
    Hyperparameter tuning for support vector regression with RBF kernel
    on the housing dataset.
    """
    X, y, X_mu, X_std, y_mu, y_std = read_house()
    n_threshold = int(X.shape[0] * 0.8)
    X_train, y_train = X[:n_threshold], y[:n_threshold]
    X_test, y_test = X[n_threshold:], y[n_threshold:]

    rmse_1, rmse_best, sv_1, sv_best, lambda_choices = [], [], [], [], []

    for s in sigmas:
        kernel = RBF(sigma = s)
        model_1 = SVR(kernel = kernel, lambda_ = 1, epsilon = epsilon)
        model_1.fit(X_train, y_train)
        preds_1 = model_1.predict(X_test)
        rmse_1.append(rmse(preds_1, y_test, y_mu, y_std))
        sv_1.append(model_1.support_vectors.sum() / X_train.shape[0])

        print("Cross validating for sigma: {}".format(str(s)))
        best_lambda = cv_svr_lambda(10, kernel, lambdas, X_train, y_train, y_mu, y_std, epsilon)
        lambda_choices.append(best_lambda)
        model_2 = SVR(kernel = kernel, lambda_ = best_lambda, epsilon = epsilon)
        model_2.fit(X_train, y_train)
        preds_2 = model_2.predict(X_test)
        rmse_best.append(rmse(preds_2, y_test, y_mu, y_std))
        sv_best.append(model_2.support_vectors.sum() / X_train.shape[0])

    plt.style.use("ggplot")
    fig, ax = plt.subplots(nrows = 1, ncols = 2)
    ax[0].plot(sigmas, rmse_1, label = r"$\lambda = 1$")
    ax[0].plot(sigmas, rmse_best, label = r"CV $\lambda$")
    ax[0].set_xlabel(r"$\sigma$")
    ax[0].set_ylabel("RMSE on validation set")
    ax[0].legend()

    ax[1].plot(sigmas, sv_1, label = r"$\lambda = 1$")
    ax[1].plot(sigmas, sv_best, label = r"CV $\lambda$")
    ax[1].set_xlabel(r"$\sigma$")
    ax[1].set_ylabel("Proportion of support vectors")
    ax[1].legend()

    fig.suptitle("Housing dataset - Support Vector Regression with RBF kernel")
    plt.show()

    print("Best sigma for lambda = 1 RMSE and proportion of support vectors: {} {} {}".format(sigmas[np.argmin(rmse_1)], min(rmse_1), sv_1[np.argmin(rmse_1)]))
    print("Best CV lambda, sigma RMSE and proportion of support vectors: {} {} {} {}".format(lambda_choices[np.argmin(rmse_best)], sigmas[np.argmin(rmse_best)], min(rmse_best), sv_best[np.argmin(rmse_best)]))

if __name__ == "__main__":
    # sine_test_poly([2, 3, 5, 10, 15, 20], ["#C77CFF", "#1536CA", "#15CA15", "#0CAFDF", "#AF1439", "#818890"])
    # sine_test_rbf([0.01, 0.1, 0.5, 1, 5], ["#C77CFF", "#1536CA", "#15CA15", "#0CAFDF", "#AF1439"])
    # sine_test_svr_poly([10, 20], ["#C77CFF", "#1536CA"])
    # sine_test_svr_rbf([0.1, 0.5], ["#C77CFF", "#1536CA"])
    lambdas = [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 100]
    sigmas = np.linspace(0.1, 10, num = 100)
    # housing_krr_poly(lambdas)
    # housing_krr_rbf(lambdas, sigmas)
    # housing_svr_poly(lambdas, epsilon = 10)
    housing_svr_rbf(lambdas, sigmas, epsilon = 10)