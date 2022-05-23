import numpy as np
import pandas as pd
import sys
import time
import json
import matplotlib.pyplot as plt

from scipy.special import softmax
from scipy.optimize import fmin_l_bfgs_b
from enum import Enum
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression

# Utils
def relu(x):
    return np.maximum(0.0, x)

def drelu(x):
    res = np.zeros(x.shape)
    res[x > 0] = 1
    return res

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dsigmoid(x):
    t = sigmoid(x)
    return t * (1 - t)

# Losses for individual observations returneed as a corresponding vector
# Sum of losses of individual observations is computed by the Network class
def square_loss(y, yhat):
    return 1 / (2 * len(y)) * ((y - yhat) ** 2)

def dsquare_loss(y, yhat):
    return -1 / len(y) * (y - yhat)

def log_loss(y, yhat):
    yhat_s = softmax(yhat, axis = 1)
    row_ind = np.array(list(range(len(y))))
    return -1 / len(y) * np.log(yhat_s[row_ind, y])

def dlog_loss(y, yhat):
    yhat_s = softmax(yhat, axis = 1)
    row_ind = np.array(list(range(len(y))))
    yhat_s[row_ind, y] -= 1
    return yhat_s / len(y)

def cum_log_loss(y, preds, average = True, logged = False):
    row_ind = np.array(list(range(len(y))))
    factor = - 1 / len(y) if average else -1
    if not logged:
        return factor * np.log(preds[row_ind, y]).sum()
    else:
        return factor * preds[row_ind, y].sum()

class NetworkType(Enum):
    R = 1
    C = 2

class Network:
    def __init__(self, units, loss, dloss, net_type,
                lambda_ = 0.001, 
                activations = [], 
                dactivations = [],
                output_activation = lambda x : x,
                output_dactivation = lambda x : np.ones(x.shape),
                seed = 3):

        if not isinstance(net_type, NetworkType):
            raise Exception("Invalid network type!")

        self.units = units.copy()
        self.lambda_ = lambda_
        self.loss = loss
        self.dloss = dloss

        self.net_type = net_type
        self.activations = [relu for unit in units] if len(activations) == 0 else \
                            activations
        self.dactivations = [drelu for unit in units] if len(dactivations) == 0 else \
                             dactivations

        self.output_activation = output_activation
        self.output_dactivation = output_dactivation
        self.activations.append(output_activation)
        self.dactivations.append(output_dactivation)
        self.random = np.random.default_rng(seed)

    def init_layers(self, X, y):
        self.input_dim = X.shape[1]
        self.output_dim = 1 if self.net_type is NetworkType.R else max(y) + 1
        self.units.append(self.output_dim)
        self.tw, self.tb = 0, 0
        prev_dim = self.input_dim

        for unit in self.units:
            self.tw += prev_dim * unit
            self.tb += unit
            prev_dim = unit

        # Xavier weight initialization
        prev_dim, prev_W, prev_b = self.input_dim, 0, 0
        self.W = np.empty(self.tw)
        self.b = np.empty(self.tb)

        for unit in self.units:
            self.W[prev_W : unit * prev_dim + prev_W] = self.random.uniform(low = - 1 / np.sqrt(prev_dim),
            high = 1 / np.sqrt(prev_dim), size = (prev_dim * unit))
            self.b[prev_b : unit + prev_b] = self.random.uniform(low = -1 / np.sqrt(prev_dim),
            high = 1 / np.sqrt(prev_dim), size = (unit))

            prev_W += unit * prev_dim
            prev_b  += unit
            prev_dim = unit
        
    def update_weights(self, w):
        W = w[:self.tw]
        b = w[self.tw:]
        self.W = W.copy()
        self.b = b.copy()

    def forward(self, X, ret = False):
        A, Z = [], []
        a = X.T
        prev_W, prev_b, prev_dim = 0, 0, self.input_dim

        for unit, activation in zip(self.units, self.activations):
            W = self.W[prev_W : unit * prev_dim + prev_W].reshape(unit, prev_dim)
            b = self.b[prev_b : unit + prev_b]
            z = W.dot(a) + b[:, np.newaxis]
            a = activation(z)

            prev_W += unit * prev_dim
            prev_b += unit
            prev_dim = unit

            if ret:
                A.append(a.T)
                Z.append(z.T)
        
        if ret:
            return A, Z
        
        return a.T

    def backward(self, w, X, y):
        # Partial derivatives of the loss
        self.update_weights(w)
        # a - N x nl
        # z - N x nl
        # delta_l - nl x N
        # W_l - nl x n_{l-1}
        # true_delta - delta.sum(axis = 1)

        A, Z = self.forward(X, ret = True)

        delta_front = (self.dloss(y, A[-1]) * self.dactivations[-1](Z[-1])).T
        pW = np.zeros(self.tw)
        pb = np.zeros(self.tb)

        last_j_W = self.tw
        last_j_b = self.tb

        for i in range(len(A) - 2, -1, -1):
            dW = delta_front.dot(A[i])
            db = delta_front.sum(axis = 1)

            iW = last_j_W - dW.shape[0] * dW.shape[1]
            ib = last_j_b - db.shape[0]
            pW[iW : last_j_W] = dW.ravel() + self.lambda_ * self.W[iW : last_j_W]
            pb[ib : last_j_b] = db.ravel()
            
            front_W = self.W[iW : last_j_W].reshape(dW.shape[0], dW.shape[1])
            last_j_W = iW
            last_j_b = ib
            
            delta = (front_W.T.dot(delta_front)) * self.dactivations[i](Z[i]).T

            delta_front = delta

        dW = delta_front.dot(X)
        db = delta_front.sum(axis = 1)

        pW[0 : last_j_W] = dW.ravel() + self.lambda_ * self.W[0 : last_j_W]
        pb[0 : last_j_b] = db.ravel()

        return np.concatenate((pW, pb))

    def cum_loss(self, w, X, y):
        # L2 regularized loss
        self.update_weights(w)
        yhat = self.forward(X)
        return np.sum(self.loss(y, yhat)) + 0.5 * self.lambda_ * np.sum(self.W ** 2)

    def cum_loss_no_update(self, X, y, reg = True):
        yhat = self.forward(X)
        reg = 0.5 * self.lambda_ * np.sum(self.W ** 2) if reg else 0
        return np.sum(self.loss(y, yhat)) + reg

    def conform_test_cases(self):
        """
        After fitting the model, weight matrices need to be initalized using different indexing
        in order to pass test cases. Also, the last row of every weight matrix contains intercepts
        for that layer.
        """
        prev_W, prev_b, prev_dim = 0, 0, self.input_dim
        self.weights = []

        for unit in self.units:
            W = self.W[prev_W : unit * prev_dim + prev_W].reshape(unit, prev_dim).T
            b = self.b[prev_b : unit + prev_b]
            self.weights.append(np.vstack((W, b)))

            prev_W += unit * prev_dim
            prev_b += unit
            prev_dim = unit
    
    def fit(self, X, y):
        self.init_layers(X, y)
        # Reshape y for regression problems to make backpropagation work for single
        # neuron in the output layer!
        if self.net_type is NetworkType.R:
            y = y.reshape(X.shape[0], 1)
        w0 = np.concatenate((self.W, self.b))
        solution = fmin_l_bfgs_b(self.cum_loss, w0, args = (X, y), fprime = self.backward)
        w_star = solution[0]
        self.update_weights(w_star)
        # Return the transposed weight matrices wiht intercepts in the last row to conform the 
        # test cases.

        self.conform_test_cases()
        return self.weights

def verify_gradient(network, h = 1e-6, eps = 1e-4, N = 100, p = 7, c = 3, verbose = False):
    # Generate a random dataset to verify that backpropagation is correctly implemented
    X = network.random.normal(size = (N, p))

    if network.net_type is NetworkType.R:
        y = network.random.normal(size = (N, 1))
    
    else:
        y = network.random.choice(list(range(c)), size = N, replace = True)

    network.init_layers(X, y)
    w = network.backward(np.concatenate((network.W, network.b)), X, y)
    pW = w[:network.tw]
    pb = w[network.tw:]

    l = network.cum_loss_no_update(X, y)

    # Verify weights
    for i in range(network.tw):
        network.W[i] += h
        change_i = network.cum_loss_no_update(X, y)
        network.W[i] -= h

        numerical_pd = (change_i - l) / h
        diff = np.abs(pW[i] - numerical_pd)

        if (diff > eps) and verbose:
            print("Error bigger than eps encountered for weight {:d}".format(i))
            print("Absolute error: {}".format(str(diff)))
            print(numerical_pd, pW[i])
    
    # Verify biasees
    for i in range(network.tb):
        network.b[i] += h
        change_i = network.cum_loss_no_update(X, y)
        network.b[i] -= h

        numerical_pd = (change_i - l) / h
        diff = np.abs(pb[i] - numerical_pd)

        if (diff > eps) and verbose:
            print("Error bigger than eps encountered for bias {:d}".format(i))
            print("Absolute error: {}".format(str(diff)))
        

class ANNClassification():
    def __init__(self, units = [], lambda_ = 0.001, activations = [], dactivations = [], seed = 3):
        self.net = Network(units, log_loss, dlog_loss, NetworkType.C, lambda_ = lambda_,
                           activations = activations, dactivations = dactivations, seed = seed)

    def fit(self, X, y):
        self.w = self.net.fit(X, y)
        # Required by unit tests...
        return self

    def predict(self, X):
        return softmax(self.net.forward(X), axis = 1)

    def weights(self):
        return self.w

class ANNRegression():
    def __init__(self, units = [], lambda_ = 0.001, activations = [], dactivations = [], seed = 4):
        self.net = Network(units, square_loss, dsquare_loss, NetworkType.R, lambda_ = lambda_,
        activations = activations, dactivations = dactivations, seed = seed)

    def fit(self, X, y):
        self.w = self.net.fit(X, y)
        # Required by unit tests...
        return self

    def predict(self, X):
        return self.net.forward(X).ravel()

    def weights(self):
        return self.w

def housing_reg_load():
    df = pd.read_csv("housing2r.csv")
    # Randomly shuffle the dataset before any training is done
    df = df.sample(frac = 1, random_state = 42)
    X = df.loc[:, df.columns != "y"].to_numpy()
    y = df["y"].to_numpy()
    X = (X - X.mean(axis = 0)) / X.std(axis = 0)
    return X, y

def housing_class_load():
    df = pd.read_csv("housing3.csv")
    df = df.sample(frac = 1, random_state = 42)
    df["Class"] = df["Class"].apply(lambda x : 0 if x == "C1" else 1)
    X = df.loc[:, df.columns != "Class"].to_numpy()
    y = df["Class"].to_numpy()
    X = (X - X.mean(axis = 0)) / X.std(axis = 0)
    return X, y

def house_grid_search_reg(X, y, lambda_ = [0.01, 0.1, 0.5], 
                          layers = [[], [10], [50], [10, 20], [50, 20],
                                  [10, 20 ,30], [50, 20, 30]], k = 10):
    
    best_mse, best_l, best_u = None, None, None
    cv = KFold(n_splits = k)

    sys.path.append("..")
    from homework_4.hw_kernels import SVR, RBF

    for u in layers:
        for l in lambda_:
            s = 0
            print("Estimating decay {} and layers {}".format(str(l), str(u)))

            for train, test in cv.split(X, y):
                current_model = ANNRegression(units = u, lambda_ = l)
                current_model.fit(X[train, :], y[train])
                preds = current_model.predict(X[test])
                s += ((y[test] - preds) ** 2).sum()
        
            s /= X.shape[0]
            print("Estimated loss: {}".format(str(s)))

            if best_mse is None or s < best_mse:
                best_mse = s
                best_l = l
                best_u = u

    print("ReLU networks")
    print("Best MSE obtained for neural network: {}".format(str(best_mse)))
    print("Hidden layers: {}".format(str(best_u)))
    print("L2 weight decay: {}".format(str(best_l)))

    best_mse, best_l, best_u = None, None, None
    for l in lambda_:
        for u in layers:
            s = 0
            print(l, u)
            for train, test in cv.split(X, y):
                activations = [sigmoid for i in range(len(u))]
                dactivations = [dsigmoid for i in range(len(u))]
                current_model = ANNRegression(units = u, lambda_ = l, 
                                              activations = activations, 
                                              dactivations = dactivations)
                current_model.fit(X[train, :], y[train])
                preds = current_model.predict(X[test])
                s += ((y[test] - preds) ** 2).sum()
        
            s /= X.shape[0]
            print("Estimated loss: {}".format(str(s)))
            if best_mse is None or s < best_mse:
                best_mse = s
                best_l = l
                best_u = u

    print("Sigmoid networks:")
    print("Best MSE obtained for neural network: {}".format(str(best_mse)))
    print("Hidden layers: {}".format(str(best_u)))
    print("L2 weight decay: {}".format(str(best_l)))

    s = 0
    for train, test in cv.split(X, y):
        current_model = SVR(RBF(sigma = 4.1), lambda_ = 0.01, epsilon = 10)
        current_model = current_model.fit(X[train, :], y[train])
        preds = current_model.predict(X[test, :])
        s += ((y[test] - preds) ** 2).sum()
    
    s /= X.shape[0]
    print("MSE for SVR: {}".format(str(s)))

def house_grid_search_class(X, y, lambda_ = [0.01, 0.1, 0.5], 
                            layers = [[], [10], [50], [10, 20], [50, 20],
                            [10, 20 ,30], [50, 20, 30]], k = 10):
    
    best_ll, best_l, best_u = None, None, None
    cv = StratifiedKFold(n_splits = k)

    sys.path.append("..")
    from homework_2.solution import MultinomialLogReg

    for l in lambda_:
        for u in layers:
            s = 0
            print(l, u)
            for train, test in cv.split(X, y):
                current_model = ANNClassification(units = u, lambda_ = l, seed = 3)
                current_model.fit(X[train, :], y[train])
                preds = current_model.predict(X[test])
                # Log loss computation
                row_ind = list(range(len(test)))
                preds = current_model.predict(X[test])
                s -= np.log(preds[row_ind, y[test]]).sum()
        
            s /= X.shape[0]
            print("Estimated loss: {}".format(str(s)))
            if best_ll is None or s < best_ll:
                best_ll = s
                best_l = l
                best_u = u

    print("ReLU networks")
    print("Best average log-loss obtained for neural network: {}".format(str(best_ll)))
    print("Hidden layers: {}".format(str(best_u)))
    print("L2 weight decay: {}".format(str(best_l)))

    best_ll, best_l, best_u = None, None, None
    for l in lambda_:
        for u in layers:
            s = 0
            print(l, u)
            for train, test in cv.split(X, y):
                activations = [sigmoid for i in range(len(u))]
                dactivations = [dsigmoid for i in range(len(u))]
                current_model = ANNClassification(units = u, lambda_ = l, 
                                              activations = activations, 
                                              dactivations = dactivations, seed = 3)
                current_model.fit(X[train, :], y[train])
                # Log loss computation
                row_ind = list(range(len(test)))
                preds = current_model.predict(X[test])
                s -= np.log(preds[row_ind, y[test]]).sum()
        
            s /= X.shape[0]
            print("Estimated loss: {}".format(str(s)))
            if best_ll is None or s < best_ll:
                best_ll = s
                best_l = l
                best_u = u

    print("Sigmoid networks:")
    print("Best average log-loss obtained for neural network: {}".format(str(best_ll)))
    print("Hidden layers: {}".format(str(best_u)))
    print("L2 weight decay: {}".format(str(best_l)))

    s = 0
    for train, test in cv.split(X, y):
        current_model = MultinomialLogReg().build(X[train, :], y[train])
        preds = current_model.predict(X[test, :])
        # Log loss computation
        row_ind = list(range(len(test)))
        preds = current_model.predict(X[test])
        s -= np.log(preds[row_ind, y[test]]).sum()
    
    s /= X.shape[0]
    print("Best average log-loss for multinomial logistic regression: {}".format(str(s)))

def read_huge_train(f = 1):
    df = pd.read_csv("./train.csv")
    df = df.sample(frac = 1, random_state = 42)
    df = df.loc[:, df.columns != "id"]
    df["target"] = df["target"].apply(lambda x : int(x.split("_")[1]) - 1)
    offset = int(df.shape[0] * f)
    X = df.loc[:, df.columns != "target"].to_numpy()

    X = X[:offset, :]
    y = df["target"].to_numpy()
    y = y[:offset]

    mu, std = X.mean(axis = 0), X.std(axis = 0)
    X = (X - mu) / std

    return X, y, mu, std

def read_huge_test(mu, std):
    df = pd.read_csv("./test.csv")
    df = df.sample(frac = 1, random_state = 42)
    df = df.loc[:, df.columns != "id"]
    X = df.to_numpy()

    X = (X - mu) / std
    return X

def huge_dataset_cv(X, y, units, lambdas, fit_networks = True, fit_baseline = False, k = 5):
    cv = StratifiedKFold(n_splits = k)

    if fit_networks:
        for lambda_ in lambdas:
            for unit in units:
                print(lambda_, unit)
                preds, time_ = np.empty(X.shape[0]), []

                for train, test in cv.split(X, y):
                    net = ANNClassification(units = unit, lambda_ = lambda_)
                    start = time.time()
                    net.fit(X[train, :], y[train])
                    finish = time.time()

                    time_taken = finish - start
                    print("Time taken: {:.4f}".format(finish - start))
                    time_.append(time_taken)
                    yhat = net.predict(X[test, :])
                    preds[test] = -np.log(yhat[list(range(len(test))), y[test]])

                print("Estimated loss: {}".format(str(preds.mean())))
                print("Total time taken: {}".format(str(sum(time_))))

                # Cache results to a local file
                out = [lambda_, unit, preds.mean(), preds.std(), time_]
                with open("./huge_dataset_results.txt", "a") as file:
                    file.write(str(out))
                    file.write("\n")

    if fit_baseline:
        baseline_loss = 0
        for train, test in cv.split(X, y):
            baseline = LogisticRegression(solver = "newton-cg").fit(X[train, :], y[train])
            preds = baseline.predict_proba(X[test, :])
            baseline_loss += cum_log_loss(y[test], preds, average = False)
        
        print("Baseline loss: {}".format(str(baseline_loss / X.shape[0])))

def huge_dataset_comparison():
    lambda_dict = {}
    total_time = 0

    with open("./huge_dataset_results.txt", "r") as f:
        for line in f:
            results = json.loads(line)
            l = results[0]
            units = results[1]
            mu, std = results[2], results[3]
            total_time += sum(results[4])

            if lambda_dict.get(l) is None:
                lambda_dict[l] = []

            lambda_dict[l].append([units, mu ,std])

    print("Estimated hours spent on computation: {}".format(total_time / 3600))
    
    plt.style.use("ggplot")
    
    pos = np.arange(len(lambda_dict)) + 1
    print(pos)

    for l in lambda_dict.keys():
        fig, ax = plt.subplots()
        ax.set_xlim(0, 2)

        units = [str(item[0]) for item in lambda_dict[l]]
        mus = np.array([item[1] for item in lambda_dict[l]])
        stds = np.array([item[1] for item in lambda_dict[l]])
        
        # 50000 was the number of original observations, luckily could be hardcoded here!
        ci_l = mus - 1.96 * stds / np.sqrt(50000)
        ci_r = mus + 1.96 * stds / np.sqrt(50000)

        ax.set_title(r"$\lambda$={}".format(str(l)))
        ax.barh(units, mus)

        for i in range(len(mus)):
            ax.plot((ci_l[i], ci_r[i]), (i, i), linewidth = 2, color = "black", label = "95% CI")

        ax.set_yticklabels([str(unit) for unit in units])
        ax.set_ylabel("Units")
        ax.set_xlabel("Mean log-loss after CV")
    plt.show()

def create_final_predictions(lambda_ = 0.01, units = [20, 50]):
    net = ANNClassification(units = units, lambda_ = lambda_)
    X_train, y_train, mu, std = read_huge_train()
    X_test = read_huge_test(mu, std)
    
    start = time.time()
    net.fit(X_train, y_train)
    end = time.time()
    print("Time required to fit the final model: {} minutes".format(str((end - start) / 60)))
    preds = net.predict(X_test)

    with open("./final.txt", "w+") as f:
        f.write("id,Class_1,Class_2,Class_3,Class_4,Class_5,Class_6,Class_7,Class_8,Class_9")
        f.write("\n")

        for i in range(preds.shape[0]):
            f.write(str(i + 1))
            for j in range(9):
                f.write(",")
                f.write(str(preds[i, j]))
            f.write("\n")

if __name__ == "__main__":
    X, y = housing_class_load()
    house_grid_search_class(X, y)
    '''
    X, y, mu, std = read_huge_train(f = 0.1)
    lambdas = [0.5]
    units = [[10], [10, 20], [10, 10, 10]]
    # huge_dataset_cv(X, y, units, lambdas)
    # huge_dataset_comparison()
    create_final_predictions()
    '''