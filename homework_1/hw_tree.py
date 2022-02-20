import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import time

from collections import Counter

def all_columns(X, rand):
    return list(range(X.shape[1]))

def random_sqrt_columns(X, rand):
    return rand.sample(list(range(X.shape[1])), int(np.sqrt(X.shape[1])))

class Tree:
    def __init__(self, rand = None, get_candidate_columns = all_columns, min_samples = 2):
        """
        Constructor for the Tree class
            # rand - Random class instance, used for reproduction of results
            get_candidate_columns - function which returns indices of features to be considered
                                    during the split procedure. Needed for random forest implementation
            min_samples - Number of minimum samples in the given node
        """
        self.rand = rand
        self.get_candidate_columns = get_candidate_columns
        self.min_samples = min_samples

    def __build(self, X, y):
        """
        Private utility function which recursivley builds the decision tree.
        X - dataset in the current node
        y - target values of the data points in the current node
        """
        d = X.shape[0]

        p = y.sum() / d

        # Number of nodes is smaller than the pre-set minimum number of nodes
        # or the node is completely pure - prunning cases
        if d < self.min_samples or p * (1 - p) == 0:
            return TreeNode(-1, -1, None, None, p)
        
        # Get candidate columns. For full decision trees, every column is considered.
        # For random forests, in each split we randomly pick sqrt(|columns|) columns
        # to reduce the correlation between trees.
        columns = self.get_candidate_columns(X, self.rand)

        min_cost = np.Inf
        best_c = -1
        v = -1
        
        for c in columns:
            values = np.unique(X[:, c])

            for value in values:
                # Splitting the dataset
                index_l = X[:, c] <= value
                index_r = X[:, c] > value

                X_left = X[index_l, :]
                X_right = X[index_r, :]

                y_left = y[index_l]
                y_right = y[index_r]

                # Split by the largest value is impossible
                if len(y_right) == 0:
                    continue
                
                # Compute the gini impurity index for child nodes
                p_left = y_left.sum() / len(y_left)
                p_right = y_right.sum() / len(y_right)

                gini_left = 1 - (p_left ** 2 + (1 - p_left) ** 2)
                gini_right = 1 - (p_right ** 2 + (1 - p_right) ** 2)

                cost = len(y_left) / d * gini_left + len(y_right) / d * gini_right

                # If possible, update the minimum impurity index
                if cost < min_cost:
                    min_cost = cost
                    best_c = c
                    v = value
                    X_sl = X_left
                    X_sr = X_right
                    y_sl = y_left
                    y_sr = y_right
                
                # If pure split was found, we terminate the search
                if min_cost == 0:
                    break
            # Same reasoning
            if min_cost == 0:
                break

        # If no reasonable split was found, prune
        if min_cost == np.Inf:
            return TreeNode(-1, -1, None, None, p)

        # Recusrivley build left and right subtrees
        left = self.__build(X_sl, y_sl)
        right = self.__build(X_sr, y_sr)
        return TreeNode(best_c, v, left, right, p)

    def build(self, X, y):
        """
        X - feature matrix
        y - vector of target values
        """
        return self.__build(X, y)

class TreeNode:
    def __init__(self, index, t, left, right, p):
        """
        Constructor for the TreeNode class.
            index - Index of the column for which the split in this node is performed
            t - threshold for the column value used to make a split
            left - pointer to the left child
            right - pointer to the right child
            p - ratio of positive samples in the region defined by the current node, used for predictions and Gini impurity index calculation
        """
        self.index = index
        self.t = t
        self.left = left
        self.right = right 
        self.p = p

    def __predict(self, X, predictions, indices):
        """
        Private function for recursive prediction. Dataset recursivley propagated 
        through the tree.
        """
        # Leaf node encountered
        if self.index == -1:
            # Make a majority vote prediction
            predictions[indices] = 1 if self.p >= 1 - self.p else 0
            return
        
        # Split the dataset
        selector_l = X[:, self.index] <= self.t
        selector_r = X[:, self.index] > self.t

        X_left = X[selector_l, :]
        X_right = X[selector_r, :]
        indices_l = indices[selector_l]
        indices_r = indices[selector_r]

        self.left.__predict(X_left, predictions, indices_l)
        self.right.__predict(X_right, predictions, indices_r)
        return

    def predict(self, X):
        predictions = np.zeros(X.shape[0])
        indices = np.array([i for i in range(X.shape[0])])
        self.__predict(X, predictions, indices)
        return predictions
        
class RandomForest:
    def __init__(self, rand = None, n = 50):
        """
        Random forest class constructor.
            rand - used for reproducibility
            n - number of bootstrapped samples (or the number of trees if you will)
        """
        self.n = n
        self.rand = rand
        self.rftree = Tree(rand = rand, get_candidate_columns = random_sqrt_columns, min_samples = 2)

    def build(self, X, y):
        """
        Builds a random forest from the given training data.
            X - feature matrix
            y - vector of target values
        """

        # Resulting list of decision trees
        # Also, for each tree we keep track of samples that were included in the training 
        # of that particular tree, so that we can use out of bag samples later during
        # the implementation of feature importance with random forests.

        trees = []
        out_of_bag_samples = []

        for i in range(self.n):
            # Generate a bootstrap sample
            data_b = self.rand.choices(list(range(X.shape[0])), k = X.shape[0])
            X_b = X[data_b,:]
            y_b = y[data_b]

            # Build a tree with randomized feature selection from the bootstrapped dataset
            trees.append(self.rftree.build(X_b, y_b))
            out_of_bag_samples.append(list(set(range(X.shape[0])) - set(data_b)))

        return RFModel(trees, out_of_bag_samples, X, y, self.rand)

class RFModel:
    def __init__(self, trees, out_of_bag_samples, X, y, rand):
        """
        Besides the trees, we have to also save the out of bag indices for every tree,
        and the original training set for estimating feature importance with random forests.
        """
        self.X = X
        self.y = y
        self.trees = trees
        self.out_of_bag_samples = out_of_bag_samples
        self.rand = rand

    def predict(self, X):
        """
        Make a prediction with the RF model. The predictions are made using 
        the majority vote method. 
        """
        
        # prediction_matrix[i][j] - prediction for j-th observation by i-th tree
        prediction_matrix = np.array([tree.predict(X) for tree in self.trees])
        predictions = prediction_matrix.sum(axis = 0)
        n = prediction_matrix.shape[0]
        predictions = [1 if votes >= n - votes else 0 for votes in predictions]
        return predictions
    
    def __out_of_bag_error(self, i = -1):
        """
        Computes the out of bag error of the random forest model by permuting
        the values of the i-th feature on every out of bag dataset. The default value
        of i is -1, which means that no permuting will take place, and hence by default 
        the function returns standard out of bag error of RF. 
        """

        # For every point x_i in the training set save we save the list of predictions
        #  made by the trees that did not have x_i in the bootstrapped dataset.
        predictions = [[] for j in range(self.X.shape[0])]

        for tree, oob_dataset in zip(self.trees, self.out_of_bag_samples):
            # If by some chance OOB dataset happens to be empty (which is unlikely but still
            # possible), just continue iterating

            if len(oob_dataset) == 0:
                continue
            
            samples = self.X[oob_dataset, :].copy()

            # If i is non -1, we permute the values of the i-th feature
            # Notice that we are using the rand object instead of np.permutation
            # for reproducibility!

            if i != -1:
                samples[:, i] = np.array(self.rand.sample(list(samples[:, i]), samples.shape[0]))

            t_predictions = tree.predict(samples)

            for j in range(len(t_predictions)):
                predictions[oob_dataset[j]].append(t_predictions[j])
        
        # For every x_i in the training set now use majority voting 
        # for class prediction
        predictions = [1 if sum(pred) >= len(pred) - sum(pred) else 0 for pred in predictions]

        # Compare the predictions with actual labels and return the MC rate
        return sum([True for j in range(self.X.shape[0]) if predictions[j] != self.y[j]]) / self.X.shape[0]
       
    def importance(self):
        """
        Feature importance with random forests implementation, as described in the 
        paper by Breiman(2001) - the inventor of random forests.
        """

        importances = []
        oob_standard = self.__out_of_bag_error()
        
        for j in range(self.X.shape[1]):
            oob_j = self.__out_of_bag_error(i = j)
            # Increase in MC rate is reported as the featurue importance
            # (We report percentages because the numbers can be small)
            importances.append((oob_j - oob_standard) * 100)
        
        return importances

def hw_tree_full(train, test):
    """
    Building a single decision tree for the given dataset.
        train - tuple containing X_train and y_train respectively
        test - tuple containing X_test and y_test respectively
    """

    X_train_matrix, y_train_matrix = train[0], train[1]
    X_test_matrix, y_test_matrix = test[0], test[1]
    
    start = time.time()
    t = Tree().build(X_train_matrix, y_train_matrix)

    print("Build finished in: {} seconds".format(time.time() - start))

    start_pred = time.time()

    pred_train = t.predict(X_train_matrix)
    pred_test = t.predict(X_test_matrix)

    print("Predictions finished in: {} seconds".format(time.time() - start_pred))

    # Compute the error rate on training and test set
    mc_train = sum([pred_train[i] != y_train_matrix[i] for i in range(len(pred_train))]) / len(pred_train)
    mc_test = sum([pred_test[i] != y_test_matrix[i] for i in range(len(pred_test))]) / len(pred_test)

    # Compute the standard errors
    se_train = np.sqrt(mc_train * (1 - mc_train) / len(pred_train))
    se_test = np.sqrt(mc_test * (1 - mc_test) / len(pred_test))

    print("Full decision tree")
    print("Training set")
    print("Computed MC rate: {:.4f}".format(mc_train))
    print("Standard error: {:.4f}".format(se_train))
    print("Test set")
    print("Computed MC rate: {:.4f}".format(mc_test))
    print("Standard error: {:.4f}".format(se_test))

    return (mc_train, se_train), (mc_test, se_test)

def hw_randomforests(train, test):
    """
    Building a random forest of 100 trees for the given dataset.
        train - tuple containing X_train and y_train respectively
        test - tuple containing X_test and y_test respectively
    """

    X_train_matrix, y_train_matrix = train[0], train[1]
    X_test_matrix, y_test_matrix = test[0], test[1]

    start = time.time()
    t = RandomForest(rand = random.Random(1), n = 100).build(X_train_matrix, y_train_matrix)

    print("Build finished in: {} seconds".format(time.time() - start))

    start_pred = time.time()
    pred_train = t.predict(X_train_matrix)
    pred_test = t.predict(X_test_matrix)

    print("Predictions finished in {:.4f} seconds".format(time.time() - start_pred))

    # Compute the error rate on training and test set
    mc_train = sum([pred_train[i] != y_train_matrix[i] for i in range(len(pred_train))]) / len(pred_train)
    mc_test = sum([pred_test[i] != y_test_matrix[i] for i in range(len(pred_test))]) / len(pred_test)

    # Compute the standard errors
    se_train = np.sqrt(mc_train * (1 - mc_train) / len(pred_train))
    se_test = np.sqrt(mc_test * (1 - mc_test) / len(pred_test))

    print(se_test)

    print("Random forest")
    print("Training set")
    print("Computed MC rate: {:.4f}".format(mc_train))
    print("Standard error: {:.4f}".format(se_train))
    print("Test set")
    print("Computed MC rate: {:.4f}".format(mc_test))
    print("Standard error: {:.4f}".format(se_test))

    return (mc_train, se_train), (mc_test, se_test)

def dataset_extraction():
    """
    Parses tki-resistance.csv file to get training and test sets.
    """

    # Loading gthe dataset
    df = pd.read_csv("./tki-resistance.csv")
    # Converting classes to 0-1
    df["Class"] = df["Class"].apply(lambda x : 1 if x == "Bcr-abl" else 0)
    
    # Extracting the training and test datasets, as per instructions for the homework
    df_train = df.iloc[:130,:].copy()
    df_test = df.iloc[130:,:].copy()

    X_train = df_train.loc[:, df.columns != "Class"].copy()
    y_train = df_train.loc[:, df.columns == "Class"].copy()

    X_train_matrix = X_train.to_numpy()
    y_train_matrix = y_train.to_numpy().flatten()

    X_test = df_test.loc[:, df.columns != "Class"].copy()
    y_test = df_test.loc[:, df.columns == "Class"].copy()

    X_test_matrix = X_test.to_numpy()
    y_test_matrix = y_test.to_numpy().flatten()

    return (X_train_matrix, y_train_matrix), (X_test_matrix, y_test_matrix)

def mc_versus_n(lower_bound, upper_bound, increment):
    """
    Misclassification rate on the test set versus the number of trees on the given dataset.
    For this homework, we consider all integers k in [50, 100] with increments of 5
    as the number of DTs in the random forest. We will also plot the MC rate on the
    trainig set as well for comparison.
    """

    # Loading gthe dataset
    df = pd.read_csv("./tki-resistance.csv")
    # Converting classes to 0-1
    df["Class"] = df["Class"].apply(lambda x : 1 if x == "Bcr-abl" else 0)
    
    # Extracting the training and test datasets, as per instructions for the homework
    df_train = df.iloc[:130,:].copy()
    df_test = df.iloc[130:,:].copy()

    X_train = df_train.loc[:, df.columns != "Class"].copy()
    y_train = df_train.loc[:, df.columns == "Class"].copy()

    X_train_matrix = X_train.to_numpy()
    y_train_matrix = y_train.to_numpy().flatten()

    X_test = df_test.loc[:, df.columns != "Class"].copy()
    y_test = df_test.loc[:, df.columns == "Class"].copy()

    X_test_matrix = X_test.to_numpy()
    y_test_matrix = y_test.to_numpy().flatten()

    num_trees = []
    mcs_train, mcs_test = [], []

    for i in range(lower_bound, upper_bound + increment, increment):
        num_trees.append(i)
        print("Constructing RF with {:d} trees".format(i))
        t = RandomForest(rand = random.Random(1), n = i).build(X_train_matrix, y_train_matrix)

        test_pred = t.predict(X_test_matrix)
        train_pred = t.predict(X_train_matrix)

        mc_train = sum([y_train_matrix[j] != train_pred[j] for j in range(len(train_pred))]) / len(train_pred)
        mc_test = sum([y_test_matrix[j] != test_pred[j] for j in range(len(test_pred))]) / len(test_pred)

        mcs_train.append(mc_train)
        mcs_test.append(mc_test)
    
    plt.style.use("ggplot")
    fig, ax = plt.subplots()
    ax.set_xlabel("Number of trees in RF")
    ax.set_ylabel("MC rate on the test set", labelpad = 10)

    ax.plot(num_trees, mcs_test, label = "Test set")
    ax.scatter(num_trees, mcs_test)

    ax.plot(num_trees, mcs_train, label = "Training set")
    ax.scatter(num_trees, mcs_train)

    ax.set_xticks(range(lower_bound, upper_bound + increment, increment))
    ax.legend()

    plt.show()

def feature_importance():
    """
    Compute feature importances on the given dataset
    """

    df = pd.read_csv("./tki-resistance.csv")

    # Converting classes to 0-1
    df["Class"] = df["Class"].apply(lambda x : 1 if x == "Bcr-abl" else 0)
    
    # Extracting the training and test datasets, as per instructions for the homework
    df_train = df.iloc[:130,:].copy()

    feature_names = df_train.columns

    X_train = df_train.loc[:, df.columns != "Class"].copy()
    y_train = df_train.loc[:, df.columns == "Class"].copy()

    X_train_matrix = X_train.to_numpy()
    y_train_matrix = y_train.to_numpy().flatten()

    start_b = time.time()
    model = RandomForest(rand = random.Random(1), n = 100).build(X_train_matrix, y_train_matrix)
    print("Time taken for building the RF model: {:.4f}".format(time.time() - start_b))

    start_i = time.time()
    imps = model.importance()
    print("Time taken for computing feature importance: {:.4f}".format(time.time() - start_i))

    print(imps)

    d = {feature: imp for feature, imp in zip(feature_names, imps)}
    d_filtered = dict(filter(lambda x : x[1] != 0, d.items()))

    print("Total number of features: {:d}".format(len(d)))
    print("Number of features with zero importance: {:d}".format(len(d) - len(d_filtered)))
    
    d_filtered = sorted(d_filtered.items(), key = lambda x : x[1], reverse = True)

    mappings = {}

    for item in d_filtered:
        if mappings.get(item[1]) is None:
            mappings[item[1]] = [item[0]]
        
        else:
            mappings[item[1]].append(item[0])
    
    plt.style.use("ggplot")
    fig, ax = plt.subplots()

    x = ["Group {:d}".format(i + 1) for i in range(len(mappings))]
    y = [key for key in mappings.keys()]

    for group, imp in zip(x, y):
        print("{} features:".format(group))
        print(mappings[imp])

    ax.barh(x, y, edgecolor = "black", linewidth = 1.2)
    ax.set_ylabel("Feature group")
    ax.set_xlabel("Percent increase in MC rate", labelpad = 10)
    ax.axvline(x = 0, color = "black", linewidth = 2, linestyle = "--")
    plt.show()

def dt_vs_rf():
    """
    Comparing DT and RF models on the given dataset. For uncertainty quantification
    we generate a 95% confidence interval for model misclassification rate.
    """
    train, test = dataset_extraction()

    (dt_train_mc, dt_train_se), (dt_test_mc, dt_test_se) = hw_tree_full(train, test)
    (rf_train_mc, rf_train_se), (rf_test_mc, rf_test_se) = hw_randomforests(train, test)

    dt_un_l = dt_test_mc - 1.96 * dt_test_se
    dt_un_r = dt_test_mc + 1.96 * dt_test_se

    rf_un_l = rf_test_mc - 1.96 * rf_test_se
    rf_un_r = rf_test_mc + 1.96 * rf_test_se

    print("DT: [{:.4f}, {:.4f}]".format(dt_un_l, dt_un_r))
    print("RF: [{:.4f}, {:.4f}]".format(rf_un_l, rf_un_r))

def root_dist(num_trees = 100):
    """
    We generate 100 artifical datasets with bootstrap, construct full decision trees on them,
    and check the distribution of features in the roots of newly made trees. We compare the 
    results to feature importance obtained by running Breiman's algorithm.
    """

    c = Counter()
    df = pd.read_csv("./tki-resistance.csv")
    df["Class"] = df["Class"].apply(lambda x : 1 if x == "Bcr-abl" else 0)

    X = df.loc[:, df.columns != "Class"].to_numpy()
    y = df.loc[:, "Class"].to_numpy()

    # Reproducibility object
    rnd = random.Random(1)

    for i in range(num_trees):
        indices = rnd.choices(list(range(X.shape[0])), k = X.shape[0])
        
        X_boot = X[indices, :].copy()
        y_boot = y[indices].copy()

        start = time.time()

        print("Started training tree {:d}".format(i + 1))
        t = Tree().build(X_boot, y_boot)
        print("Finished training tree {:d} in {:.4f}\n".format(i + 1, time.time() - start))

        c[df.columns[t.index]] += 1

    m = max(c.values())
    items = sorted(c.items(), key = lambda x : x[1])
    features, freqs = zip(*items)

    features = [feature.split(".")[0] for feature in features]

    plt.style.use("ggplot")
    fig, ax = plt.subplots()

    ax.bar(features, freqs, edgecolor = "black", linewidth = 1.2)
    ax.set_xlabel("Feature label")
    ax.set_ylabel("Number of appearances as a root of a DT", labelpad = 10)
    ax.set_yticks(range(0, m + 1, 2))
    plt.show()

if __name__ == "__main__":
    root_dist(num_trees = 100)