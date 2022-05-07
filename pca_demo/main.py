from sklearn.datasets import load_iris
from numpy.linalg import svd, eigh
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import numpy as np

"""
Iris dataset summary

Feature order:
    ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

Class values:
    ['setosa' 'versicolor' 'virginica']
"""

def pca_cov(X, ncomp = 2):
    """
    Principal component analysis through diagonalization of the covariance matrix.
    
    Parameters:
        X - the data matrix, not necessarily standardized by columns
        ncomp - number of principal components to use
    
    Returns:
        ev - ncomp highest eigenv alues of the covariance matrix
        pc - ncomp normed eigenvectors that correspond to ncomp highest eigen values of 
             the covariance matrix
    """
    # Covariance matrix, using the Bessel corrected estimator for variance
    C = 1 / (X.shape[0] - 1) * X.T.dot(X)

    # Diagonalizing the covariance matrix
    W, V = eigh(C)
    # Reverse eigen values and eigenvectors because the returned order of 
    # eigen values is non-decreasing

    W = W[::-1]
    V = V[:, ::-1]

    return W[:ncomp], V[:, :ncomp]

def pca_svd(X, ncomp = 2):
    """
    Principal component analysis through singular value decomposition of the standardized data
    matrix.

    Returns:
        ev - ncomp highest singular values of the data matrix
        pc - ncomp right normed eigenvectors of the data matrix 
             (or equivalently, ncomp eigenvectors of the covariancec matrix)
    """
    # Perform full SVD
    U, S, V = svd(X, full_matrices = True)

    # Relationship between singular values of X and eigen values of the covariance matrix
    S = 1 / (X.shape[0] - 1) * S ** 2

    return S[:ncomp], V.T[:, :ncomp]

def project(X, pc):
    """
    Projects the data matrix X to a lower dimensional space using the computed
    principal axes
    """
    return X.dot(pc)

def explained_variance(w):
    """
    Computes the explained variance and cummulative
    explained variance for each principal component using the corresponding eigen values
    (singular values)
    """
    ind = w / w.sum()
    cu = np.cumsum(ind)
    return ind, cu

if __name__ == "__main__":
    """
    Using PCA to perform a 2D projection of the Iris dataset.
    """

    obj = load_iris()
    
    X = obj.data
    y = obj.target

    colors = ["red", "green", "blue"]
    labels = ["setosa", "versicolor", "virginica"]

    # Important, data matrix should always be standardized before performing PCA!
    X = (X - X.mean(axis = 0)) / X.std(axis = 0)

    W, V = pca_cov(X)
    X_1 = project(X, V)
    print(V)
    print("Eigen values of the covariance matrix:")
    print(W)
    print("Total and cummulative explained variance:")
    print(explained_variance(W))
    print()

    S, V_prime = pca_svd(X)
    X_2 = project(X, V_prime)
    print(V_prime)
    print("Transformed singular values of the data matrix:")
    print(S)
    print("Total and cummulative explained variance:")
    print(explained_variance(S))
    print()

    plt.style.use("ggplot")
    fig, ax = plt.subplots(ncols = 2)

    for i in range(3):
        ind = y == i
        ax[0].scatter(X_1[ind, 0], X_1[ind, 1], color = colors[i], label = labels[i])
        ax[1].scatter(X_2[ind, 0], X_2[ind, 1], color = colors[i], label = labels[i])

    ax[0].set_title("PCA with diagonalization of the covariance matrix")
    ax[0].legend()
    ax[0].set_xlabel("PC1")
    ax[0].set_ylabel("PC2")

    ax[1].set_title("PCA with singular value composition of the data matrix")
    ax[1].legend()
    ax[1].set_xlabel("PC1")
    ax[1].set_ylabel("PC2")

    plt.show()

    """
    Conclusion: This example demonstrates that the choice of optimal projection basis is not unique,
                and depends on the choice of orthogonal basis of eigenvectors. One way of saying 
                this is that the solution is unique up to a rotation and translation.
    """