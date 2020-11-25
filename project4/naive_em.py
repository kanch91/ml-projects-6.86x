"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture



def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """

    n, d = X.shape
    mu, var, pi = mixture  # Unpacking mixture tuple
    k = mu.shape[0]

    # Computing the normal distribution matrix: (N, K)
    pre = (2 * np.pi * var) ** (d / 2)

    # Calculating the exponent term: norm matrix/(2*variance)
    post = np.linalg.norm(X[:, None] - mu, ord=2, axis=2) ** 2  # Vectorized version for faster
    post = np.exp(-post / (2 * var))

    #    post = np.zeros((n, k), dtype=np.float64) # Loop version: Array to hold posterior probability & normal matrix
    #    for i in range(n):  # Use single loop to complete Normal matrix: faster than broadcasting in 3D
    #        dist = X[i,:] - mu     # Compute difference: will be (K,d) for each n
    #        norm = np.sum(dist**2, axis=1)  # Norm: will be (K,) for each n
    #        post[i,:] = np.exp(-norm/(2*var))   # This is the exponent term of normal

    post = post / pre  # Final Normal matrix: will be (n, K)
    numerator = post * pi # Calculating the numerator
    denominator = np.sum(numerator, axis=1).reshape(-1, 1)  # This is the vector p(x;theta)
    post = numerator / denominator  # This is the matrix of posterior probability p(j|i)
    log_likelihood = np.sum(np.log(denominator), axis=0).item()  # Log-likelihood

    return post, log_likelihood

    raise NotImplementedError


def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """

    n, d = X.shape
    k = post.shape[1]

    nj = np.sum(post, axis=0)  # shape is (K, )
    pi = nj / n  # Cluster probability; shape is (K, )
    mu = np.dot(post.T, X) / nj.reshape(-1, 1)  # Revised means; shape is (K,d)
    norms = np.linalg.norm(X[:, None] - mu, ord=2, axis=2) ** 2  # Vectorized version

    #    norms = np.zeros((n, k), dtype=np.float64) # Loop version: Matrix to hold all the norms: (n,K)
    #    for i in range(n):
    #        dist = X[i,:] - mu # Calculating the distribution
    #        norms[i,:] = np.sum(dist**2, axis=1) #Calculating norm

    var = np.sum(post * norms, axis=0) / (nj * d)  # Updating the variance; shape is (K, )

    return GaussianMixture(mu, var, pi)

    raise NotImplementedError


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """

    old_log_likelihood = None
    new_log_likelihood = None  # To keep a track of log likelihood to check the convergence

    # Starting the main loop for runnning the EM Algorithm
    while old_log_likelihood is None or (new_log_likelihood - old_log_likelihood > 1e-6 * np.abs(new_log_likelihood)):
        old_log_likelihood = new_log_likelihood
        post, new_log_likelihood = estep(X, mixture) # Running the E-step
        mixture = mstep(X, post) # Running the M-step

    return mixture, post, new_log_likelihood

    raise NotImplementedError
