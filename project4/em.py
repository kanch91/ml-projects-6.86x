"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """

    n, d = X.shape
    mu, var, pi = mixture  # Unpack mixture tuple
    K = mu.shape[0]

    # ##### Loop version: 2nd fastest way to calculate norms #####
    #
    # # f(u,j) matrix that's used to store the normal matrix and log of posterior probability: (p(j|u))
    # f = np.zeros((n,K), dtype=np.float64)
    #
    #    # Compute the normal matrix: Single loop implementation
    # for i in range(n):
    #     Cu_indices = X[i,:] != 0 # For each user pick only columns that have ratings
    #     dim = np.sum(Cu_indices) # Dimension of Cu (no. of non-zero entries)
    #     pre_exp = (-dim/2.0)*np.log((2*np.pi*var)) # log of pre-exponent for this user's gaussian distribution
    #     # Calculate the exponent term of the gaussian
    #     diff = X[i, Cu_indices] - mu[:, Cu_indices]    # This will be (K,|Cu|)
    #     norm = np.sum(diff**2, axis=1)  # This will be (K,)
    #
    #     # Now onto the final log normal matrix: log(N(...))
    #     # We will need log(normal), exp will cancel, so no need to calculate it
    #     f[i,:] = pre_exp - norm/(2*var)  # This is the ith users log gaussian dist vector: (K,)
    #
    # ##### End of the loop version #####

    ##### Vectorized version: 1st & 3rd fastest way to calculate norms #####

    # Create a delta matrix to indicate where X is non-zero, which will help us pick Cu indices
    delta = X.astype(bool).astype(int)

    # Exponent term: norm matrix/(2*variance)
    # This is using 3D broadcasting: slowest of all
    #    f = np.sum(((X[:, None, :] - mu)*delta[:, None, :])**2, axis=2)/(2*var)

    # This is using indicator matrix: fastest of all
    f = (np.sum(X ** 2, axis=1)[:, None] + (delta @ mu.T ** 2) - 2 * (X @ mu.T)) / (2 * var)
    # Pre-exponent term: A matrix of shape (n, K)
    pre_exp = (-np.sum(delta, axis=1).reshape(-1, 1) / 2.0) @ (np.log((2 * np.pi * var)).reshape(-1, 1)).T
    f = pre_exp - f

    ##### End of the vectorized version #####

    f = f + np.log(pi + 1e-16)  # This is the f(u,j) matrix

    # log of normalizing term in p(j|u)
    logsums = logsumexp(f, axis=1).reshape(-1, 1)  # Store this to calculate log_lh
    log_posts = f - logsums  # This is the log of posterior probability matrix: log(p(j|u))
    log_likelihood = np.sum(logsums, axis=0).item()  # This is the log likelihood

    return np.exp(log_posts), log_likelihood

    raise NotImplementedError



def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """

    n, d = X.shape
    mu_rev, _, _ = mixture
    K = mu_rev.shape[0]

    pi_rev = np.sum(post, axis=0) / n # Calculate revised pi(j): same expression as in the naive case

    delta = X.astype(bool).astype(int) # Create delta matrix indicating where X is non-zero

    # Update means only when sum_u(p(j|u)*delta(l,Cu)) >= 1
    denominator = post.T @ delta  # Denominator (K,d): Only include dims that have information
    numerator = post.T @ X  # Numerator (K,d)
    update_indices = np.where(denominator >= 1)  # Indices for updation
    mu_rev[update_indices] = numerator[update_indices] / denominator[update_indices]  # Only update where necessary (denom>=1)

    # Update variances
    denominator_var = np.sum(post * np.sum(delta, axis=1).reshape(-1, 1), axis=0)  # Shape: (K,)

    ##### Loop version for norms calculation ######

    # Norm matrix for variance calc
    #    norms = np.zeros((n, K), dtype=np.float64)
    #
    #    for i in range(n):
    #        # For each user pick only columns that have ratings
    #        Cu_indices = X[i,:] != 0
    #        diff = X[i, Cu_indices] - mu_rev[:, Cu_indices]    # This will be (K,|Cu|)
    #        norms[i,:] = np.sum(diff**2, axis=1)  # This will be (K,)

    ##### End of the loop version ######

    ##### Vectorized version for norms calculation #####

    # norms = np.sum(((X[:, None, :] - mu_rev)*delta[:, None, :])**2, axis=2)
    norms = np.sum(X ** 2, axis=1)[:, None] + (delta @ mu_rev.T ** 2) - 2 * (X @ mu_rev.T)

    ##### End of the vectorized version #####

    # Revised var: if var(j) < 0.25, set it = 0.25
    var_rev = np.maximum(np.sum(post * norms, axis=0) / denominator_var, min_variance)

    return GaussianMixture(mu_rev, var_rev, pi_rev)

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
    new_log_likelihood = None  # Keep track of log likelihood to check convergence

    # Start the main loop
    while old_log_likelihood is None or (new_log_likelihood - old_log_likelihood > 1e-6 * np.abs(new_log_likelihood)):
        old_log_likelihood = new_log_likelihood
        post, new_log_likelihood = estep(X, mixture) # E-step
        mixture = mstep(X, post, mixture) # M-step
    return mixture, post, new_log_likelihood

    raise NotImplementedError


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    n, d = X.shape
    X_pred = X.copy()
    K, _ = mixture.mu.shape

    for i in range(n):
        mask = X[i, :] != 0
        mask0 = X[i, :] == 0
        post = np.zeros(K)
        for j in range(K):
            log_likelihood = log_gaussian(X[i, mask], mixture.mu[j, mask],
                                          mixture.var[j])
            post[j] = np.log(mixture.p[j]) + log_likelihood
        post = np.exp(post - logsumexp(post))
        X_pred[i, mask0] = np.dot(post, mixture.mu[:, mask0])
    return X_pred


def log_gaussian(x: np.ndarray, mean: np.ndarray, var: float) -> float:
    """Computes the log probablity of vector x under a normal distribution

    Args:
        x: (d, ) array holding the vector's coordinates
        mean: (d, ) mean of the gaussian
        var: variance of the gaussian

    Returns:
        float: the log probability
    """
    d = len(x)
    log_prob = -d / 2.0 * np.log(2 * np.pi * var)
    log_prob -= 0.5 * ((x - mean)**2).sum() / var
    return log_prob
