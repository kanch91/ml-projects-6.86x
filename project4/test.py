import numpy as np
import em
import common

# Testing implementation of EM algorithm
X = np.loadtxt("test_incomplete.txt")
X_gold = np.loadtxt("test_complete.txt")

n, d = X.shape
K = 4
seed = 0

mix_conv, post_conv, log_lh_conv = em.run(X, *common.init(X, K, seed))
X_predict = em.fill_matrix(X, mix_conv)
rmse = common.rmse(X_gold, X_predict)


# Comparison of EM for matrix completion with K = 1 and 12
import time

X = np.loadtxt("netflix_incomplete.txt")
X_gold = np.loadtxt("netflix_complete.txt")

K = [1, 12]  # Clusters to try
log_lh = [0, 0, 0, 0, 0]  # Log likelihoods for different seeds
best_seed = [0, 0] # Best seed for cluster based on highest log likelihoods
mixtures = [0, 0, 0, 0, 0] # Mixtures for best seeds
posts = [0, 0, 0, 0, 0] # Posterior probability for best seeds
rmse = [0., 0.] # RMS Error for clusters

start_time = time.perf_counter()

for k in range(len(K)):
    for i in range(5):
        # Run EM
        mixtures[i], posts[i], log_lh[i] = \
            em.run(X, *common.init(X, K[k], i))

    # Print lowest cost
    print("\n-------------------Number of Clusters:", K[k], "------------------")
    print("Highest log likelihood using EM is:", np.max(log_lh), "\n")

    # Save best seed for plotting
    best_seed[k] = np.argmax(log_lh)
    # Use the best mixture to fill prediction matrix
    X_pred = em.fill_matrix(X, mixtures[best_seed[k]])
    rmse[k] = common.rmse(X_gold, X_pred)

print("------------------------------------------------------")
print("RMS Error for K = 12 is: {:.4f}".format(rmse[1]))
end_time = time.perf_counter()
print("Run time: {:.4f} seconds".format(end_time - start_time))
