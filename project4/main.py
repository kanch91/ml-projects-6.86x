import numpy as np
import kmeans
import common
import naive_em
import em

X = np.loadtxt("toy_data.txt")

K = [1, 2, 3, 4]    # Trying clusters for K-means
seeds = [0, 1, 2, 3, 4]     # Trying seeds

# Costs for different seeds
kMeans_cost = [0, 0, 0, 0, 0]
EM_cost = [0, 0, 0, 0, 0]

# Best seed for cluster based on lowest costs
kMeans_best_seed = [0, 0, 0, 0]
EM_best_seed = [0, 0, 0, 0]

# Mixtures for best seeds
kMeans_mixtures = [0, 0, 0, 0, 0]
EM_mixtures = [0, 0, 0, 0, 0]

# Storing the posterior probabilities for best seeds
kMeans_posts = [0, 0, 0, 0, 0]
EM_posts = [0, 0, 0, 0, 0]

# BIC score of cluster
bic = [0., 0., 0., 0.]

for k in range(len(K)):
    for i in range(len(seeds)):
        # Running the kMeans algorithm written in file "kmeans.py"
        kMeans_mixtures[i], kMeans_posts[i], kMeans_cost[i] = \
            kmeans.run(X, *common.init(X, K[k], seeds[i]))

        # # Running the Naive EM algorithm written in file "naive_em.py"
        # mixtures_EM[i], posts_EM[i], EM_cost[i] = \
        #     naive_em.run(X, *common.init(X, K[k], seeds[i]))

    # Printing the lowest cost
    print("---------------------------------------------------")
    print("Number of Clusters:", k + 1)
    print("Lowest cost using kMeans is:", np.min(kMeans_cost))
    print("Highest log likelihood using EM is:", np.max(EM_cost))
    print("---------------------------------------------------\n")

    # Saving the best seed for plotting
    kMeans_best_seed[k] = np.argmin(kMeans_cost)
    EM_best_seed[k] = np.argmax(EM_cost)

    # Plotting the kMeans and EM results obtained
    common.plot(X,
                kMeans_mixtures[kMeans_best_seed[k]],
                kMeans_posts[kMeans_best_seed[k]],
                title="kMeans")

    # common.plot(X,
    #             mixtures_EM[EM_best_seed[k]],
    #             posts_EM[EM_best_seed[k]],
    #             title="EM")
    #
    # # BIC score for EM algorithm
    # bic[k] = common.bic(X, mixtures_EM[EM_best_seed[k]], np.max(EM_cost))

# Print the best K based on BIC
print("---------------------------------------------------")
print("BIC")
print("Best K is:", np.argmax(bic) + 1)
print("BIC for the best K is:", np.max(bic))
print("---------------------------------------------------\n")
