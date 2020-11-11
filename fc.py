import numpy as np
from scipy.stats import poisson


def main(n_obs, b, alpha=0.9):
    """Plot a confidence interval for feldman-cousins."""
    # Take mus in a given range
    # TODO: optimize this range
    epsilon = 0.005
    mus = np.arange(0, 50+epsilon, epsilon)  # shape : (10000,)

    # For each mu, calculate Poisson probability over possible x
    x = np.arange(0, 16)
    # TODO: this take ~98% of the program time. Optimize?
    P_n_mu = [poisson(mu + b).pmf(x) for mu in mus]  # (10000, 16)

    # For each mu entry in the matrix, get the maximum poisson likelihood
    # Construct a look-up-table. Given b. For each n, give best P(n|mu_best+b)
    table = {}
    for n in x:
        row = poisson(x + b).pmf(n)
        table[n] = max(row)

    # Now calculate R
    R = []
    for row in P_n_mu:
        newRow = []
        for i, val in enumerate(row):
            newRow += [val / table[x[i]]]
        R += [newRow]

    # Process R according to the FC algorithm to get min/max n
    lowerN, upperN = [], []
    lenRow = len(x)
    for j, row in enumerate(R):
        row = np.array(row)
        idx = np.argsort(row)
        n, row = x[idx][::-1], P_n_mu[j][idx][::-1]
        # Find the first index where cumulative sum of row > alpha
        sumRow = np.cumsum(row)
        for i, val in enumerate(sumRow):
            if val > 0.9:
                break
        if i == lenRow - 1:
            continue
        # Return the highest, lowest n where sumRow < alpha (+1 element)
        n_that_pass = n[:i+1]
        lowerN += [min(n_that_pass)]
        upperN += [max(n_that_pass)]

    from matplotlib import pyplot as plt
    plt.plot(lowerN, mus[:len(lowerN)])
    plt.plot(upperN, mus[:len(lowerN)])
    plt.xticks(x)
    plt.yticks(x)
    plt.xlim(0, 15)
    plt.ylim(0, 15)
    plt.grid(color="grey", linestyle="--", linewidth=.5, alpha=.67)
    plt.show()


main(0, 3)
