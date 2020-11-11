"""
Calculate FC upper limits in a poisson experiment.

Validated against ROOT TFeldmanCousins :-) Use the function getPoisson_UL_FC.
"""
import numpy as np
from scipy.stats import poisson


def get_limits_n(mu, b, x, table, alpha=0.9):
    """Calculate n bounds for a given mu following FC algorithm."""
    # Calculate poissonian probability of n given mu
    P_n_mu = poisson(mu + b).pmf(x)

    # Calculate R
    R = []
    for i, val in enumerate(P_n_mu):
        R += [val / table[x[i]]]

    # Process R according to the FC algorithm to get min/max n
    idx = np.argsort(R)
    n, row = x[idx][::-1], P_n_mu[idx][::-1]
    # Find the first index where cumulative sum of row > alpha
    sumRow = np.cumsum(row)
    for i, val in enumerate(sumRow):
        if val > alpha:
            break

    n_that_pass = n[:i+1]
    lowerN, upperN = min(n_that_pass), max(n_that_pass)

    return lowerN, upperN


def get_table(x, b):
    """Construct a look-up-table. Given b. For each n, get P(n|mu_best+b)."""
    table = {}
    for n in x:
        row = poisson(x + b).pmf(n)
        table[n] = max(row)
    return table


def getPoisson_UL_FC(b, n_obs, alpha=0.9, threshold=0.001):
    """Calculate a FC upper limit for n_obs observed and b background."""
    # Start at mu = 0, x = np.arange(15)
    mu = 0.
    mu_step = b / 2.
    x = np.arange(15)
    table = get_table(x, b)

    done = False
    while not done:
        lowerN, upperN = get_limits_n(mu, b, x, table, alpha)

        # Be wary of edge effects
        if upperN >= x[-2]:
            # Kinda inefficient but ok because it usually doesn't happen often
            x = np.append(x, x[-1]+1)
            table = get_table(x, b)
            continue

        # If you overshoot, reduces step size and go back
        if lowerN > n_obs+1:
            mu_step /= 2
            mu -= mu_step

        # If you undershoot, continue climbing up
        elif lowerN < n_obs+1:
            mu += mu_step

        # Continue until the step size reaches a given threshold
        else:
            if mu_step <= threshold:
                break
            mu_step /= 2
            mu -= mu_step

    return mu
