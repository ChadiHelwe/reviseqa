from statsmodels.stats.proportion import proportion_confint as pc
import numpy as np
from scipy.stats import binom


def lo(x, n, alpha):
    """
    clopper-pearson lower bound

    x - list of count of positive outcomes
    n - number of trials
    alpha - failure prob

    returns (lower_bound, upper_bound)
    """
    lower, upper = pc(x, n, alpha=alpha, method="beta")
    return lower, upper


def lor(x, n, alpha):
    """
    randomized clopper-pearson lower bound

    x - list of counts of positive outcomes
    n - number of trials
    alpha - failure prob

    output - list of lower bounds
    """
    if type(x) == int:
        x = [x]
    v = np.random.rand(len(x))
    lo = np.zeros(len(x))
    hi = np.ones(len(x))

    for _ in range(25):
        mid = (hi + lo) / 2
        a = 1 - binom.cdf(x, n, mid) + v * binom.pmf(x, n, mid)
        mask = a > alpha
        hi[mask] = mid[mask]
        lo[~mask] = mid[~mask]
    return lo


if __name__ == "__main__":
    p = 0.7
    n = 1000
    alpha = 0.05
    x = np.random.binomial(n, p, 1)
    print(x)
    lo_det = lo(x, n, alpha)
    print(lo_det)
    lo_ran = lor(x, n, alpha)
    print(lo_ran)
    lo_ran = lor(x, n, alpha)
    print(lo_ran)
    lo_det = lo(n - x, n, alpha)
    print(lo_det)
    lo_ran = lor(n - x, n, alpha)
    print(lo_ran)
    lo_ran = lor(n - x, n, alpha)
    print(lo_ran)

    # print('det failures', (lo_det > 0.7).mean())
    # print('ran failures', (lo_ran > 0.7).mean())
