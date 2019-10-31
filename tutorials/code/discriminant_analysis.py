import numpy as np
from numpy import genfromtxt
from scipy.stats import multivariate_normal


def fit_gaussian(data, label=None):
    ys = data[:, 0]
    xs = data[:, 1:]

    if label is not None:
        xs = xs[ys == label]

    pi = xs.shape[0] / data.shape[0]
    N = xs.shape[0]
    mu = xs.sum(axis=0) / N
    var = (xs - mu).T @ (xs - mu) / N
    return mu, var, pi


def likelihood(data, classes):
    ys = data[:, 0]
    xs = data[:, 1:]

    logpdf = 0
    for label, params in classes.items():
        mu, var, pi = params
        logp_xs = multivariate_normal.logpdf(xs, mu, np.sqrt(var))
        logp_ys = np.ones_like(ys) * np.log(pi)
        logpdf += np.sum((logp_xs + logp_ys)[ys == label])

    return logpdf


def BIC(data, classes, d):
    return likelihood(data, classes) - d * np.log(data.shape[0]) / 2


# Read data and skip head line
data = genfromtxt("heightWeightData.csv", delimiter=",")
data = data[1:]

male = 1
female = 2

mu_all, var_all, pi_all = fit_gaussian(data)
mu_male, var_male, pi_male = fit_gaussian(data, male)
mu_female, var_female, pi_female = fit_gaussian(data, female)

# Shared full matrix
classes = {
    male: (mu_male, var_all, pi_male),
    female: (mu_female, var_all, pi_female)
}
print("Shared full:", BIC(data, classes, 2 * (2 + 1) + 4))

# Independent full matrices
classes = {
    male: (mu_male, var_male, pi_male),
    female: (mu_female, var_female, pi_female)
}
print("Independent full:", BIC(data, classes, 2 * (4 + 2 + 1)))
