import numpy as np


def get_mle(data):
    return data.sum(0) / data.shape[0]


def likelihood(x, theta):
    return np.prod((x * theta) + (1 - x) * (1 - theta))


D_politics = np.array([
    [1, 0, 1, 1, 1, 0, 1, 1],
    [0, 0, 0, 1, 0, 0, 1, 1],
    [1, 0, 0, 1, 1, 0, 1, 0],
    [0, 1, 0, 0, 1, 1, 0, 1],
    [0, 0, 0, 1, 1, 0, 1, 1],
    [0, 0, 0, 1, 1, 0, 0, 1],
])

D_sport = np.array([
    [1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [1, 1, 0, 1, 0, 0, 0, 0],
    [1, 1, 0, 1, 0, 0, 0, 1],
    [1, 1, 0, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 1, 0, 0],
    [1, 1, 1, 1, 1, 0, 1, 0],
])

x = np.array([1, 0, 0, 1, 1, 1, 1, 0])


theta_politics = get_mle(D_politics)
theta_sport = get_mle(D_sport)

politics = likelihood(x, theta_politics) * (6 / 13)
sport = likelihood(x, theta_sport) * (7 / 13)

print(politics / (politics + sport))
