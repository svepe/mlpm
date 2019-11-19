import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


data = genfromtxt("data/heightWeightData.csv", delimiter=",")
data = data[1:]
ys = data[:, 0]
xs = data[:, 1:]
N = data.shape[0]
D = xs.shape[1]

male = 1
female = 2


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def plot_map(w_map, nll):
    plt.gcf().suptitle("p(female | height, weight)")
    plt.gca().clear()
    plt.title("NLL = {}".format(nll))
    plt.xlabel("Height (inches)")
    plt.ylabel("Weight (lbs)")

    x1, x2 = np.meshgrid(
        np.linspace(50, 80, 100),
        np.linspace(80, 300, 100),
    )

    X = np.stack((x1, x2, np.ones_like(x2)), axis=2).reshape(-1, 3)
    p = sigmoid(X @ w_map).reshape(100, 100)

    cf = plt.contourf(x1, x2, p, 100, cmap="jet")

    plt.scatter(
        xs[ys == male, 0],
        xs[ys == male, 1],
        marker="o",
        color="white",
        label="male"
    )

    plt.scatter(
        xs[ys == female, 0],
        xs[ys == female, 1],
        marker="v",
        color="black",
        label="female"
    )
    plt.legend()

    plt.draw()
    plt.pause(1e-3)


y = ys[:, np.newaxis] - 1
X = np.hstack((xs, np.ones((N, 1))))
w = np.array([[-0.1], [0.2], [0.2]])
# w = np.random.randn(D + 1, 1)


for iter in range(1000):
    mu = sigmoid(X @ w)

    nll = - y.T @ np.log(mu) - (1 - y).T @ np.log(1 - mu)

    g = X.T @ (mu - y)
    w -= 1e-6 * g

    plot_map(w, np.asscalar(nll))

plt.show()
