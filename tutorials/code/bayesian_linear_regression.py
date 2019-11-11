import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt


from scipy.stats import multivariate_normal as mvn


def plot_mvn(dist, ax):
    w0, w1 = np.meshgrid(
        np.linspace(-1, 1, 100),
        np.linspace(-1, 1, 100)
    )
    ws = np.stack((w0, w1), axis=2).reshape(-1, 2)
    p = dist.pdf(ws).reshape(100, 100)
    ax.contourf(w0, w1, p, 100, cmap="jet")


def plot_likelihood(pt, var, ax):
    w0, w1 = np.meshgrid(
        np.linspace(-1, 1, 100),
        np.linspace(-1, 1, 100)
    )
    ws = np.stack((w0, w1), axis=2).reshape(-1, 2)
    x, y = pt
    p = np.exp(-(y - ws @ [x, 1]) ** 2 / (2 * var)).reshape(100, 100)
    p /= np.sqrt(2 * np.pi * var)
    ax.contourf(w0, w1, p, 100, cmap="jet")


def plot_line_samples(dist, n, ax):
    ws = dist.rvs(n)
    left_ys = ws @ [-1, 1]
    right_ys = ws @ [1, 1]
    for left_y, right_y in zip(left_ys, right_ys):
        ax.plot([-1, 1], [left_y, right_y], "k")


def plot_data_points(pts, ax):
    pts = np.array(pts)
    ax.plot(pts[:, 0], pts[:, 1], "rx")


def plot_update(pts, prior, posterior, noise):
    global fig, axs
    plot_mvn(prior, axs[2])
    plot_likelihood(pts[-1], noise, axs[1])
    plot_mvn(posterior, axs[0])

    init_axis(axs[3], "Data Space")
    plot_line_samples(posterior, 10, axs[3])
    plot_data_points(pts, axs[3])

    fig.canvas.draw()


def calculate_posterior(pt, noise, prior):
    x = np.array([[pt[0], 1]])
    y = np.array([pt[1]])
    w0 = prior.mean
    V0 = prior.cov
    # eq. 7.58
    Vn = noise * inv(noise * inv(V0) + x.T @ x)
    # eq. 7.56
    wn = Vn @ inv(V0) @ w0 + Vn @ x.T @ y / noise
    return mvn(mean=wn, cov=Vn)


def update(event):
    global axs, pts, prior, posterior, noise

    if event.inaxes != axs[3]:
        return

    pts.append([event.xdata, event.ydata])

    posterior = calculate_posterior(pts[-1], noise, prior)

    plot_update(pts, prior, posterior, noise)

    prior = posterior


def init_axis(ax, title):
    ax.clear()
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_aspect("equal")
    ax.set_title(title)


def init_figure():
    fig = plt.figure("Bayesian Linear Regression")
    axs = (
        fig.add_subplot(141),
        fig.add_subplot(142),
        fig.add_subplot(143),
        fig.add_subplot(144)
    )

    init_axis(axs[0], "Posterior")
    init_axis(axs[1], "Likelihood")
    init_axis(axs[2], "Prior")
    init_axis(axs[3], "Data Space")
    return fig, axs


if __name__ == "__main__":
    pts = []
    noise = 0.2 ** 2
    prior = mvn(mean=np.zeros(2), cov=np.eye(2) / 2.0)
    posterior = mvn(mean=np.zeros(2), cov=np.eye(2) / 2.0)

    fig, axs = init_figure()
    plot_mvn(prior, axs[2])
    plot_line_samples(prior, 10, axs[3])

    fig.canvas.mpl_connect("button_press_event", update)
    plt.show()
