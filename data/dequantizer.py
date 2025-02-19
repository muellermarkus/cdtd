import numpy as np
import scipy.stats as stats


class Dequantizer:
    def __init__(self, eps=1e-7):
        self.left_edges_y = {}
        self.left_edges_x = {}
        self.right_edges_y = {}
        self.right_edges_x = {}
        self.slopes = {}
        self.eps = eps

    def fit(self, x):
        assert len(x.shape) == 2
        # for each column in x, get bounds necessary to do dequantization
        for i in range(x.shape[1]):
            self.get_edges(x[:, i], i)

    def transform(self, x):
        output = []

        for i in range(x.shape[1]):
            z = x[:, i]
            idx = np.searchsorted(self.left_edges_x[i], z, side="right")
            idx = idx - 1
            idx[idx < 0] = 0

            # clip inputs to learned domain
            inp = z.copy()
            inp[idx == 0] = self.left_edges_x[i][0]
            inp[idx == len(self.left_edges_x[i]) - 1] = self.left_edges_x[i][-1]

            left_x = np.take(self.left_edges_x[i], idx, axis=0)
            right_x = np.take(self.right_edges_x[i], idx, axis=0)

            # add uniform noise in [0, next value)
            inp = inp + np.random.rand(inp.shape[0]) * (right_x - inp)

            # linearly interpolate edges
            slope = np.take(self.slopes[i], idx, axis=0)
            left_y = np.take(self.left_edges_y[i], idx, axis=0)
            interpolation = left_y + (inp - left_x) * slope

            # transform to normal distribution, avoiding values too close to 0.0 and 1.0
            y = stats.norm.ppf(interpolation)
            clip_min = stats.norm.ppf(self.eps - np.spacing(1))
            clip_max = stats.norm.ppf(1 - (self.eps - np.spacing(1)))
            y = np.clip(y, clip_min, clip_max)
            output.append(y)

        return np.stack(output, 1)

    def inverse_transform(self, x):
        output = []
        for i in range(x.shape[1]):
            y = stats.norm.cdf(x[:, i])
            idx = np.searchsorted(self.left_edges_y[i], y, side="right")
            idx = idx - 1
            idx[idx < 0] = 0
            output.append(self.left_edges_x[i][idx])

        return np.stack(output, 1)

    def get_edges(self, x, i):
        z = np.sort(x.copy())
        nobs = len(z)
        vals, counts = np.unique(z, return_counts=True)
        empirical_pdf = counts / nobs

        self.left_edges_x[i] = vals
        self.right_edges_x[i] = np.concatenate((vals[1:], (vals[-1] + 1.0)[np.newaxis]))
        self.right_edges_y[i] = empirical_pdf.cumsum(0)
        self.left_edges_y[i] = np.concatenate(
            (np.zeros((1,)), self.right_edges_y[i][:-1])
        )
        self.slopes[i] = (self.right_edges_y[i] - self.left_edges_y[i]) / (
            self.right_edges_x[i] - self.left_edges_x[i]
        )
