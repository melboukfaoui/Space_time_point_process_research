"""Space-time Hawkes process: simulation (branching algorithm) and maximum-likelihood fitting.

Model (conditional intensity):

    lambda(t, x, y) = mu + sum_{j: t_j < t} K * beta * exp(-beta (t - t_j))
                                         * exp(-((x - x_j)^2 + (y - y_j)^2) / (2 sigma^2)) / (2 pi sigma^2)

mu    background events per unit time per unit area
K     expected number of direct children per event (branching ratio, needs K < 1)
beta  temporal decay rate (mean parent-to-child delay is 1 / beta)
sigma spatial spread of children around their parent (Gaussian, per coordinate)

The simulator is a Python port of R/simulate_hawkes_clustering.R.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

PARAM_NAMES = ("mu", "K", "beta", "sigma")


def simulate_hawkes(
    tmax: float,
    W: tuple[float, float, float, float],
    mu: float,
    K: float,
    beta: float,
    sigma: float,
    seed: int | None = None,
    clip: bool = True,
) -> np.ndarray:
    """Simulate a space-time Hawkes process with the branching (cluster) algorithm.

    Returns an array of shape (n, 3) with columns (t, x, y), sorted by time.
    If ``clip`` is True, only events inside [0, tmax] x W are returned (the observed data);
    if False, children that fall after tmax or outside W are kept, as in the R version.
    """
    if not 0 <= K < 1:
        raise ValueError("K must be in [0, 1); otherwise the cascade never dies out")
    rng = np.random.default_rng(seed)
    xmin, xmax, ymin, ymax = W
    area = (xmax - xmin) * (ymax - ymin)

    # Generation 0: background ("immigrant") events, a homogeneous Poisson process
    n_bg = rng.poisson(mu * area * tmax)
    gen_t = rng.uniform(0, tmax, n_bg)
    gen_x = rng.uniform(xmin, xmax, n_bg)
    gen_y = rng.uniform(ymin, ymax, n_bg)
    all_t, all_x, all_y = [gen_t], [gen_x], [gen_y]

    # Each later generation: every event of the previous one has Poisson(K) children
    while gen_t.size > 0:
        n_children = rng.poisson(K, gen_t.size)
        par_t = np.repeat(gen_t, n_children)
        par_x = np.repeat(gen_x, n_children)
        par_y = np.repeat(gen_y, n_children)
        m = par_t.size
        gen_t = par_t + rng.exponential(1 / beta, m)
        gen_x = par_x + rng.normal(0, sigma, m)
        gen_y = par_y + rng.normal(0, sigma, m)
        all_t.append(gen_t)
        all_x.append(gen_x)
        all_y.append(gen_y)

    events = np.column_stack([np.concatenate(all_t), np.concatenate(all_x), np.concatenate(all_y)])
    if clip:
        t, x, y = events.T
        inside = (t <= tmax) & (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
        events = events[inside]
    return events[np.argsort(events[:, 0])]


def _pairs(events: np.ndarray):
    """All (i, j) with j earlier than i, plus their time gaps and squared distances."""
    t, x, y = events.T
    i, j = np.tril_indices(len(t), k=-1)
    dt = t[i] - t[j]
    d2 = (x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2
    return i, dt, d2


def neg_log_likelihood(theta, events, tmax, W, pairs=None) -> float:
    """Negative log-likelihood. ``theta`` is log(mu, K, beta, sigma)."""
    mu, K, beta, sigma = np.exp(theta)
    t, x, y = events.T
    n = len(t)
    xmin, xmax, ymin, ymax = W
    i, dt, d2 = pairs if pairs is not None else _pairs(events)

    # Part 1: sum of log intensities at the observed events
    g = K * beta * np.exp(-beta * dt) * np.exp(-d2 / (2 * sigma**2)) / (2 * np.pi * sigma**2)
    lam = mu + np.bincount(i, weights=g, minlength=n)
    log_lam_sum = np.sum(np.log(lam))

    # Part 2: compensator = integral of lambda over [0, tmax] x W
    time_mass = 1 - np.exp(-beta * (tmax - t))
    px = norm.cdf((xmax - x) / sigma) - norm.cdf((xmin - x) / sigma)
    py = norm.cdf((ymax - y) / sigma) - norm.cdf((ymin - y) / sigma)
    area = (xmax - xmin) * (ymax - ymin)
    compensator = mu * area * tmax + K * np.sum(time_mass * px * py)

    return -(log_lam_sum - compensator)


def _numerical_hessian(f, theta, h=1e-4):
    k = len(theta)
    H = np.zeros((k, k))
    for a in range(k):
        for b in range(a, k):
            ea, eb = np.eye(k)[a] * h, np.eye(k)[b] * h
            H[a, b] = H[b, a] = (
                f(theta + ea + eb) - f(theta + ea - eb) - f(theta - ea + eb) + f(theta - ea - eb)
            ) / (4 * h * h)
    return H


@dataclass
class HawkesFit:
    estimate: dict[str, float]
    std_error: dict[str, float]
    log_likelihood: float
    converged: bool

    def summary(self, truth: dict[str, float] | None = None) -> str:
        lines = [f"{'param':>6} {'estimate':>10} {'std err':>10}" + (f" {'true':>8}" if truth else "")]
        for p in PARAM_NAMES:
            row = f"{p:>6} {self.estimate[p]:>10.4f} {self.std_error[p]:>10.4f}"
            if truth:
                row += f" {truth[p]:>8.4f}"
            lines.append(row)
        lines.append(f"log-likelihood = {self.log_likelihood:.2f}, converged = {self.converged}")
        return "\n".join(lines)


def fit_hawkes(events, tmax, W, start=(0.1, 0.5, 1.0, 1.0)) -> HawkesFit:
    """Maximum-likelihood fit of (mu, K, beta, sigma)."""
    pairs = _pairs(events)

    def f(theta):
        return neg_log_likelihood(theta, events, tmax, W, pairs)

    res = minimize(f, np.log(start), method="BFGS")
    est = np.exp(res.x)
    cov_log = np.linalg.inv(_numerical_hessian(f, res.x))
    se = est * np.sqrt(np.diag(cov_log))  # delta method: SE(p) = p * SE(log p)
    return HawkesFit(
        estimate=dict(zip(PARAM_NAMES, est)),
        std_error=dict(zip(PARAM_NAMES, se)),
        log_likelihood=-res.fun,
        converged=bool(res.success),
    )
