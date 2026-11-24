"""Temporal Hawkes process with an exponential kernel.

Simulation by Ogata's thinning algorithm, fitting by maximum likelihood with the O(n)
recursion, and goodness of fit by the time-rescaling theorem.

Model (conditional intensity):

    lambda(t) = mu + sum_{t_j < t} alpha * beta * exp(-beta (t - t_j))

mu     background rate (events per unit time)
alpha  branching ratio: expected number of direct children per event (needs alpha < 1)
beta   decay rate: a child arrives on average 1 / beta after its parent
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from stpp.hawkes import _numerical_hessian

PARAM_NAMES = ("mu", "alpha", "beta")


def simulate_hawkes_thinning(
    mu: float, alpha: float, beta: float, T: float, seed: int | None = None
) -> np.ndarray:
    """Simulate event times on [0, T] by Ogata's thinning algorithm."""
    if not 0 <= alpha < 1:
        raise ValueError("alpha must be in [0, 1) for a stationary process")
    rng = np.random.default_rng(seed)
    events = []
    t = 0.0
    excitation = 0.0  # sum over past events of alpha*beta*exp(-beta (t - t_j)), at the current t

    while True:
        lam_bar = mu + excitation  # the intensity only decays until the next event: an upper bound
        w = rng.exponential(1 / lam_bar)  # candidate waiting time from a Poisson(lam_bar) clock
        t += w
        if t > T:
            break
        excitation *= np.exp(-beta * w)  # let the excitation decay to the candidate time
        lam_t = mu + excitation  # the true intensity there
        if rng.uniform() * lam_bar <= lam_t:  # accept with probability lam_t / lam_bar
            events.append(t)
            excitation += alpha * beta  # the new event adds its own jump
    return np.array(events)


def _recursion(times: np.ndarray, beta: float, with_derivative: bool = False):
    """A_i = sum_{j<i} exp(-beta (t_i - t_j)) in O(n).

    With ``with_derivative``, also B_i = dA_i/dbeta = -sum_{j<i} (t_i - t_j) exp(-beta (t_i - t_j)).
    """
    n = len(times)
    A = np.zeros(n)
    B = np.zeros(n)
    gaps = np.diff(times)
    decay = np.exp(-beta * gaps)
    for i in range(1, n):
        A[i] = decay[i - 1] * (1.0 + A[i - 1])
        if with_derivative:
            B[i] = decay[i - 1] * (B[i - 1] - gaps[i - 1] * (1.0 + A[i - 1]))
    return (A, B) if with_derivative else A


def neg_log_likelihood(theta: np.ndarray, times: np.ndarray, T: float) -> float:
    """Negative log-likelihood; theta = log(mu, alpha, beta)."""
    return neg_log_likelihood_and_grad(theta, times, T)[0]


def neg_log_likelihood_and_grad(theta: np.ndarray, times: np.ndarray, T: float):
    """Negative log-likelihood and its exact gradient with respect to theta = log(mu, alpha, beta)."""
    mu, alpha, beta = np.exp(theta)
    A, B = _recursion(times, beta, with_derivative=True)
    lam = mu + alpha * beta * A
    tail = np.exp(-beta * (T - times))
    ll = np.sum(np.log(lam)) - mu * T - alpha * np.sum(1.0 - tail)

    d_mu = np.sum(1.0 / lam) - T
    d_alpha = np.sum(beta * A / lam) - np.sum(1.0 - tail)
    d_beta = np.sum(alpha * (A + beta * B) / lam) - alpha * np.sum((T - times) * tail)
    grad_log = np.array([mu * d_mu, alpha * d_alpha, beta * d_beta])  # chain rule: d/dlog(p) = p d/dp
    return -ll, -grad_log


def compensator_at_events(times: np.ndarray, mu: float, alpha: float, beta: float) -> np.ndarray:
    """Lambda(t_i) = integral of lambda from 0 to t_i, for each event, in O(n).

    Lambda(t_i) = mu t_i + alpha * sum_{j<i} (1 - exp(-beta (t_i - t_j))) = mu t_i + alpha (i - A_i),
    where i (0-based) is the number of earlier events.
    """
    A = _recursion(times, beta)
    return mu * times + alpha * (np.arange(len(times)) - A)


@dataclass
class TemporalHawkesFit:
    estimate: dict[str, float]
    std_error: dict[str, float]
    log_likelihood: float
    converged: bool

    def summary(self, truth: dict[str, float] | None = None) -> str:
        head = f"{'param':>6} {'estimate':>10} {'std err':>10}" + (f" {'true':>8}" if truth else "")
        lines = [head]
        for p in PARAM_NAMES:
            row = f"{p:>6} {self.estimate[p]:>10.4f} {self.std_error[p]:>10.4f}"
            if truth:
                row += f" {truth[p]:>8.4f}"
            lines.append(row)
        lines.append(f"log-likelihood = {self.log_likelihood:.2f}, converged = {self.converged}")
        return "\n".join(lines)


def fit_hawkes_temporal(
    times: np.ndarray, T: float, start: tuple[float, float, float] | None = None
) -> TemporalHawkesFit:
    """Maximum-likelihood fit of (mu, alpha, beta)."""
    if start is None:
        start = (0.5 * len(times) / T, 0.5, 1.0)  # half the events as background is a neutral guess

    def f(theta):
        return neg_log_likelihood(theta, times, T)

    def f_and_grad(theta):
        return neg_log_likelihood_and_grad(theta, times, T)

    res = minimize(f_and_grad, np.log(start), jac=True, method="BFGS")
    est = np.exp(res.x)
    cov_log = np.linalg.inv(_numerical_hessian(f, res.x))
    se = est * np.sqrt(np.diag(cov_log))
    return TemporalHawkesFit(
        estimate=dict(zip(PARAM_NAMES, est)),
        std_error=dict(zip(PARAM_NAMES, se)),
        log_likelihood=-res.fun,
        converged=bool(res.success),
    )
