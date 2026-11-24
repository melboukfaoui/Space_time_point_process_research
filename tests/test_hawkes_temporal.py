import numpy as np
import pytest
from scipy.stats import kstest

from stpp.hawkes_temporal import (
    _recursion,
    compensator_at_events,
    fit_hawkes_temporal,
    neg_log_likelihood,
    neg_log_likelihood_and_grad,
    simulate_hawkes_thinning,
)

TRUTH = {"mu": 0.5, "alpha": 0.6, "beta": 2.0}
T = 2000.0


def test_mean_count_matches_theory():
    # Stationary mean rate is mu / (1 - alpha) = 1.25, so E[N] is about 2500
    counts = [len(simulate_hawkes_thinning(**TRUTH, T=T, seed=s)) for s in range(20)]
    assert abs(np.mean(counts) / (TRUTH["mu"] * T / (1 - TRUTH["alpha"])) - 1) < 0.05


def test_recursion_matches_brute_force():
    times = np.sort(np.random.default_rng(0).uniform(0, 10, 50))
    brute = np.array([np.sum(np.exp(-1.3 * (t - times[times < t]))) for t in times])
    assert np.allclose(_recursion(times, 1.3), brute)


def test_gradient_matches_finite_differences():
    times = simulate_hawkes_thinning(**TRUTH, T=200, seed=3)
    theta = np.log([0.4, 0.5, 1.5])
    _, grad = neg_log_likelihood_and_grad(theta, times, 200)
    h = 1e-6
    numeric = [
        (neg_log_likelihood(theta + h * e, times, 200) - neg_log_likelihood(theta - h * e, times, 200))
        / (2 * h)
        for e in np.eye(3)
    ]
    assert np.allclose(grad, numeric, rtol=1e-5)


def test_compensator_matches_brute_force():
    times = simulate_hawkes_thinning(**TRUTH, T=100, seed=4)
    brute = [0.5 * t + 0.6 * np.sum(1 - np.exp(-2.0 * (t - times[times < t]))) for t in times]
    assert np.allclose(compensator_at_events(times, **TRUTH), brute)


def test_poisson_special_case():
    times = np.sort(np.random.default_rng(1).uniform(0, 100, 80))
    ll = -neg_log_likelihood(np.log([0.8, 1e-12, 1.0]), times, 100)
    assert ll == pytest.approx(80 * np.log(0.8) - 0.8 * 100, rel=1e-6)


def test_fit_recovers_parameters_and_residuals_are_exponential():
    times = simulate_hawkes_thinning(**TRUTH, T=T, seed=0)
    fit = fit_hawkes_temporal(times, T)
    assert fit.converged
    for p, true in TRUTH.items():
        assert abs(fit.estimate[p] - true) < 4 * fit.std_error[p], p
    residuals = np.diff(compensator_at_events(times, **fit.estimate), prepend=0.0)
    assert kstest(residuals, "expon").pvalue > 0.01
