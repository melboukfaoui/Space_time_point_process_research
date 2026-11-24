import numpy as np
import pytest

from stpp.hawkes import fit_hawkes, neg_log_likelihood, simulate_hawkes

W = (0, 10, 0, 10)
TRUTH = {"mu": 0.05, "K": 0.5, "beta": 1.0, "sigma": 0.3}


def test_simulation_is_inside_window_and_sorted():
    ev = simulate_hawkes(100, W, **TRUTH, seed=1)
    t, x, y = ev.T
    assert np.all(np.diff(t) >= 0)
    assert t.max() <= 100 and x.min() >= 0 and x.max() <= 10
    assert y.min() >= 0 and y.max() <= 10


def test_expected_count():
    # E[N] is about mu * |W| * T / (1 - K) = 1000, a little less because of edge losses
    counts = [len(simulate_hawkes(100, W, **TRUTH, seed=s)) for s in range(30)]
    assert 800 < np.mean(counts) < 1000


def test_supercritical_rejected():
    with pytest.raises(ValueError):
        simulate_hawkes(10, W, mu=0.1, K=1.2, beta=1, sigma=0.3)


def test_no_excitation_reduces_to_poisson_likelihood():
    ev = simulate_hawkes(50, W, mu=0.1, K=0.0, beta=1, sigma=0.3, seed=2)
    theta = np.log([0.1, 1e-12, 1.0, 0.3])
    n = len(ev)
    poisson_ll = n * np.log(0.1) - 0.1 * 100 * 50
    assert -neg_log_likelihood(theta, ev, 50, W) == pytest.approx(poisson_ll, rel=1e-6)


def test_fit_recovers_parameters():
    ev = simulate_hawkes(100, W, **TRUTH, seed=0)
    fit = fit_hawkes(ev, 100, W)
    assert fit.converged
    for p, true in TRUTH.items():
        assert abs(fit.estimate[p] - true) < 4 * fit.std_error[p], p
