import numpy as np

from stpp import simulate_poisson


def test_poisson_shape_and_bounds():
    events = simulate_poisson(rate=50, T=10, bounds=(0, 2, 0, 1), seed=0)
    assert events.shape[1] == 3
    t, x, y = events.T
    assert np.all(np.diff(t) >= 0)
    assert t.min() >= 0 and t.max() <= 10
    assert x.min() >= 0 and x.max() <= 2
    assert y.min() >= 0 and y.max() <= 1


def test_poisson_mean_count():
    counts = [len(simulate_poisson(rate=20, T=5, seed=s)) for s in range(200)]
    assert abs(np.mean(counts) - 100) < 5
