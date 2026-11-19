"""Simulators for space-time point processes."""

from __future__ import annotations

import numpy as np


def simulate_poisson(
    rate: float,
    T: float,
    bounds: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0),
    seed: int | None = None,
) -> np.ndarray:
    """Simulate a homogeneous space-time Poisson process.

    Parameters
    ----------
    rate : expected number of events per unit time per unit area.
    T : length of the time window [0, T].
    bounds : spatial window (xmin, xmax, ymin, ymax).
    seed : random seed.

    Returns
    -------
    Array of shape (n, 3) with columns (t, x, y), sorted by time.
    """
    rng = np.random.default_rng(seed)
    xmin, xmax, ymin, ymax = bounds
    area = (xmax - xmin) * (ymax - ymin)
    n = rng.poisson(rate * T * area)
    t = np.sort(rng.uniform(0.0, T, n))
    x = rng.uniform(xmin, xmax, n)
    y = rng.uniform(ymin, ymax, n)
    return np.column_stack([t, x, y])
