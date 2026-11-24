"""Simulate space-time Hawkes data under several scenarios and fit the model to each.

Uses our own thinning simulator and maximum-likelihood fit from scripts/hawkes_spacetime.py.
Each scenario is simulated with a different seed from the fit's starting values, so the fit
has to find the parameters on its own.

Run:  uv run python scripts/fit_simulated_data.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))  # so we can import the neighbouring script
from hawkes_spacetime import PARAM_NAMES, fit_hawkes_st, simulate_hawkes_st

W = (0.0, 10.0, 0.0, 10.0)  # observation window (10 x 10 square)
T = 100.0  # time horizon
SCENARIOS = {  # true parameters for each simulated dataset
    "A": {"mu": 0.05, "K": 0.5, "beta": 1.0, "sigma": 0.3, "label": "moderate clustering"},
    "B": {"mu": 0.03, "K": 0.7, "beta": 2.0, "sigma": 0.5, "label": "strong clustering"},
    "C": {"mu": 0.10, "K": 0.2, "beta": 0.5, "sigma": 0.2, "label": "weak clustering"},
}

rows = []
for seed, (name, scenario) in enumerate(SCENARIOS.items(), start=2026):
    truth = {p: scenario[p] for p in PARAM_NAMES}  # just the four parameters
    ev = simulate_hawkes_st(**truth, T=T, W=W, seed=seed)  # 1. simulate (t, x, y) events
    est, se, lo, hi, ll, ok = fit_hawkes_st(ev, T, W)  # 2. fit by maximum likelihood
    print(f"\nScenario {name} ({scenario['label']}): {len(ev)} events, converged = {ok}")
    print(f"  {'param':6} {'true':>6} {'estimate':>9} {'std err':>8} {'95% CI':>20}  covers truth")
    for k, p in enumerate(PARAM_NAMES):
        covered = lo[k] <= truth[p] <= hi[k]  # 3. is the truth inside the 95% CI?
        print(f"  {p:6} {truth[p]:>6} {est[k]:>9.4f} {se[k]:>8.4f}   [{lo[k]:.4f}, {hi[k]:.4f}]  "
              f"{'yes' if covered else 'NO'}")
        rows.append({"scenario": name, "param": p, "z": (est[k] - truth[p]) / se[k], "covered": covered})

summary = pd.DataFrame(rows)
print(f"\n95% CIs covering the truth: {summary.covered.sum()} of {len(summary)}")
print("z = (estimate - true) / std err; roughly within +/-2 if the fit is right:")
print(summary.pivot(index="param", columns="scenario", values="z").round(2).loc[PARAM_NAMES])
