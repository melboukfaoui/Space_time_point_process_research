"""Space-time Hawkes process: simulate it, fit it by maximum likelihood, and check the fit.

Events are (t, x, y): a time in [0, T] and a location in the rectangle W = [xmin, xmax] x [ymin, ymax].

Model (conditional intensity = event rate per unit time per unit area, given the past):

    lambda(t, x, y) = mu + sum over past events j of
                      K * beta * exp(-beta (t - t_j))                                   <- when
                        * exp(-((x - x_j)^2 + (y - y_j)^2) / (2 sigma^2)) / (2 pi sigma^2)   <- where

    mu     background rate per unit time per unit area
    K      branching ratio: each event triggers on average K direct children (need K < 1)
    beta   temporal decay: a child follows its parent after on average 1 / beta
    sigma  spatial spread: a child lands about sigma away from its parent in each coordinate

Steps:
    1. Simulate with Ogata's thinning algorithm (extended to space).
    2. Fit (mu, K, beta, sigma) by maximum likelihood.
    3. Standard errors and 95% confidence intervals from the Hessian.
    4. Double-check the fit: expected count = observed count, and profile curves.
    5. Likelihood-ratio test against a homogeneous space-time Poisson process.
    6. Goodness of fit with the time-rescaling theorem.
    7. Plots.

Run:  uv run python scripts/hawkes_spacetime.py
"""

import os  # to create the output folder

import matplotlib.pyplot as plt  # plotting
import numpy as np  # arrays and maths
from scipy.optimize import minimize  # numerical optimiser
from scipy.stats import chi2, kstest, norm  # chi-squared (LR test), KS test, normal CDF

PARAM_NAMES = ["mu", "K", "beta", "sigma"]  # labels for results


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------
def intensity(t, x, y, past, mu, K, beta, sigma):
    """lambda(t, x, y) given `past`, an (m, 3) array of earlier events (t_j, x_j, y_j)."""
    dt = t - past[:, 0]  # time since each past event
    d2 = (x - past[:, 1]) ** 2 + (y - past[:, 2]) ** 2  # squared distance to each past event
    when = beta * np.exp(-beta * dt)  # Exp(beta) density: integrates to 1 over time
    where = np.exp(-d2 / (2 * sigma**2)) / (2 * np.pi * sigma**2)  # 2-D Gaussian: integrates to 1 over the plane
    return mu + np.sum(K * when * where)  # background + K times each event's (time x space) bump


def spatial_share(x, y, sigma, W):
    """Share of a Gaussian bump centred at (x, y) that lies inside the rectangle W.

    The 2-D Gaussian has independent x and y parts, so the probability of the rectangle is
    P(xmin < X < xmax) * P(ymin < Y < ymax), each a difference of normal CDFs.
    """
    xmin, xmax, ymin, ymax = W
    px = norm.cdf((xmax - x) / sigma) - norm.cdf((xmin - x) / sigma)  # share inside in x
    py = norm.cdf((ymax - y) / sigma) - norm.cdf((ymin - y) / sigma)  # share inside in y
    return px * py


# ---------------------------------------------------------------------------
# 1. Simulation: thinning in space and time
# ---------------------------------------------------------------------------
def simulate_hawkes_st(mu, K, beta, sigma, T, W, seed=0):
    """Simulate events in [0, T] x W.

    Treat the process as: background events (rate mu*|W|, uniform location in W) plus, for each
    past event j, children at rate K*beta*exp(-beta(t - t_j)) with a Gaussian location around
    (x_j, y_j). Summed over everything, the rate of candidates is

        R(t) = mu*|W| + sum_j K*beta*exp(-beta(t - t_j)),

    which only decreases between events, so R(now) bounds R until the next event.
    1) Propose a candidate time from a Poisson clock with rate R(now).
    2) Keep it with probability R(candidate time) / R(now)   (ordinary thinning in time).
    3) Decide who "caused" it: background with probability mu|W| / R, event j with probability
       K*beta*exp(-beta(t - t_j)) / R. Draw its location from that source.
    4) If the location falls outside W, it is not observed: drop it.
    Steps 2-4 together accept a point at (t, x, y) at exactly the rate lambda(t, x, y).
    """
    if not 0 <= K < 1:  # K >= 1: every event has >= 1 child on average,
        raise ValueError("need 0 <= K < 1")  # so the process explodes
    rng = np.random.default_rng(seed)  # seeded random numbers: reproducible
    xmin, xmax, ymin, ymax = W  # unpack the window
    area = (xmax - xmin) * (ymax - ymin)  # |W|
    events = []  # accepted events (t, x, y)
    t = 0.0  # the clock

    def candidate_rate(s, past):  # R(s): total rate of candidates at time s
        return mu * area + np.sum(K * beta * np.exp(-beta * (s - past[:, 0])))

    while True:
        past = np.array(events).reshape(-1, 3)  # history as an (m, 3) array (works when empty too)
        R_bar = candidate_rate(t, past)  # bound: R only decays until the next event
        t += rng.exponential(1 / R_bar)  # next candidate time: Exp(R_bar) waiting time
        if t > T:  # past the end of the window:
            break  # stop
        R_t = candidate_rate(t, past)  # true candidate rate at that time
        if rng.uniform() >= R_t / R_bar:  # step 2: reject with probability 1 - R_t / R_bar
            continue
        weights = np.concatenate([[mu * area], K * beta * np.exp(-beta * (t - past[:, 0]))])  # each source's rate
        source = rng.choice(len(weights), p=weights / weights.sum())  # step 3: pick a source
        if source == 0:  # background:
            x, y = rng.uniform(xmin, xmax), rng.uniform(ymin, ymax)  # uniform location in W
        else:  # child of past event number source-1:
            x, y = past[source - 1, 1:] + rng.normal(0, sigma, 2)  # parent location + N(0, sigma^2) in x and y
        if xmin <= x <= xmax and ymin <= y <= ymax:  # step 4: keep only if it lands inside W
            events.append((t, x, y))
    return np.array(events)  # (n, 3) array, already sorted by time


# ---------------------------------------------------------------------------
# 2. Log-likelihood
# ---------------------------------------------------------------------------
def log_likelihood(params, ev, T, W):
    """sum_i log lambda(t_i, x_i, y_i)  -  integral over [0, T] x W of lambda."""
    mu, K, beta, sigma = params  # unpack the four parameters
    t, x, y = ev.T  # the three columns
    xmin, xmax, ymin, ymax = W
    area = (xmax - xmin) * (ymax - ymin)
    # First term: log intensity at each event, using only earlier events ev[:i] (sorted by time).
    log_lam = sum(np.log(intensity(t[i], x[i], y[i], ev[:i], mu, K, beta, sigma)) for i in range(len(t)))
    # Second term (compensator) = expected number of events in [0, T] x W:
    #   background: mu * |W| * T
    #   event j:    K * (1 - exp(-beta (T - t_j)))  [time share before T]
    #                 * spatial_share(x_j, y_j)      [space share inside W]
    compensator = mu * area * T + K * np.sum((1 - np.exp(-beta * (T - t))) * spatial_share(x, y, sigma, W))
    return log_lam - compensator


def numerical_hessian(f, z, h=1e-4):
    """Second derivatives of f at z by central finite differences."""
    k = len(z)
    H = np.zeros((k, k))
    for a in range(k):
        for b in range(k):
            ea, eb = np.eye(k)[a] * h, np.eye(k)[b] * h  # small steps in directions a and b
            H[a, b] = (f(z + ea + eb) - f(z + ea - eb) - f(z - ea + eb) + f(z - ea - eb)) / (4 * h * h)
    return H


# ---------------------------------------------------------------------------
# 3. Fitting
# ---------------------------------------------------------------------------
def fit_hawkes_st(ev, T, W):
    """Maximum-likelihood estimates, standard errors and 95% confidence intervals."""
    n = len(ev)
    xmin, xmax, ymin, ymax = W
    area = (xmax - xmin) * (ymax - ymin)
    start = np.log([0.5 * n / (area * T), 0.5, 1.0, 1.0])  # starting guess on the log scale

    def f(log_params):  # -loglik as a function of log-parameters (keeps parameters positive)
        return -log_likelihood(np.exp(log_params), ev, T, W)

    result = minimize(f, start, method="Nelder-Mead",
                      options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 5000})
    log_hat = result.x  # best log-parameters
    estimate = np.exp(log_hat)  # back to the original scale
    cov_log = np.linalg.inv(numerical_hessian(f, log_hat))  # covariance = inverse observed information
    se_log = np.sqrt(np.diag(cov_log))  # SEs of the log-parameters
    se = estimate * se_log  # delta method: SE(p) ~ p * SE(log p)
    ci_low, ci_high = np.exp(log_hat - 1.96 * se_log), np.exp(log_hat + 1.96 * se_log)  # 95% CIs
    return estimate, se, ci_low, ci_high, -result.fun, result.success


# ---------------------------------------------------------------------------
# 6. Residuals (time-rescaling theorem)
# ---------------------------------------------------------------------------
def rescaled_gaps(ev, mu, K, beta, sigma, W):
    """Gaps between Lambda(t_i) = integral over [0, t_i] x W of lambda. Should be Exp(1)."""
    t, x, y = ev.T
    xmin, xmax, ymin, ymax = W
    area = (xmax - xmin) * (ymax - ymin)
    share = spatial_share(x, y, sigma, W)  # each event's spatial share inside W
    Lam = np.array([
        mu * area * t[i] + K * np.sum((1 - np.exp(-beta * (t[i] - t[:i]))) * share[:i])
        for i in range(len(t))
    ])
    return np.diff(Lam, prepend=0.0)


def close_pairs(ev, dt_max=1.0, d_max=1.0):
    """Number of event pairs that are close in BOTH time (< dt_max) and space (< d_max).

    Self-excitation makes events cluster locally in space-time, so a Hawkes process has many more
    such pairs than a Poisson process with the same number of events.
    """
    t, x, y = ev.T
    i, j = np.triu_indices(len(t), k=1)  # every pair (i, j) with i < j
    near_t = np.abs(t[i] - t[j]) < dt_max  # close in time
    near_s = (x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2 < d_max**2  # close in space
    return int(np.sum(near_t & near_s))


def simulate_poisson_st(rate, T, W, rng):
    """Homogeneous space-time Poisson process: Poisson count, then uniform times and locations."""
    xmin, xmax, ymin, ymax = W
    m = rng.poisson(rate * (xmax - xmin) * (ymax - ymin) * T)
    ev = np.column_stack([rng.uniform(0, T, m), rng.uniform(xmin, xmax, m), rng.uniform(ymin, ymax, m)])
    return ev[np.argsort(ev[:, 0])]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    true = {"mu": 0.05, "K": 0.5, "beta": 1.0, "sigma": 0.3}
    T, W = 100.0, (0.0, 10.0, 0.0, 10.0)
    area = (W[1] - W[0]) * (W[3] - W[2])

    # 1. Simulate
    ev = simulate_hawkes_st(**true, T=T, W=W, seed=1)
    n = len(ev)
    print(f"Simulated {n} events; rough expectation mu|W|T/(1-K) = "
          f"{true['mu'] * area * T / (1 - true['K']):.0f} (a bit less: children can land outside W)\n")

    # 2-3. Fit
    est, se, lo, hi, ll_hawkes, ok = fit_hawkes_st(ev, T, W)
    print(f"{'param':6} {'true':>6} {'estimate':>9} {'std err':>8} {'95% CI':>18}")
    for k, name in enumerate(PARAM_NAMES):
        print(f"{name:6} {true[name]:>6} {est[k]:>9.4f} {se[k]:>8.4f}   [{lo[k]:.4f}, {hi[k]:.4f}]")
    print(f"max log-likelihood = {ll_hawkes:.2f}, optimiser converged: {ok}\n")

    # 4. Double-check: at the MLE, expected count Lambda(T) equals observed count n
    mu_h, K_h, beta_h, sigma_h = est
    Lambda_T = mu_h * area * T + K_h * np.sum((1 - np.exp(-beta_h * (T - ev[:, 0])))
                                              * spatial_share(ev[:, 1], ev[:, 2], sigma_h, W))
    print(f"Check: observed n = {n}, expected Lambda(T) = {Lambda_T:.3f}\n")

    # 5. Likelihood-ratio test vs homogeneous space-time Poisson (K = 0): mu_hat = n / (|W| T)
    ll_poisson = n * np.log(n / (area * T)) - n
    lr = 2 * (ll_hawkes - ll_poisson)
    print(f"LR test vs Poisson: LR = {lr:.1f}, p ~ {chi2.sf(lr, df=3):.2g}")
    print(f"AIC: Poisson = {2 - 2 * ll_poisson:.1f}, Hawkes = {8 - 2 * ll_hawkes:.1f}\n")

    # 6. Goodness of fit
    gaps = rescaled_gaps(ev, *est, W)
    gaps_poisson = np.diff(ev[:, 0] * n / T, prepend=0.0)
    print(f"KS vs Exp(1): Hawkes p = {kstest(gaps, 'expon').pvalue:.3f}, "
          f"Poisson p = {kstest(gaps_poisson, 'expon').pvalue:.2g}")
    print("(time-only check: weak here, because bursts are local in space and get diluted when pooled)\n")

    # 6b. Space-time check: close pairs, observed vs simulated from each fitted model
    # (a "parametric bootstrap": if a model is right, the data should look like its simulations)
    rng = np.random.default_rng(123)
    obs = close_pairs(ev)
    sims_hawkes = [close_pairs(simulate_hawkes_st(*est, T=T, W=W, seed=s)) for s in range(1000, 1019)]
    sims_poisson = [close_pairs(simulate_poisson_st(n / (area * T), T, W, rng)) for _ in range(19)]
    print(f"Close pairs (dt < 1, distance < 1): observed = {obs}")
    print(f"  19 simulations from fitted Hawkes:  {min(sims_hawkes)} to {max(sims_hawkes)}")
    print(f"  19 simulations from fitted Poisson: {min(sims_poisson)} to {max(sims_poisson)}")

    # 7. Plots: events in space, fitted intensity map at the busiest moment, residual QQ plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    sc = axes[0].scatter(ev[:, 1], ev[:, 2], c=ev[:, 0], s=5, cmap="viridis")
    axes[0].set(xlabel="x", ylabel="y", aspect="equal", title="Events (colour = time)")
    fig.colorbar(sc, ax=axes[0], label="time")
    recent = [np.sum((ev[:, 0] > ti - 2) & (ev[:, 0] <= ti)) for ti in ev[:, 0]]  # events in last 2 time units
    t_star = ev[int(np.argmax(recent)), 0] + 0.05  # just after the busiest moment
    gx, gy = np.meshgrid(np.linspace(0, 10, 150), np.linspace(0, 10, 150))
    past = ev[ev[:, 0] < t_star]
    lam_map = np.vectorize(lambda a, b: intensity(t_star, a, b, past, *est))(gx, gy)
    im = axes[1].imshow(lam_map, origin="lower", extent=[0, 10, 0, 10], cmap="magma")
    axes[1].set(xlabel="x", ylabel="y", title=f"Fitted λ(t, x, y) at t = {t_star:.1f}")
    fig.colorbar(im, ax=axes[1], label="intensity")
    q = -np.log(1 - (np.arange(1, n + 1) - 0.5) / n)
    axes[2].plot(q, np.sort(gaps), ".", ms=3, label="Hawkes fit")
    axes[2].plot(q, np.sort(gaps_poisson), ".", ms=3, label="Poisson fit")
    axes[2].plot([0, q.max()], [0, q.max()], c="grey", lw=1)
    axes[2].set(xlabel="Exp(1) quantiles", ylabel="rescaled gaps", title="Residual QQ plot")
    axes[2].legend()
    plt.tight_layout()
    os.makedirs("_output", exist_ok=True)
    plt.savefig("_output/hawkes_spacetime_fit.png", dpi=130)
    print("\nPlot saved to _output/hawkes_spacetime_fit.png")
