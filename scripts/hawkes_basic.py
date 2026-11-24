"""Hawkes process: simulate it, fit it by maximum likelihood, and check the fit.

Model (conditional intensity = instantaneous event rate given the past):

    lambda(t) = mu + sum over past events t_j < t of  alpha * beta * exp(-beta * (t - t_j))

    mu     background rate: events per unit time that arrive "on their own"
    alpha  branching ratio: each event triggers on average alpha direct children (need alpha < 1)
    beta   decay rate: a triggered event follows its parent after on average 1 / beta time units

Steps:
    1. Simulate event times with Ogata's thinning algorithm.
    2. Fit (mu, alpha, beta) by maximum likelihood.
    3. Standard errors and 95% confidence intervals from the Hessian.
    4. Likelihood-ratio test: is the Hawkes model better than a plain Poisson process?
    5. Goodness of fit with the time-rescaling theorem (residuals should be Exp(1)).
    6. Plot the fitted intensity and the residual QQ plot.

Run:  uv run python scripts/hawkes_basic.py
"""

import os  # to create the output folder

import matplotlib.pyplot as plt  # plotting
import numpy as np  # arrays and maths
from scipy.optimize import minimize  # numerical optimiser for the likelihood
from scipy.stats import chi2, kstest  # chi-squared distribution (LR test) and KS test (residuals)

PARAM_NAMES = ["mu", "alpha", "beta"]  # used to label results


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------
def intensity(t, past, mu, alpha, beta):
    """lambda(t) given the event times `past` that happened before t."""
    # Each past event t_j adds a bump alpha*beta*exp(-beta*(t - t_j)).
    # The bump has total area alpha (beta*exp(-beta*u) integrates to 1), so on average
    # each event triggers alpha children. np.sum adds all bumps; mu is the constant baseline.
    return mu + np.sum(alpha * beta * np.exp(-beta * (t - past)))


# ---------------------------------------------------------------------------
# 1. Simulation: Ogata's thinning algorithm
# ---------------------------------------------------------------------------
def simulate_hawkes(mu, alpha, beta, T, seed=0):
    """Simulate event times on [0, T].

    Idea: propose candidates from a faster Poisson clock with rate lam_bar >= lambda(t),
    and keep each candidate with probability lambda(t) / lam_bar. In a tiny interval dt
    that gives an event with probability lam_bar*dt * lambda(t)/lam_bar = lambda(t)*dt,
    which is exactly the model.
    """
    if not 0 <= alpha < 1:  # alpha >= 1 means each event triggers >= 1 child on average:
        raise ValueError("need 0 <= alpha < 1")  # the process explodes, so refuse it
    rng = np.random.default_rng(seed)  # random number generator; the seed makes runs reproducible
    events = []  # accepted event times
    t = 0.0  # current time on the clock
    while True:
        past = np.array(events)  # events so far, as an array for vectorised maths
        lam_bar = intensity(t, past, mu, alpha, beta)  # rate right now. Between events lambda only
        # decays, so this value is an upper bound until the next accepted event.
        t += rng.exponential(1 / lam_bar)  # waiting time to the next candidate: Exp(lam_bar).
        # NumPy's exponential takes the mean 1/rate.
        if t > T:  # candidate is past the end of the window:
            break  # stop
        lam_t = intensity(t, past, mu, alpha, beta)  # true rate at the candidate time (lower,
        # because the bumps decayed while we waited)
        if rng.uniform() < lam_t / lam_bar:  # accept with probability lam_t / lam_bar
            events.append(t)  # accepted: it is now part of the history and raises lambda
        # If rejected, we loop again from the candidate time with a new (lower) bound.
    return np.array(events)


# ---------------------------------------------------------------------------
# 2. Log-likelihood
# ---------------------------------------------------------------------------
def log_likelihood(params, times, T):
    """Point-process log-likelihood:  sum_i log lambda(t_i)  -  integral_0^T lambda(t) dt."""
    mu, alpha, beta = params  # unpack the three parameters
    # First term: log of the intensity at each event, using only events before it
    # (times[:i] = all events earlier than event i, because times are sorted).
    # It rewards a high intensity where events actually happened.
    log_lam = sum(np.log(intensity(ti, times[:i], mu, alpha, beta)) for i, ti in enumerate(times))
    # Second term (the "compensator"): expected number of events in [0, T].
    #   integral of mu over [0, T]                    = mu * T
    #   integral of event j's bump from t_j to T      = alpha * (1 - exp(-beta*(T - t_j)))
    # (an event near T has had little time to trigger children, so it counts for less).
    # It penalises a high intensity everywhere; without it the fit would make lambda huge.
    compensator = mu * T + alpha * np.sum(1 - np.exp(-beta * (T - times)))
    return log_lam - compensator


def neg_log_likelihood_log_scale(log_params, times, T):
    """What the optimiser minimises. It works with log(parameters)."""
    # Optimising log(mu), log(alpha), log(beta) lets the optimiser move freely over all real
    # numbers, while exp() guarantees the actual parameters stay positive.
    return -log_likelihood(np.exp(log_params), times, T)  # minus sign: optimisers minimise


def numerical_hessian(f, x, h=1e-4):
    """Matrix of second derivatives of f at x, by central finite differences."""
    k = len(x)  # number of parameters
    H = np.zeros((k, k))  # empty k x k matrix
    for a in range(k):
        for b in range(k):
            ea = np.eye(k)[a] * h  # small step in direction a
            eb = np.eye(k)[b] * h  # small step in direction b
            # d^2 f / dx_a dx_b ~ [f(+a+b) - f(+a-b) - f(-a+b) + f(-a-b)] / (4 h^2)
            H[a, b] = (f(x + ea + eb) - f(x + ea - eb) - f(x - ea + eb) + f(x - ea - eb)) / (4 * h * h)
    return H


# ---------------------------------------------------------------------------
# 3. Fitting
# ---------------------------------------------------------------------------
def fit_hawkes(times, T):
    """Maximum-likelihood estimates with standard errors."""
    n = len(times)  # number of events
    start = np.log([0.5 * n / T, 0.5, 1.0])  # starting guess on the log scale. The long-run rate
    # n/T estimates mu/(1-alpha), so if alpha were 0.5, mu would be about 0.5*n/T.

    def f(log_params):  # objective as a function of the log-parameters only
        return neg_log_likelihood_log_scale(log_params, times, T)

    result = minimize(f, start, method="Nelder-Mead",  # Nelder-Mead: needs no gradients
                      options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 4000})  # tight tolerances
    log_hat = result.x  # best log-parameters
    estimate = np.exp(log_hat)  # back to the original scale

    # Standard errors. For large samples the MLE is approximately normal with covariance
    # = inverse of the Hessian of -loglik at the optimum (the "observed information").
    # A sharply curved likelihood means a well-determined parameter and a small SE.
    cov_log = np.linalg.inv(numerical_hessian(f, log_hat))  # covariance of the log-parameters
    se_log = np.sqrt(np.diag(cov_log))  # their standard errors
    se = estimate * se_log  # delta method: d(exp(x))/dx = exp(x), so SE(p) ~ p * SE(log p)

    # 95% confidence interval, built on the log scale (so it never goes negative) and then
    # transformed back: exp(log_hat +/- 1.96 * se_log).
    ci_low = np.exp(log_hat - 1.96 * se_log)
    ci_high = np.exp(log_hat + 1.96 * se_log)
    return estimate, se, ci_low, ci_high, -result.fun, result.success  # -fun = max loglik


# ---------------------------------------------------------------------------
# 5. Residuals (time-rescaling theorem)
# ---------------------------------------------------------------------------
def rescaled_gaps(times, mu, alpha, beta):
    """Gaps between Lambda(t_i), where Lambda(t) = integral_0^t lambda(s) ds.

    Theorem: if the model is correct, Lambda(t_1), Lambda(t_2), ... is a Poisson process with
    rate 1, so these gaps are independent Exp(1). If the model is wrong, they are not.
    """
    Lam = np.array([
        mu * ti + alpha * np.sum(1 - np.exp(-beta * (ti - times[:i])))  # same formula as the
        for i, ti in enumerate(times)                                   # compensator, up to t_i
    ])
    return np.diff(Lam, prepend=0.0)  # Lambda(t_1) - 0, Lambda(t_2) - Lambda(t_1), ...


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    true = {"mu": 0.5, "alpha": 0.6, "beta": 2.0}  # parameters used to simulate
    T = 1000  # length of the time window

    # 1. Simulate
    times = simulate_hawkes(**true, T=T, seed=1)
    n = len(times)
    print(f"Simulated {n} events; expected about mu*T/(1-alpha) = {true['mu'] * T / (1 - true['alpha']):.0f}\n")

    # 2-3. Fit
    est, se, lo, hi, ll_hawkes, ok = fit_hawkes(times, T)
    print(f"{'param':6} {'true':>6} {'estimate':>9} {'std err':>8} {'95% CI':>18}")
    for k, name in enumerate(PARAM_NAMES):
        print(f"{name:6} {true[name]:>6} {est[k]:>9.3f} {se[k]:>8.3f}   [{lo[k]:.3f}, {hi[k]:.3f}]")
    print(f"max log-likelihood = {ll_hawkes:.2f}, optimiser converged: {ok}\n")

    # 4. Likelihood-ratio test against Poisson (the special case alpha = 0).
    # Poisson MLE: mu_hat = n/T, with log-likelihood n*log(n/T) - n.
    ll_poisson = n * np.log(n / T) - n
    lr = 2 * (ll_hawkes - ll_poisson)  # how much better Hawkes fits
    # Hawkes has 2 extra parameters, so compare to chi-squared(2). Approximate here, because
    # alpha = 0 is on the edge of the parameter space, but a huge LR is decisive anyway.
    print(f"LR test vs Poisson: LR = {lr:.1f}, p-value ~ {chi2.sf(lr, df=2):.2g}")
    print(f"AIC: Poisson = {2 * 1 - 2 * ll_poisson:.1f}, Hawkes = {2 * 3 - 2 * ll_hawkes:.1f} (lower is better)\n")

    # 5. Goodness of fit
    gaps = rescaled_gaps(times, *est)
    gaps_poisson = np.diff(times * n / T, prepend=0.0)  # Poisson's Lambda(t) is (n/T)*t
    print(f"KS test of rescaled gaps vs Exp(1): Hawkes p = {kstest(gaps, 'expon').pvalue:.3f}, "
          f"Poisson p = {kstest(gaps_poisson, 'expon').pvalue:.2g}")
    print("(a large p-value means no evidence against the model)")

    # 6. Plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    grid = np.linspace(0, 50, 2000)  # zoom in on the first 50 time units
    window = times[times <= 50]
    ax1.plot(grid, [intensity(g, window[window < g], *est) for g in grid], label="fitted λ(t)")
    ax1.plot(window, np.zeros_like(window), "|", ms=12, c="k", label="events")
    ax1.axhline(est[0], ls="--", c="grey", lw=1, label="fitted μ")
    ax1.set(xlabel="time", ylabel="intensity", title="Fitted intensity (first 50 time units)")
    ax1.legend()
    q = -np.log(1 - (np.arange(1, n + 1) - 0.5) / n)  # Exp(1) theoretical quantiles
    ax2.plot(q, np.sort(gaps), ".", ms=3, label="Hawkes fit")
    ax2.plot(q, np.sort(gaps_poisson), ".", ms=3, label="Poisson fit")
    ax2.plot([0, q.max()], [0, q.max()], c="grey", lw=1)  # perfect-fit diagonal
    ax2.set(xlabel="Exp(1) quantiles", ylabel="rescaled gaps", title="Residual QQ plot")
    ax2.legend()
    plt.tight_layout()
    out = "_output/hawkes_fit.png"
    os.makedirs("_output", exist_ok=True)  # create the output folder if needed
    plt.savefig(out, dpi=130)
    print(f"\nPlot saved to {out}")
