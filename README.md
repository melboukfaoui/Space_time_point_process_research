# Space-Time Point Process Research

**Neural detection of localized regime shifts in self-exciting space-time point processes, with applications to seismicity and limit order books.**

Mamoune El Boukfaoui · Supervisor: Robert Clements (USF Data Institute)

## Overview

Events such as earthquakes and limit-order-book orders arrive in clustered, self-exciting
cascades, and their dynamics can change abruptly in only part of the space (one fault zone;
one side of the book). We train neural classifiers on simulated space-time Hawkes processes
to detect **whether**, **when** and **where** a regime shift happened, and compare them with
likelihood-ratio, scan-statistic and CUSUM baselines. Real-data applications: Oklahoma
induced seismicity (USGS ComCat) and high-frequency order-book data.

## Repository layout

```
.
├── src/stpp/          Python package: simulators, features, models
├── R/                 R functions
├── notebooks/         Quarto notebooks (.qmd), Python and R together
├── scripts/           One-off and setup scripts
├── tests/             pytest tests
├── data/
│   ├── raw/           Downloaded data (not committed)
│   └── processed/     Derived data (not committed)
├── paper/             Manuscript
├── pyproject.toml     Python dependencies (uv)
├── uv.lock            Locked Python versions (uv)
├── renv.lock          Locked R versions (renv)
└── _quarto.yml        Quarto project settings
```

## Setup

You need [uv](https://docs.astral.sh/uv/), [R](https://cran.r-project.org/) (≥ 4.3) and
[Quarto](https://quarto.org/). On macOS:

```bash
brew install uv quarto   # or skip quarto: `uv sync` installs it as a dev dependency
brew install --cask r
```

### Python (uv)

```bash
uv sync            # creates .venv with all dependencies from uv.lock
uv run pytest      # run tests
uv add <package>   # add a dependency (updates pyproject.toml and uv.lock)
```

### R (renv)

First time only, if `renv.lock` does not exist yet:

```bash
Rscript scripts/setup_r.R
```

Everyone else, after cloning:

```bash
Rscript -e 'renv::restore()'
```

Add a package with `renv::install("pkg")` and then `renv::snapshot()`, and commit `renv.lock`.

### Quarto notebooks

Notebooks are `.qmd` files in `notebooks/`. A notebook with any R chunk is run by knitr, and
its Python chunks run through `reticulate` using `.venv` (set in `.Rprofile`), so variables
pass between languages: `py$x` from R, `r.x` from Python.

```bash
uv sync                                           # Python env must exist first
uv run quarto render notebooks/00-setup-check.qmd     # one notebook
uv run quarto preview notebooks/00-setup-check.qmd    # live preview while editing
```

`notebooks/00-setup-check.qmd` is a quick test that both languages work.

## Hawkes process code

| File | What it is |
|---|---|
| `scripts/hawkes_basic.py` | Temporal Hawkes: thinning simulation, MLE fit, SEs/CIs, Poisson LR test, residuals (every line commented) |
| `scripts/hawkes_spacetime.py` | Space-time Hawkes: the same steps in space and time, plus a space-time close-pairs check |
| `notebooks/03-hawkes-explained.qmd` | Explained version of `hawkes_basic.py`: maths and line-by-line notes |
| `notebooks/04-hawkes-spacetime-explained.qmd` | Explained version of `hawkes_spacetime.py` |
| `notebooks/01-hawkes-simulate-and-fit.qmd` | Space-time Hawkes via Robert's branching simulator (`src/stpp/hawkes.py`) |
| `notebooks/02-hawkes-thinning-and-fit.qmd` | Temporal Hawkes with O(n) likelihood and exact gradient (`src/stpp/hawkes_temporal.py`) |
| `R/simulate_hawkes_clustering.R` | Robert's R branching simulator |

Run a script with `uv run python scripts/hawkes_spacetime.py`; render a page with
`uv run quarto render notebooks/04-hawkes-spacetime-explained.qmd` (output in `_output/`).

## Data

Raw data is not committed. Planned sources:

- **USGS ComCat** earthquake catalogue: `https://earthquake.usgs.gov/fdsnws/event/1/`
- **Binance** public trade and book-depth files: `https://data.binance.vision`
- **LOBSTER** NASDAQ message data (subscription; access to be confirmed)

Download scripts go in `scripts/` and write to `data/raw/`.
