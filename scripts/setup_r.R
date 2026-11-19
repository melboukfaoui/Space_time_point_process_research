# One-time R setup: creates the renv environment and records it in renv.lock.
# Run from the repo root:  Rscript scripts/setup_r.R
# Afterwards, collaborators only need:  Rscript -e 'renv::restore()'

if (!requireNamespace("renv", quietly = TRUE)) {
  install.packages("renv", repos = "https://cloud.r-project.org")
}

renv::init(bare = TRUE, restart = FALSE)
renv::install(c(
  "reticulate", # run Python chunks from R / Quarto
  "knitr",
  "rmarkdown",
  "ggplot2",
  "data.table"
))
renv::snapshot(type = "all", prompt = FALSE)
