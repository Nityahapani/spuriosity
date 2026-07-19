# spuriosity worked examples

Three notebooks, each showing a full workflow: generate a DGP with
`spuriosity` → fit one or more reference methods → compare recovered
estimates against the known ground truth → takeaway.

All three are self-contained and runnable directly in Google Colab (open
via GitHub's "Open in Colab" badge, or upload manually) — the first code
cell in each installs `spuriosity` from this repository with the right
extras via `%pip install`. Each has been executed end-to-end against a
genuinely blank Python environment (no local project state) to confirm
the install-and-run path works exactly as written.

## Notebooks

- **[`01_did_selection_bias.ipynb`](01_did_selection_bias.ipynb)** — Did
  my difference-in-differences estimator recover the true treatment
  effect under attrition bias? Uses `SelectionBias` and `did_fit`.
- **[`02_causal_forest_vs_ols.ipynb`](02_causal_forest_vs_ols.ipynb)** —
  Does a causal forest beat a naive linear-interaction model on a
  confounded, genuinely nonlinear heterogeneous treatment effect? Uses
  `Confounder`, `add_hte`, and `causal_forest_fit`.
- **[`03_weak_instruments.ipynb`](03_weak_instruments.ipynb)** — Does my
  IV strategy recover the true coefficient, and what happens when the
  instrument is weak? Uses `Endogeneity`, `iv2sls_fit`, and
  `first_stage_f_stat`.

## A note on constructing DiD scenarios

Notebook 1 documents a real gotcha worth knowing generally: if you want a
DiD-style scenario where treatment *group membership* is fixed from the
start but the *effect* only begins at some later period, use
`add_treatment(start_period=0, ...)` for group membership and a separate
`add_structural_break(kind="coefficient_shift", ...)` for when the effect
kicks in. Using `add_treatment`'s own `start_period` to encode "when the
effect begins" makes the treatment column collinear with `did_fit`'s
`post` indicator, since both would only vary together after that period.
