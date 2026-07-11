# Contributing to spuriosity

Thanks for your interest in contributing. This project is in early alpha —
architecture and API surface are still settling, so it's a good time to shape
things, but expect churn.

## Development setup

```bash
git clone https://github.com/Nityahapani/spuriosity.git
cd spuriosity
pip install -e ".[dev,viz,sklearn]"
```

## Running tests

```bash
pytest
```

## Commit conventions

This repo uses [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` — new feature
- `fix:` — bug fix
- `docs:` — documentation only
- `chore:` — tooling, config, non-code changes
- `test:` — adding or fixing tests
- `refactor:` — code change that isn't a fix or a feature

Keep commits atomic — one logical change per commit. Prefer several small,
reviewable commits over one large one.

## Security stance: `pandas.eval` usage

`spuriosity` uses `pandas.eval`-style expression evaluation in two places:

1. **Outcome/HTE formulas** that use the callable path (not the `patsy` string
   path) may pass through evaluation of user-provided expressions.
2. **Selection bias rules** (`add_selection_bias(rule=...)`) accept arbitrary
   boolean expressions evaluated via `pandas.eval`.

This is a deliberate design choice: arbitrary boolean/algebraic expressions are
the only practical way to give users full expressive power without maintaining
a bespoke mini-DSL. However, `pandas.eval` (and Python `eval` generally) can
execute arbitrary code if not constrained.

**Mitigations in place:**

- All evaluation calls use `pandas.eval(expr, local_dict=..., global_dict=...)`
  with explicit, minimal namespaces — never the caller's ambient
  globals/locals, and never Python's builtin `eval`/`exec` directly.
- Only the variables defined on the `PanelGenerator` (i.e., columns that will
  exist in the generated DataFrame) are exposed in `local_dict`. No access to
  the filesystem, network, or arbitrary Python objects is provided.
- `spuriosity` is designed for use with **trusted, user-authored expressions**
  (i.e., the researcher writing their own DGP spec) — it is *not* designed to
  safely evaluate untrusted third-party input (e.g., expressions submitted by
  an anonymous user of a hosted service). If you're embedding `spuriosity` in
  a context where formula/rule strings come from an untrusted source, treat
  that as your application's responsibility to sandbox further, not something
  `spuriosity` guarantees on your behalf.

If you find a way to escape the `local_dict`/`global_dict` restriction, or a
case where an expression can reach outside the intended namespace, please open
an issue — this is treated as a real bug, not a theoretical concern.

## Architecture notes

Before contributing a new pathology or major feature, please read
[`docs/design_spec.md`](docs/design_spec.md) — it documents the reasoning
behind the current API shape (why panel-only, why patsy, why function-based
`StressTest`, the RNG sub-stream architecture, etc.) so we don't relitigate
settled tradeoffs in every PR.

## Adding a new pathology

Pathologies subclass the base `Pathology` class in `pathologies.py` and must:

1. Implement the DGP modification logic.
2. Report what ground truth fields it affects/populates.
3. Participate in `validate_combo()` — declare any known conflicts with other
   pathology types (as warnings, not hard errors — v1 policy is permissive).

## Questions

Open an issue on GitHub for design discussions before submitting large PRs —
saves rework on both sides.
