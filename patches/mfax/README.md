# MFAX local patches (commit 9acc1eb)

These patches are applied to a clean clone of
`https://github.com/CWibault/mfax.git` at commit `9acc1eb` to produce the
`mfax_lq_grid_results.json` artifact in `ch06_macro/sims/`. They are not
upstreamed: they are local conveniences for running the public scripts under
Python 3.11 and parsing per-iteration return values out of stdout.

## What changed

`py311_compat.patch` covers two unrelated, mechanical edits across seven files:

1. **Python 3.11 dataclass `default_factory` compatibility (5 files).** Five
   environment param classes declare mutable `jax.Array` defaults at class
   scope:
   - `mfax/envs/base/toy/beach_bar_1d.py`
   - `mfax/envs/base/toy/linear_quadratic.py`
   - `mfax/envs/pushforward/macro/endogenous.py`
   - `mfax/envs/sample/macro/endogenous.py`

   Python 3.11 dataclasses reject the bare `jax.Array = jnp.array([...])`
   form. The patch wraps each in `field(default_factory=lambda: jnp.array(...))`,
   which is the standard fix and preserves the values byte-for-byte. The
   `linear_quadratic.py` env (the one used in this chapter) is in the touched
   set; the others are touched only to keep an unrelated import path importable.

2. **Per-iteration `Return` print (3 files).** The HSM training scripts
   - `mfax/algos/hsm/algos/spg.py`
   - `mfax/algos/hsm/algos/rspg.py`
   - `mfax/algos/rl/algos/rippo.py`

   already log `Iteration`, `Train Time`, and `Exploitability` on each eval
   step. The patch adds the existing `mean_policy_return` field to the same
   `jax.debug.print` line as a `Return:` token. This is a pure logging
   addition and matches the regex in `mfax_lq_run_grid.py` so the wrapper
   can capture the per-iteration return without changing any training math.

The full set is 39 added lines and 20 removed across seven files; see the
patch for the verbatim diff.

## Algorithm identity

None of these edits touches the SPG, RSPG, or RIPPO update rules, the
environment dynamics, or the exploitability evaluator. They only affect
default construction of param dataclasses and stdout formatting. The
`linear_quadratic` reward coefficients (`c_action=0.5`, `q=0.1`,
`kappa=0.5`, `c_term=1.0`, `sigma=1.0`, `rho=0.5`) and the action / state
shapes are untouched.

## How to reproduce

```bash
RL_REPO="$(pwd)"
git clone https://github.com/CWibault/mfax.git /tmp/mfax
git -C /tmp/mfax checkout 9acc1eb
git -C /tmp/mfax apply "$RL_REPO/patches/mfax/py311_compat.patch"

# Build a Python 3.11 venv for MFAX (see the upstream README for JAX
# version requirements), then point the runner at it:
MFAX_ROOT=/tmp/mfax MFAX_PYTHON=/tmp/mfax-venv/bin/python \
  python3 ch06_macro/sims/mfax_lq_run_grid.py
```

The resulting JSON should reproduce `mfax_lq_grid_results.json` up to
floating-point rounding under JAX's nondeterministic kernels (we run with
fixed seeds 0..9; differences across machines come from JAX backend
selection, not the patch).
