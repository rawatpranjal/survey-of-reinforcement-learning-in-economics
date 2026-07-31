# Reproducing the arXiv v6 code

This repository contains the manuscript source, simulation programs, generated figures and
tables, and captured stdout for arXiv version 6 of *A Survey of Reinforcement Learning For
Economics*. The manuscript files are unchanged from the source submitted to arXiv.

## Verify the files

From the repository root, run:

```bash
python3 scripts/check_public_code.py
```

This command checks that every program listed by the main runner exists, every figure and table
shipped with the arXiv source is present, and the corresponding source file is included. It also
checks the dependency files and README links.

## Main environment

Python 3.10 or later is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

List the simulations:

```bash
python scripts/run_all_sims.py --list
```

Run one chapter or one experiment:

```bash
python scripts/run_all_sims.py --chapter ch07
python scripts/run_all_sims.py --script knowledge_ladder
```

The full suite is compute intensive. Each chapter's `sims/` directory contains its programs,
supporting modules, publication figures, LaTeX table fragments, and `_stdout.txt` records. Cached
intermediate results are excluded because they can be regenerated from the programs.

## Code by paper chapter

| Paper section | Code | Runner filter |
|---|---|---|
| Reinforcement Learning Algorithms | [`ch02_rl_algorithms/sims/`](ch02_rl_algorithms/sims/) | `--chapter ch02` |
| Theory of Reinforcement Learning | [`ch03_theory/sims/`](ch03_theory/sims/) | `--chapter ch03_theory` |
| Empirics of Deep RL | [`ch03b_deeprl_practice/sims/`](ch03b_deeprl_practice/sims/) | `--chapter ch03b` |
| Optimal Control | [`ch04_control_problems/sims/`](ch04_control_problems/sims/) | `--chapter ch04` |
| Structural Estimation | [`ch05_econ_models/sims/`](ch05_econ_models/sims/) | `--chapter ch05` |
| Macroeconomic Models | [`ch06_macro/sims/`](ch06_macro/sims/) | `--chapter ch06_macro` |
| Games | [`ch06_games/sims/`](ch06_games/sims/) | `--chapter ch06` |
| Bandits and Dynamic Pricing | [`ch07_bandits/sims/`](ch07_bandits/sims/) | `--chapter ch07` |
| Offline Reinforcement Learning | [`ch08_offline_rl/sims/`](ch08_offline_rl/sims/) | `--chapter ch08_offline` |
| RLHF and AI Alignment | [`ch09_rlhf/sims/`](ch09_rlhf/sims/) | `--chapter ch09` |
| Causal Inference for RL | [`ch10_causal/sims/`](ch10_causal/sims/) | `--chapter ch10` |
| OPE and Dynamic Treatment Effects | [`ch10b_rl_for_ci/sims/`](ch10b_rl_for_ci/sims/) | `--chapter ch10b` |
| Causal Bandits | [`ch10c_adaptive_experiments/sims/`](ch10c_adaptive_experiments/sims/) | `--chapter ch10c` |
| Risk, Robustness, and Constraints | [`ch11_dist_robust_constrained/sims/`](ch11_dist_robust_constrained/sims/) | `--chapter ch11` |
| World Models | [`ch12_world_models/sims/`](ch12_world_models/sims/) | `--chapter ch12` |
| Field Deployments | [`ch13_field_deployments/sims/`](ch13_field_deployments/sims/) | Separate environment below |
| Mathematical Preliminaries | [`appA_preliminaries/sims/`](appA_preliminaries/sims/) | `--chapter appA` |

Shared plotting and cache utilities are in [`sims/`](sims/). The complete list for the main
environment is in [`scripts/run_all_sims.py`](scripts/run_all_sims.py).

## Chapter 13 environment

The field-deployments experiment has its own environment because SCOPE-RL and d3rlpy constrain
Gym, NumPy, and PyTorch versions.

```bash
cd ch13_field_deployments/sims
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python field_ope_reliability.py > field_ope_reliability_stdout.txt 2>&1
pytest -q
```

`field_ope_reliability.py` runs the experiment. `promo_env.py` defines the environment,
`pipeline.py` implements training and OPE, and `ope_diagnostics.py` contains the reliability
checks. The editable production-loop diagram is `horizon_pipeline.tex`.

## Paper build

The manuscript build and arXiv archive procedure are documented in
[`docs/runbooks/arxiv-package.md`](docs/runbooks/arxiv-package.md). Generated results are kept
next to their programs so readers can compare source, captured output, figures, and tables.
