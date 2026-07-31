# A Survey of Reinforcement Learning For Economics

Pranjal Rawat, Georgetown University

Paper: [arXiv:2603.08956v6](https://arxiv.org/abs/2603.08956v6)

This repository contains the source and computational materials for the current arXiv paper.
The manuscript studies reinforcement learning as a sample-based extension of dynamic
programming, with applications to control, structural estimation, games, pricing, offline
policy evaluation, causal inference, robust control, world models, and field deployments.

The manuscript source and generated paper artifacts on this branch correspond to the source
snapshot used for arXiv version 6, posted on July 27, 2026. The repository also contains the
simulation scripts used to generate the paper's figures and tables.

## Build the paper

```bash
git clone https://github.com/rawatpranjal/survey-of-reinforcement-learning-in-economics.git
cd survey-of-reinforcement-learning-in-economics/docs
pdflatex -shell-escape main.tex
bibtex main
pdflatex -shell-escape main.tex
pdflatex -shell-escape main.tex
pdflatex -shell-escape main.tex
```

The compiled manuscript is `docs/main.pdf`.

## Repository structure

```text
docs/                         Main LaTeX document and bibliography
ch00_introduction/            Introduction
ch01_history/                 History of reinforcement learning
ch02_rl_algorithms/           Reinforcement learning algorithms
ch03_theory/                  Theory of reinforcement learning
ch03b_deeprl_practice/        Empirics of deep reinforcement learning
ch04_control_problems/        Optimal control
ch05_econ_models/             Structural estimation
ch06_macro/                   Macroeconomic models
ch06_games/                   Games
ch07_bandits/                 Bandits and dynamic pricing
ch08_offline_rl/              Offline reinforcement learning
ch09_rlhf/                    Preference learning and RLHF
ch10_causal/                  Causal inference for reinforcement learning
ch10b_rl_for_ci/              Off-policy evaluation and dynamic treatment effects
ch10c_adaptive_experiments/   Causal bandits and adaptive experimentation
ch11_dist_robust_constrained/ Risk-sensitive, robust, and constrained RL
ch12_world_models/            World models and model-based RL
ch13_field_deployments/       Field deployments
appA_preliminaries/           Mathematical preliminaries
sims/                         Shared simulation utilities
scripts/                      Build and verification tools
```

## Code and reproduction

The repository includes the simulation programs, generated figures and tables, and captured
stdout used by arXiv v6. The main runner lists 74 simulations and figure generators. Chapter 13
contains one additional experiment with its own pinned dependencies.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/check_public_code.py
python scripts/run_all_sims.py --list
```

Run one chapter or one experiment without running the full suite.

```bash
python scripts/run_all_sims.py --chapter ch10b
python scripts/run_all_sims.py --script ope_estimators
```

[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) links each paper chapter to its code and explains the
Chapter 13 environment. The arXiv packaging procedure is documented in
[`docs/runbooks/arxiv-package.md`](docs/runbooks/arxiv-package.md).

## Citation

```bibtex
@article{rawat2026rl,
  title   = {A Survey of Reinforcement Learning For Economics},
  author  = {Rawat, Pranjal},
  year    = {2026},
  journal = {arXiv preprint arXiv:2603.08956}
}
```

## License

MIT
