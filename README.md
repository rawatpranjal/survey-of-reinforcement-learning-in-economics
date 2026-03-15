# A Survey of Reinforcement Learning for Economics

**Pranjal Rawat, Georgetown University**

Paper: [arXiv:XXXX.XXXXX](https://arxiv.org/) (forthcoming)

Companion survey: *Structural Econometrics and Inverse Reinforcement Learning* (Rust & Rawat, 2026)

## Overview

This repository contains the LaTeX source and simulation code for a survey covering reinforcement learning methods in economics. Topics include dynamic programming theory, policy gradient methods, optimal control, structural estimation, multi-agent games, bandits, RLHF, and causal inference. Each chapter pairs a literature review and mathematical treatment with reproducible computational experiments.

## Quick Start

```bash
git clone https://github.com/rawatpranjal/survey-of-reinforcement-learning-in-economics.git
cd survey-of-reinforcement-learning-in-economics
pip install -r requirements.txt

# Run a simulation (e.g., bandit fundamentals)
python ch07_bandits/sims/bandit_fundamentals.py

# Build the paper
cd docs && pdflatex -shell-escape main.tex && bibtex main && pdflatex -shell-escape main.tex && pdflatex -shell-escape main.tex
```

## Repository Structure

```
docs/                       # Combined LaTeX document (main.tex, refs.bib)
ch00_introduction/tex/      # Abstract and introduction
ch01_history/tex/            # Historical developments
ch02_rl_algorithms/tex/      # RL algorithms survey
ch03_theory/                 # Planning and learning theory + simulations
ch03a/                       # Illustrated example (gridworld case studies)
ch03a_bm/                    # Brock-Mirman case studies
ch03b_deeprl_practice/       # Empirics of deep RL + simulations
ch04_control_problems/       # Optimal control applications + simulations
ch05_econ_models/            # Structural estimation + simulations
ch06_games/                  # RL in games + simulations
ch07_bandits/                # Bandits and dynamic pricing + simulations
ch08_rlhf/                   # RLHF and preference learning + simulations
ch09_causal/                 # Causal inference and RL + simulations
ch10_conclusion/             # Conclusion
sims/                        # Shared utilities (plot_style.py)
```

## Simulations

Each chapter's `sims/` directory contains standalone Python scripts that produce publication-quality figures (PNG) and LaTeX tables (`.tex`). All scripts are self-contained and reproduce the results reported in the paper.

### Planning and Learning Theory (`ch03_theory/sims/`)

| Script | Description |
|--------|-------------|
| `brock_mirman_newton.py` | Brock-Mirman optimal growth: VI vs PI vs LP dual |
| `lqc_fvi_fqi.py` | Linear-quadratic control: Fitted Value Iteration vs Fitted Q-Iteration vs DQN |
| `trust_region_lqc.py` | Trust region visualization for TRPO/PPO in LQC monetary policy |
| `lqr_convergence.py` | LQR convergence rate comparison: VI, PI, and Q-learning |
| `theory_validation.py` | Empirical validation of Newton framework predictions |
| `gridworld_study.py` | Gridworld algorithm comparison (30 Monte Carlo replications) |
| `gridworld_algorithms.py` | Comprehensive gridworld algorithm implementations |
| `ssp_gridworld_20x20.py` | Stochastic shortest path on 20x20 gridworld |
| `bairds_counterexample.py` | Baird's counterexample: divergence and three fixes |
| `deadly_triad_geometry.py` | Deadly triad geometry: orthogonal vs oblique projection |
| `qlearning_geometry.py` | Q-learning on Bertsekas V,TV geometry diagram |

### Empirics of Deep RL (`ch03b_deeprl_practice/sims/`)

| Script | Description |
|--------|-------------|
| `bellman_vs_return.py` | Bellman error vs Monte Carlo return comparison |
| `brock_mirman_dqn.py` | Brock-Mirman solved with DQN |
| `brock_mirman_bellman.py` | Brock-Mirman Bellman error analysis |
| `overestimation_bias.py` | Overestimation bias in Q-learning via Jensen's inequality |

### Illustrated Example: Gridworld (`ch03a/sims/`)

| Script | Description |
|--------|-------------|
| `gridworld_illustrated.py` | 9 algorithms on a 5x5 gridworld (2 planning + 7 learning) |
| `vi/case_study.py` | Value Iteration case study |
| `pi/case_study.py` | Policy Iteration case study |
| `ql/case_study.py` | Q-Learning case study |
| `sarsa/case_study.py` | SARSA (GLIE) case study |
| `ql_trace/case_study.py` | Q(lambda) case study |
| `dqn/case_study.py` | DQN case study |
| `reinforce/case_study.py` | REINFORCE case study |
| `npg/case_study.py` | Natural Policy Gradient case study |
| `ppo/case_study.py` | PPO case study |

### Illustrated Example: Brock-Mirman (`ch03a_bm/sims/`)

| Script | Description |
|--------|-------------|
| `bm_illustrated.py` | 8 algorithms on the stochastic growth model |
| `bm_fvi_fqi.py` | FVI vs FQI with oracle and NLLS basis functions |

### RL for Optimal Control (`ch04_control_problems/sims/`)

| Script | Description |
|--------|-------------|
| `econ_benchmark.py` | Shared abstractions for economic benchmark environments |
| `benchmark_bus_engine.py` | Bus engine replacement benchmark: scaling analysis |

### Structural Estimation (`ch05_econ_models/sims/`)

| Script | Description |
|--------|-------------|
| `bus_engine_dp_vs_dqn.py` | Bus engine: DP vs DQN comparison |
| `estimation_flowcharts.py` | NFXP vs RL-based structural estimation flowcharts |

### RL in Games (`ch06_games/sims/`)

| Script | Description |
|--------|-------------|
| `kuhn_poker_equilibrium.py` | Kuhn poker equilibrium computation |
| `durable_goods_monopoly.py` | 2-period durable goods monopoly (Coase conjecture) |
| `coase_stress_tests.py` | Coase conjecture stress tests: 4 validation tests |
| `cournot_bertrand_marl.py` | Cournot and Bertrand duopoly: multi-agent Q-learning |

### Bandits and Dynamic Pricing (`ch07_bandits/sims/`)

| Script | Description |
|--------|-------------|
| `bandit_fundamentals.py` | Bandit algorithms: UCB, Thompson sampling, epsilon-greedy |
| `auction_reserve_price.py` | Auction reserve price optimization |
| `strategic_pricing.py` | Strategic pricing with reference effects |
| `structural_pricing_misra.py` | Structural dynamic pricing (Misra et al.) |
| `structural_pricing_misra_diagnostic.py` | Diagnostic analysis for structural pricing |
| `knowledge_ladder.py` | Knowledge gradient and information ladder |
| `regret_rates.py` | Regret rate comparison across bandit algorithms |
| `uninformative_price.py` | Why the optimal price reveals nothing about demand |

### RLHF and Preference Learning (`ch08_rlhf/sims/`)

| Script | Description |
|--------|-------------|
| `job_search_rlhf.py` | RLHF preference-based utility recovery in job search |
| `job_search_dpo.py` | DPO in job search model |
| `job_search_preference_learning.py` | Combined RLHF and DPO preference learning |
| `preference_learning.py` | Bradley-Terry preference learning simulation |
| `nfxp_vs_rlhf.py` | NFXP vs RLHF preference recovery comparison |
| `gridworld_rlhf.py` | RLHF on gridworld: preference-based policy learning |
| `gridworld_rlhf_extended.py` | Extended gridworld RLHF with neural net reward model |
| `dpo_diagnosis.py` | DPO failure mode diagnosis |
| `rlhf_dpo_pipeline.py` | RLHF vs DPO pipeline comparison diagram |

### Causal Inference and RL (`ch09_causal/sims/`)

| Script | Description |
|--------|-------------|
| `confounded_ope.py` | Off-policy evaluation under confounding in retail pricing |
| `identification_dags.py` | Identification strategy DAGs |

## Shared Utilities

`sims/plot_style.py` provides a consistent matplotlib style across all figures. Import and call `apply_style()` at the top of each simulation script.

## Dependencies

See `requirements.txt`. Core dependencies: NumPy, SciPy, Matplotlib, tqdm. Deep RL simulations additionally require PyTorch. Install all with:

```bash
pip install -r requirements.txt
```

## Citation

```bibtex
@article{rawat2026rl,
  title={A Survey of Reinforcement Learning for Economics},
  author={Rawat, Pranjal},
  year={2026},
  note={Georgetown University}
}
```

## License

MIT
