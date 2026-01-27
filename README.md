# Survey of Reinforcement Learning in Economics

**Pranjal Rawat, Georgetown University**

An arXiv survey paper covering reinforcement learning in economics. Each chapter includes a literature review, mathematical treatment of key papers, and working simulations.

## Chapters

| Ch | Folder | Title | Main Simulation |
|----|--------|-------|-----------------|
| 0 | `ch00_introduction/` | Abstract + Introduction | -- |
| 1 | `ch01_history/` | Historical Developments | -- |
| 2 | `ch02_planning_learning/` | Planning and Learning (DP vs RL) | Gridworld, Bus Engine |
| 3 | `ch03_rl_structural_est/` | RL for Structural Estimation | Bus Engine (DP vs DQN) |
| 4 | `ch04_inverse_rl/` | Inverse Reinforcement Learning | IRL LP Gridworld |
| 5 | `ch05_rl_in_games/` | RL in Games (MARL) | Korean Auction, NFSP |
| 6 | `ch06_bandits/` | Bandits & Online Learning | TBD |
| 7 | `ch07_applications/` | Real World Applications | -- |
| 8 | `ch08_rlhf/` | RLHF & Preference Learning | TBD |
| 9 | `ch09_conclusion/` | Conclusion & Discussion | -- |

## Building the Paper

```bash
cd docs
pdflatex -shell-escape main.tex && bibtex main && pdflatex -shell-escape main.tex && pdflatex -shell-escape main.tex
```

## Running Simulations

Each chapter's `sims/` folder contains Jupyter notebooks:

```bash
jupyter notebook ch02_planning_learning/sims/gridworld.ipynb
```

## Sister Survey

See `ORE_main/` for the companion paper: *Structural Econometrics and Inverse Reinforcement Learning* (Rust & Rawat, 2026).
