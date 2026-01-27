# Survey of Reinforcement Learning in Economics

## What This Is

An arXiv survey paper covering reinforcement learning in economics -- broadly. Each chapter covers a major topic area with a literature review, mathematical treatment of key papers, and one main working simulation that showcases RL's power relative to other methods.

This is a PhD thesis component (Pranjal Rawat, Georgetown University, advisor: John Rust).

## Sister Survey (ORE_main)

**Location:** `/Users/pranjal/Code/rl/ORE_main`
**Title:** "Structural Econometrics and Inverse Reinforcement Learning: Inferring preferences and beliefs from human behavior" (Rust & Rawat, Jan 2026)
**Scope:** Narrow -- DDC estimation + IRL only (Introduction, MDP Background, DDC Estimation, IRL, Conclusion)

**Our relationship to ORE_main:**
- We go *broader* -- RL in economics writ large (bandits, MARL, RLHF, applications, etc.)
- We avoid deep overlap with ORE_main's DDC/IRL focus. Reference it, don't replicate.
- Our tone is lighter. We take more space. We write more, add worked examples and sims.
- We can learn from ORE_main's presentation style (clean LaTeX, natbib, figure conventions).

## Master Plan

### Chapter Structure

| Ch | Folder | Title | Main Simulation | Status |
|----|--------|-------|-----------------|--------|
| 0 | `ch00_introduction/` | Abstract + Introduction | -- | Has draft tex |
| 1 | `ch01_history/` | Historical Developments | -- | Has draft tex |
| 2 | `ch02_planning_learning/` | Planning and Learning (DP vs RL) | Gridworld, Bus Engine | Has draft tex + sims |
| 3 | `ch03_rl_structural_est/` | RL for Structural Estimation | Bus Engine (DP vs DQN) | Has draft tex + sim |
| 4 | `ch04_inverse_rl/` | Inverse Reinforcement Learning | IRL LP Gridworld | Has draft tex + sims |
| 5 | `ch05_rl_in_games/` | RL in Games (MARL) | Korean Auction, NFSP | Has draft tex + sims |
| 6 | `ch06_bandits/` | Bandits & Online Learning | TBD | Needs split from `7_se_in_rl.tex` |
| 7 | `ch07_applications/` | Real World Applications | -- (survey chapter) | Has draft tex |
| 8 | `ch08_rlhf/` | RLHF & Preference Learning | TBD | Has draft tex |
| 9 | `ch09_conclusion/` | Conclusion & Discussion | -- | Has draft tex |

**Deferred:** "Economic Models for RL" chapter -- will decide placement later.

### Per-Chapter Folder Layout

```
chXX_topic/
  tex/          # LaTeX source for this chapter
  papers/       # Key reference PDFs for this chapter
  sims/         # Jupyter notebooks, Python scripts, figures
```

### Top-Level Layout

```
docs/           # Combined LaTeX (main.tex, refs.bib, figs/, econometrica.bst)
archive/        # Old repo structure preserved (qlearning/, ppo/, slides/, etc.)
ORE_main/       # Sister survey (separate project, read-only reference)
```

### Open Items

- [ ] Split `7_se_in_rl.tex` -- bandits content to `ch06_bandits/`, remainder archived
- [ ] Create bandits simulation notebook
- [ ] Create RLHF/DPO demonstration notebook
- [ ] Populate `papers/` dirs with key references per chapter
- [ ] Standardize notebook style across all sims
- [ ] Decide "Economic Models for RL" chapter placement

## Writing Style

### LaTeX Conventions
- **Document class:** `article`, 11pt, a4paper
- **Citation style:** `natbib` with `plainnat` or `econometrica` bst. Use `\citet{}` for textual, `\citep{}` for parenthetical.
- **Math:** `amsmath`, `amsthm`, `amssymb`. Use `\mathbb{}` for sets, `\mathcal{}` for calligraphic, `\boldsymbol{}` for vector notation.
- **Theorems:** numbered as Theorem, Lemma, Definition, Assumption, Corollary. Use `\newtheorem`.
- **Code listings:** `minted` package with light gray background (`bg` color), `breaklines=true`.
- **Figures:** `graphicx` + `subfig`. Store in `docs/figs/` or chapter `sims/` dir. Reference with relative paths.
- **Highlighting/TODOs:** `todonotes` package for draft notes. Remove before submission.

### Prose Style
- Academic but accessible. Target audience: economists curious about RL, and RL researchers curious about economics.
- Lighter than ORE_main. More expository. Explain intuition before formalism.
- Each chapter: motivate the topic, review key papers broadly, drill into the math of 2-3 foundational papers, then present the simulation.
- Use concrete examples early. Don't front-load notation.
- Avoid jargon from one field without explaining it to the other. Define terms from both CS and economics when first used.
- Keep paragraphs short. Use subsections liberally.

### Notation Conventions
- States: $s \in \mathcal{S}$, actions: $a \in \mathcal{A}$
- Policy: $\pi(a|s)$, value function: $V^\pi(s)$, action-value: $Q^\pi(s,a)$
- Reward: $r(s,a)$ or $R_t$, discount factor: $\gamma \in [0,1)$
- Transition: $P(s'|s,a)$ or $T(s'|s,a)$
- Utility: $u(\cdot)$, payoff: $\pi_i$ (context-dependent, distinguish from policy)
- Expectation: $\mathbb{E}[\cdot]$, probability: $\mathbb{P}(\cdot)$
- Indicator: $\mathbbm{1}\{\cdot\}$
- Follow ORE_main notation where topics overlap (MDP setup, IRL). Check `ORE_main/NOTATION_REVIEW.md`.

## Simulation Standards

### Notebook Structure
Each main simulation notebook should follow:
1. **Title + description** (markdown cell): what this demonstrates, which chapter it belongs to
2. **Setup** (code cell): imports, seeds, parameters
3. **Environment/Model definition** (code cells): the economic model or environment
4. **Algorithm implementation** (code cells): the RL method, clearly commented
5. **Baseline comparison** (code cells): DP, analytical solution, or naive method
6. **Results + visualization** (code cells): plots comparing methods, convergence, performance
7. **Discussion** (markdown cell): what we learned, limitations

### Python Conventions
- Python 3.10+
- Core: `numpy`, `scipy`, `matplotlib`
- Deep RL (where needed): `torch`
- Set random seeds explicitly: `np.random.seed(42)` at top
- Use descriptive variable names matching the math: `V` for value function, `Q` for Q-values, `pi` for policy
- Inline comments for non-obvious steps. No boilerplate docstrings.
- Plots: `matplotlib` with clear labels, titles, legends. Use `fig, ax = plt.subplots()` pattern.
- Save key figures as PNG for inclusion in LaTeX.

## File Tracking

- **This file** (`claude.md`): project context, style guide, master plan. Keep updated.
- **Changelog** (`changelog.md`): reverse-chronological log of structural changes. Update after each significant change.
- **Detailed restructure plan**: `docs/plans/2026-01-27-repo-restructure.md` (step-by-step tasks with commands)

## Key Commands

```bash
# Build LaTeX (from docs/ dir)
cd docs && pdflatex -shell-escape main.tex && bibtex main && pdflatex -shell-escape main.tex && pdflatex -shell-escape main.tex

# Run a simulation notebook
jupyter notebook chXX_topic/sims/notebook.ipynb
```
