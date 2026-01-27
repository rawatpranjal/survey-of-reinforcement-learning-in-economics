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
| 6 | `ch06_bandits/` | Bandits & Online Learning | TBD | Draft tex; simulation pending |
| 7 | `ch07_applications/` | Real World Applications | -- (survey chapter) | Stub; needs substantial expansion |
| 8 | `ch08_rlhf/` | RLHF & Preference Learning | TBD | Draft tex; simulation pending |
| 9 | `ch09_conclusion/` | Conclusion & Discussion | -- | Stub; needs expansion |

**Deferred:** "Economic Models for RL" chapter -- will decide placement later.

### Per-Chapter Folder Layout

```
chXX_topic/
  tex/          # LaTeX source for this chapter
  papers/       # Key reference PDFs for this chapter
  sims/         # Python scripts, figures (PNG), LaTeX tables (.tex)
```

### Top-Level Layout

```
docs/           # Combined LaTeX (main.tex, refs.bib, figs/, econometrica.bst)
archive/        # Old repo structure preserved (qlearning/, ppo/, slides/, etc.)
ORE_main/       # Sister survey (separate project, read-only reference)
```

### Tasklist

High priority (content gaps):

1. Create a bandits simulation script for Chapter 6. The script should implement at least UCB and Thompson Sampling on a synthetic multi-armed bandit or dynamic pricing problem, compare regret against an epsilon-greedy baseline, and produce publication-quality figures saved to `ch06_bandits/sims/`.

2. Expand Chapter 7 (Applications) from its current 26-line stub to a full survey section. Add structured case studies across domains (platform markets, finance, healthcare, operations) with explicit discussion of state/action/reward formulations used in each deployment.

3. Create an RLHF and DPO simulation script for Chapter 8. The script should demonstrate reward learning from pairwise preferences and policy optimization under the Bradley-Terry model. Output figures and a comparison table saved to `ch08_rlhf/sims/`.

4. Expand Chapter 9 (Conclusion) from its current 6-line stub. Add discussion of open theoretical problems, emerging application areas, methodological limitations of current RL approaches in economics, and directions for future interdisciplinary research.

Medium priority (content quality):

5. Split `se_in_rl_full.tex` in Chapter 6. Extract the bandits and online learning content into a standalone file; archive the remainder or redistribute to other chapters.

6. Consolidate the two tex variants in Chapter 2 (`planning_learning.tex` and `planning_learning_alt.tex`). Determine the canonical version and archive the other.

7. Triage exploratory notebooks in Chapter 4 (`testing0.ipynb`, `testing1.ipynb`, `testing2.ipynb`). Archive or document their purpose relative to the main IRL experiments.

8. Deepen the RLHF chapter tex. Expand the DPO derivation, add a worked Bradley-Terry example, and connect more explicitly to discrete choice econometrics.

Low priority (polish):

9. Populate each chapter's `papers/` directory with key reference PDFs.

10. Standardize existing simulation notebooks in Chapters 2 through 5 to match the simulation standards defined in this file.

11. Decide whether to include an "Economic Models for RL" chapter and determine its placement.

12. Verify end-to-end LaTeX compilation from `docs/`.

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

### Script Structure
The primary simulation artifact is a standalone Python script (`.py`), not a Jupyter notebook. Each script is self-contained and produces all outputs without manual intervention. Existing notebooks in Chapters 2 through 5 remain as-is; all new simulation work uses scripts.

Each script should follow this structure:
1. **Header comment:** title, chapter, one-sentence description of the experiment.
2. **Imports and configuration:** all dependencies, random seeds, and hyperparameters defined at the top.
3. **Environment or model definition:** the economic model, MDP, or game being studied.
4. **Algorithm implementation:** the RL method under evaluation, clearly commented.
5. **Baseline implementation:** the comparison method (DP, analytical solution, naive heuristic).
6. **Execution:** run all methods under identical conditions (same seeds, same environment instances).
7. **Output generation:** produce publication-quality figures (PNG, 300 dpi) and optionally LaTeX table fragments (`.tex`) saved to the chapter's `sims/` directory. All outputs must be directly includable via `\includegraphics` and `\input`.

### Study Design
Each simulation constitutes a computational experiment and should adhere to the following:
- State the hypothesis or research question the simulation addresses.
- Fix random seeds across methods to ensure controlled comparison.
- Run each method across multiple seeds (minimum 10) and report means and standard errors.
- Label all figure axes, include legends, and use consistent color schemes across related figures.
- Where applicable, include a table summarizing key numerical results (mean reward, regret, convergence iteration).

### Python Conventions
- Python 3.10+
- Core: `numpy`, `scipy`, `matplotlib`
- Deep RL (where needed): `torch`
- Set random seeds explicitly: `np.random.seed(42)` at top. Use a loop over seeds for multi-run experiments.
- Use descriptive variable names matching the math: `V` for value function, `Q` for Q-values, `pi` for policy.
- Inline comments for non-obvious steps. No boilerplate docstrings.
- Plots: `matplotlib` with clear labels, titles, legends. Use `fig, ax = plt.subplots()` pattern. Save at 300 dpi via `fig.savefig("filename.png", dpi=300, bbox_inches="tight")`.
- LaTeX table output: write `.tex` files containing `tabular` environments, directly includable with `\input{}`.

## File Tracking

- **This file** (`claude.md`): project context, style guide, master plan. Keep updated.
- **Changelog** (`changelog.md`): reverse-chronological log of structural changes. Update after each significant change.
- **Detailed restructure plan**: `docs/plans/2026-01-27-repo-restructure.md` (step-by-step tasks with commands)

## Key Commands

```bash
# Build LaTeX (from docs/ dir)
cd docs && pdflatex -shell-escape main.tex && bibtex main && pdflatex -shell-escape main.tex && pdflatex -shell-escape main.tex

# Run a simulation script
python chXX_topic/sims/script_name.py
```
