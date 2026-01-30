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
| 2 | `ch02_planning_learning/` | Unified Theory of DP and RL | -- | Has draft tex (planning_learning_theory.tex) |
| 3 | `ch03_rl_control_problems/` | RL for Control Problems | Gridworld, Bus Engine, Discount Targeting, Dispatch, Hotel RM, RTB Bidding, Execution, Datacenter, Inventory | Has draft tex + 9 sims |
| 4 | `ch04_rl_econ_models/` | Solving Economic Models with RL | Bus Engine (DP vs DQN) | Has draft tex + sim |
| 5 | `ch05_rl_in_games/` | RL in Games: Imperfect Information | Durable Goods Monopoly (Coase Conjecture), Kuhn Poker (CFR/FP equilibrium) | Has tex + 2 sims |
| 6 | `ch06_bandits/` | Economic Bandits | Fundamentals, Dynamic Pricing, Auction Reserve, BwK | Has restructured tex + 4 sims |
| 7 | `ch07_rlhf/` | RLHF & Preference Learning | TBD | Draft tex; simulation pending |
| 8 | `ch08_conclusion/` | Conclusion | -- | Stub; needs expansion |

**Removed:** `ch05_rl_as_behaviour/` (IRL content belongs in ORE_main sister survey). Archived to `archive/ch05_rl_as_behaviour/`. Old `ch03_benchmarks/` and `ch08_applications/` also archived after merge into `ch03_rl_control_problems/`.

**Note:** Old ch02 tex files (planning_learning.tex, planning_learning_alt.tex) archived to `ch02_planning_learning/tex/backups/`. Old ch02 sims (gridworld.py, bus_engine_intro.py) remain in place as reference; canonical benchmark versions are in `ch03_rl_control_problems/sims/`.

**Deferred:** "Economic Models for RL" chapter -- will decide placement later.

### Per-Chapter Folder Layout

```
chXX_topic/
  tex/          # LaTeX source for this chapter
  tex/backups/  # Timestamped backups of tex files (YYYY-MM-DD-HHMMSS_filename)
  papers/       # Key reference PDFs for this chapter
  sims/         # Python scripts, figures (PNG), LaTeX tables (.tex)
```

**Papers Directory:** Each chapter's `papers/` folder should contain reference PDFs for validating simulation implementations against the literature. Before implementing any algorithm or running benchmarks, verify formulations against reference papers in the chapter's `papers/` directory. Check that MDP definitions (state, action, transition, reward), update rules, and convergence conditions match the source. Document any intentional simplifications.

### Top-Level Layout

```
docs/           # Combined LaTeX (main.tex, refs.bib, figs/, econometrica.bst)
archive/        # Old repo structure preserved (qlearning/, ppo/, slides/, etc.)
ORE_main/       # Sister survey (separate project, read-only reference)
```

### Tasklist

High priority (content gaps):

1. ~~Create a bandits simulation script for Chapter 6.~~ DONE. Created 4 simulation scripts: `bandit_fundamentals.py`, `dynamic_pricing_bandit.py`, `auction_reserve_price.py`, `bandits_with_knapsacks.py`. Each produces PNG figures and LaTeX tables in `ch06_bandits/sims/`.

2. Create an RLHF and DPO simulation script for Chapter 7. The script should demonstrate reward learning from pairwise preferences and policy optimization under the Bradley-Terry model. Output figures and a comparison table saved to `ch07_rlhf/sims/`.

3. Expand Chapter 8 (Conclusion) from its current 6-line stub. Add discussion of open theoretical problems, emerging application areas, methodological limitations of current RL approaches in economics, and directions for future interdisciplinary research.

Medium priority (content quality):

4. ~~Split `se_in_rl_full.tex` in Chapter 6.~~ DONE. Replaced with `economic_bandits.tex` (8 subsections). Old file archived to `ch06_bandits/tex/backups/2026-01-28_se_in_rl_full.tex`.

5. Consolidate the two tex variants in Chapter 2 (`planning_learning.tex` and `planning_learning_alt.tex`). Determine the canonical version and archive the other.

6. Deepen the RLHF chapter tex (Chapter 7). Expand the DPO derivation, add a worked Bradley-Terry example, and connect more explicitly to discrete choice econometrics.

Low priority (polish):

7. Populate each chapter's `papers/` directory with key reference PDFs.

8. Standardize existing simulation notebooks in Chapters 2 through 5 to match the simulation standards defined in this file.

9. Decide whether to include an "Economic Models for RL" chapter and determine its placement.

10. Verify end-to-end LaTeX compilation from `docs/`.

## Writing Style

### LaTeX Conventions
- **Document class:** `article`, 11pt, a4paper
- **Citation style:** `natbib` with `plainnat` or `econometrica` bst. Use `\citet{}` for textual, `\citep{}` for parenthetical.
- **Math:** `amsmath`, `amsthm`, `amssymb`. Use `\mathbb{}` for sets, `\mathcal{}` for calligraphic, `\boldsymbol{}` for vector notation.
- **Theorems:** numbered as Theorem, Lemma, Definition, Assumption, Corollary. Use `\newtheorem`.
- **Code listings:** `minted` package with light gray background (`bg` color), `breaklines=true`.
- **Figures and tables:** `graphicx` + `subfig`. All figures and tables must be contained in the chapter they belong to (store in the chapter's `sims/` directory). Reference with relative paths from `docs/` (e.g., `../ch07_rlhf/sims/figure.png`).
- **Block quotes:** Use `\begin{quote}\itshape ... \end{quote}` so quoted passages render in italics.
- **Highlighting/TODOs:** `todonotes` package for draft notes. Remove before submission.

### Prose Style
- Academic formal tone. Target audience: economists curious about RL, and RL researchers curious about economics.
- Lighter than ORE_main. More expository. Explain intuition before formalism.
- Each chapter: motivate the topic, review key papers broadly, drill into the math of 2-3 foundational papers, then present the simulation.
- Do not re-define concepts already established in earlier chapters. If Chapter 2 defines the Bellman equation, later chapters should reference it (e.g., "recall the Bellman optimality equation from Section 2.3") rather than re-derive it. Use consistent notation across all chapters per the Notation Conventions below.
- Use concrete examples early. Don't front-load notation.
- Avoid jargon from one field without explaining it to the other. Define terms from both CS and economics when first used.
- Keep paragraphs short. Use subsections liberally. However, avoid leaving stub paragraphs of 1-2 lines; merge these into adjacent paragraphs so the text flows naturally. Aim for 3-6 sentences per paragraph.
- No em dashes. Use commas, semicolons, colons, or separate sentences.
- No bullet points in LaTeX prose. Use full paragraphs or numbered lists only where structurally necessary.
- No `\paragraph{Computational Experiment.}` or similar generic paragraph headers for simulation results. Use `\subsection{Simulation Study: <descriptive title>}` with a label.
- Simulation writeups must be concise: two paragraphs maximum (setup, then results), one consolidated table of all results, one figure (convergence or learning curve style, not bar plots).
- No \textbf{} anywhere in the document, including inside definitions, enumerations, and theorem environments. Use \emph{} sparingly. Let the math carry the weight.
- When presenting historical formalisms, use the author's original notation first, then map to modern RL notation.
- Focus on experiments and what practitioners did, not on what they "argued" or "proposed." Let the work speak.
- Prefer paragraphs and tables for presenting information. Use figures and graphs rarely, only when a visual representation is genuinely necessary (e.g., learning curves in simulations). Exposition should be carried by prose and math, not by diagrams.
- For simulation results and experiments: tables first, prose second. Present numerical findings in tables; keep surrounding prose minimal (1-2 sentences per table stating what it shows). Let the numbers speak.
- Always use rigorous math and formalism. Define all objects (sets, operators, functions) before using them. State assumptions explicitly. Use Definition/Theorem/Lemma environments for formal results.
- Relegate technical details (proof sketches, implementation specifics, auxiliary bounds, historical side notes) to footnotes. Keep the main text focused on the core argument and results.
- State facts objectively. Do not give comments or opinions; do not make good/bad judgments. Let results speak for themselves.

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
- Core: `numpy`, `scipy`, `matplotlib`, `tqdm`
- Deep RL (where needed): `torch`
- Set random seeds explicitly: `np.random.seed(42)` at top. Use a loop over seeds for multi-run experiments.
- Use descriptive variable names matching the math: `V` for value function, `Q` for Q-values, `pi` for policy.
- Inline comments for non-obvious steps. No boilerplate docstrings.
- Plots: `matplotlib` with clear labels, titles, legends. Use `fig, ax = plt.subplots()` pattern. Save at 300 dpi via `fig.savefig("filename.png", dpi=300, bbox_inches="tight")`.
- LaTeX table output: write `.tex` files containing `tabular` environments, directly includable with `\input{}`.

### Stdout Output Format
Each simulation script must also produce a `_stdout.txt` file capturing all console output. Run via:
```bash
python3 chXX_topic/sims/script_name.py > chXX_topic/sims/script_name_stdout.txt 2>&1
```

Stdout content requirements:
- **No opinions, only facts.** Report what the code did and what numbers it produced.
- **Copious tables.** Print parameter sweeps, results grids, and validation metrics in tabular format.
- **No subjective commentary.** Avoid words like "good", "bad", "impressive", "surprisingly". State results neutrally.
- **Structure:** Header with parameters → Experiment results (one line per configuration) → Summary statistics → Output file paths.

## Working Behavior

When working on this project, follow these principles:

- **Read papers thoroughly.** When asked to "read a paper" or "read a book", perform a deep read of the actual content. Use the Read tool to go through the file page by page or section by section. Do not just search for keywords and assume you have read it. Extract key definitions, theorems, algorithms, and notation. Summarize the main contributions and how they relate to the chapter being written.
- **Never declare victory.** Do not say "looks good", "works correctly", "all done", or similar. State what happened factually: "Script ran without errors. Output files: X, Y, Z."
- **Be skeptical of results.** If an algorithm "converges", ask why. If a number looks reasonable, verify it against theory. Do not trust outputs until independently checked.
- **State facts, not judgments.** Instead of "The Q-learning trajectory looks nice", say "Q-learning reached K=1.62 after 5000 iterations (K*=1.615)."
- **Show, don't tell.** Open figures for the user to inspect. Print numerical results. Do not describe what the user cannot see.
- **Verify before reporting.** Run the code. Check the output files exist. Confirm numerical values match expectations. Only then report completion.
- **When something fails, say so plainly.** Do not minimize or excuse failures. "Q-learning did not converge: final error 8.6, expected <0.1."

## File Tracking

- **This file** (`claude.md`): project context, style guide, master plan. Keep updated.
- **Changelog** (`changelog.md`): reverse-chronological log of structural changes. Update after each significant change.
- **Detailed restructure plan**: `docs/plans/2026-01-27-repo-restructure.md` (step-by-step tasks with commands)

## Key Commands

```bash
# Build full document (from docs/ dir)
cd docs && pdflatex -shell-escape main.tex && bibtex main && pdflatex -shell-escape main.tex && pdflatex -shell-escape main.tex

# Build single chapter (faster, from docs/ dir)
# Replace chXX and tex path as needed
cd docs && pdflatex -shell-escape -jobname=ch02_planning_learning "\def\chapterfile{../ch02_planning_learning/tex/unified_planning_learning}\input{compile_chapter}" && bibtex ch02_planning_learning && pdflatex -shell-escape -jobname=ch02_planning_learning "\def\chapterfile{../ch02_planning_learning/tex/unified_planning_learning}\input{compile_chapter}" && pdflatex -shell-escape -jobname=ch02_planning_learning "\def\chapterfile{../ch02_planning_learning/tex/unified_planning_learning}\input{compile_chapter}"

# Run a simulation script
python chXX_topic/sims/script_name.py
```

**Always compile after modifying any `.tex` file and show the PDF output path.** When working on a single chapter, prefer chapter compilation for faster iteration.
