# Repo Restructure: Survey of Reinforcement Learning in Economics

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reorganize the RL research repo into a chapter-based structure for a broad arXiv survey paper "Survey of Reinforcement Learning in Economics", with one folder per chapter containing LaTeX, papers, and a main simulation.

**Architecture:** Each chapter gets a folder (`ch00_introduction/`, `ch01_history/`, etc.) with subdirs for `tex/`, `papers/`, and `sims/`. A top-level `docs/` folder holds the combined LaTeX main file. An `archive/` folder preserves the original loose repo structure. Meta files (`plan.md`, `changelog.md`, `claude.md`) live at repo root.

**Tech Stack:** LaTeX, Python (numpy/scipy/matplotlib), Jupyter notebooks

---

## Chapter Structure

```
ch00_introduction/       -- Abstract + Intro (no sim)
ch01_history/            -- Historical Developments (no sim)
ch02_planning_learning/  -- DP vs RL, unified view (sim: gridworld, bus engine)
ch03_rl_structural_est/  -- RL for Structural Estimation (sim: bus engine DP vs DQN)
ch04_inverse_rl/         -- Inverse RL (sim: IRL LP gridworld)
ch05_rl_in_games/        -- MARL, game theory, equilibria (sim: korean auction, NFSP)
ch06_bandits/            -- Bandits & online learning (sim: TBD)
ch07_applications/       -- Real world applications & deployments (no sim, survey)
ch08_rlhf/               -- RLHF & Preference Learning (sim: TBD)
ch09_conclusion/         -- Conclusion & discussion (no sim)
```

Note: "Economic Models for RL" chapter is deferred -- will plan placement later.

## Sister Survey Reference

The **ORE_main** project at `/Users/pranjal/Code/rl/ORE_main` is the narrower sister paper:
- **Title:** "Structural Econometrics and Inverse Reinforcement Learning: Inferring preferences and beliefs from human behavior" (Rust & Rawat, Jan 2026)
- **Scope:** DDC estimation + IRL (Introduction → MDP Background → DDC Estimation → IRL → Conclusion)
- **Relationship:** Our survey deliberately goes *broader* -- covering RL in economics writ large (bandits, MARL, RLHF, applications, etc.). We avoid deep overlap with ORE_main's DDC/IRL focus but can learn from their presentation style. Our survey is lighter in tone, takes more space, includes worked examples and simulations throughout.

## File Mapping: What Goes Where

| Source | Destination |
|--------|-------------|
| `ore_project/0_abstract.tex` | `ch00_introduction/tex/abstract.tex` |
| `ore_project/1_intro.tex` | `ch00_introduction/tex/intro.tex` |
| `ore_project/2_history.tex` | `ch01_history/tex/history.tex` |
| `ore_project/3_example1.tex` | `ch02_planning_learning/tex/planning_learning.tex` |
| `ore_project/3_example.tex` | `ch02_planning_learning/tex/planning_learning_alt.tex` |
| `original_sims/01_gridworld.ipynb` | `ch02_planning_learning/sims/gridworld.ipynb` |
| `original_sims/02_bus_engine.ipynb` | `ch02_planning_learning/sims/bus_engine_intro.ipynb` |
| `ore_project/4_rl_in_se.tex` | `ch03_rl_structural_est/tex/rl_in_se.tex` |
| `original_sims/04_bus_engine.ipynb` | `ch03_rl_structural_est/sims/bus_engine_dp_vs_dqn.ipynb` |
| `ore_project/5_irl.tex` | `ch04_inverse_rl/tex/irl.tex` |
| `inv/` (all files) | `ch04_inverse_rl/sims/` |
| `inv.ipynb` | `ch04_inverse_rl/sims/irl_notebook.ipynb` |
| `ore_project/6_marl.tex` | `ch05_rl_in_games/tex/marl.tex` |
| `korean/` (all notebooks) | `ch05_rl_in_games/sims/korean/` |
| `fp/` (all notebooks) | `ch05_rl_in_games/sims/fictitious_play/` |
| `NFSP-FQI.ipynb` | `ch05_rl_in_games/sims/nfsp_fqi.ipynb` |
| `original_sims/03_auctions.ipynb` | `ch05_rl_in_games/sims/auctions.ipynb` |
| `original_sims/03_dynamic_game.ipynb` | `ch05_rl_in_games/sims/dynamic_game.ipynb` |
| `ore_project/7_se_in_rl.tex` | split: bandits content to `ch06_bandits/tex/`, remainder to `archive/` for later |
| `ore_project/8_apps.tex` | `ch07_applications/tex/applications.tex` |
| `rlhf/RLHF.tex` | `ch08_rlhf/tex/rlhf.tex` |
| `ore_project/9_conclusion.tex` | `ch09_conclusion/tex/conclusion.tex` |
| `ore_project/ore.tex` | `docs/main.tex` (rewritten to point to new paths) |
| `ore_project/refs.bib` | `docs/refs.bib` |
| `ore_project/refs.bib2` | `docs/refs_extended.bib` |
| `ore_project/econometrica.bst` | `docs/econometrica.bst` |
| `ore_project/figs/` | `docs/figs/` |
| `ore_project/slides/` | `archive/slides/` |
| `qlearning/` | `archive/qlearning/` (large notebooks, reuse as needed) |
| `ppo/` | `archive/ppo/` (large notebooks, reuse as needed) |
| `md/` | `archive/md/` |
| `original_sims/active_firms_exp1.png` | `ch05_rl_in_games/sims/active_firms_exp1.png` |
| `original_sims/LICENSE`, `README.md` | `archive/original_sims_meta/` |
| `ore_project.zip` | `archive/ore_project.zip` |
| `rawat_rust_ore_main.pdf` | `archive/rawat_rust_ore_main.pdf` |

---

### Task 1: Create directory skeleton

**Files:**
- Create: All chapter dirs with `tex/`, `papers/`, `sims/` subdirs
- Create: `docs/`, `archive/`

**Step 1: Create all directories**

```bash
# Chapter dirs
for ch in ch00_introduction ch01_history ch02_planning_learning ch03_rl_structural_est ch04_inverse_rl ch05_rl_in_games ch06_bandits ch07_applications ch08_rlhf ch09_conclusion; do
  mkdir -p "$ch"/{tex,papers,sims}
done

# Top-level dirs
mkdir -p docs/figs
mkdir -p archive/slides archive/original_sims_meta
```

**Step 2: Verify structure**

```bash
find . -type d -not -path './.git/*' | sort
```

Expected: All chapter dirs with tex/papers/sims subdirs visible.

**Step 3: Commit**

```bash
# Add .gitkeep to empty dirs so git tracks them
find . -type d -empty -not -path './.git/*' -exec touch {}/.gitkeep \;
git add .
git commit -m "feat: create chapter-based directory skeleton for RL in Economics survey"
```

---

### Task 2: Unzip and distribute ORE LaTeX files

**Files:**
- Source: `ore_project.zip` (already unzipped to `/tmp/ore_project/`)
- Destinations: per mapping table above

**Step 1: Copy LaTeX chapter files to chapter dirs**

```bash
cp /tmp/ore_project/0_abstract.tex ch00_introduction/tex/abstract.tex
cp /tmp/ore_project/1_intro.tex ch00_introduction/tex/intro.tex
cp /tmp/ore_project/2_history.tex ch01_history/tex/history.tex
cp /tmp/ore_project/3_example1.tex ch02_planning_learning/tex/planning_learning.tex
cp /tmp/ore_project/3_example.tex ch02_planning_learning/tex/planning_learning_alt.tex
cp /tmp/ore_project/4_rl_in_se.tex ch03_rl_structural_est/tex/rl_in_se.tex
cp /tmp/ore_project/5_irl.tex ch04_inverse_rl/tex/irl.tex
cp /tmp/ore_project/6_marl.tex ch05_rl_in_games/tex/marl.tex
cp /tmp/ore_project/7_se_in_rl.tex ch06_bandits/tex/se_in_rl_full.tex  # full file, split later
cp /tmp/ore_project/8_apps.tex ch07_applications/tex/applications.tex
cp /tmp/ore_project/9_conclusion.tex ch09_conclusion/tex/conclusion.tex
```

**Step 2: Copy RLHF tex**

```bash
cp rlhf/RLHF.tex ch08_rlhf/tex/rlhf.tex
```

**Step 3: Copy docs-level files (bib, bst, figs, main tex)**

```bash
cp /tmp/ore_project/refs.bib docs/refs.bib
cp /tmp/ore_project/refs.bib2 docs/refs_extended.bib
cp /tmp/ore_project/econometrica.bst docs/econometrica.bst
cp /tmp/ore_project/figs/* docs/figs/
cp /tmp/ore_project/ore.tex docs/main.tex
```

**Step 4: Archive slides**

```bash
cp -r /tmp/ore_project/slides/* archive/slides/
```

**Step 5: Commit**

```bash
git add .
git commit -m "feat: distribute ORE LaTeX files into chapter structure"
```

---

### Task 3: Move existing simulations into chapter dirs

**Step 1: Move original_sims into chapters**

```bash
cp original_sims/01_gridworld.ipynb ch02_planning_learning/sims/gridworld.ipynb
cp original_sims/02_bus_engine.ipynb ch02_planning_learning/sims/bus_engine_intro.ipynb
cp original_sims/04_bus_engine.ipynb ch03_rl_structural_est/sims/bus_engine_dp_vs_dqn.ipynb
cp original_sims/03_auctions.ipynb ch05_rl_in_games/sims/auctions.ipynb
cp original_sims/03_dynamic_game.ipynb ch05_rl_in_games/sims/dynamic_game.ipynb
cp original_sims/active_firms_exp1.png ch05_rl_in_games/sims/active_firms_exp1.png
```

**Step 2: Copy inv/ sims**

```bash
cp -r inv/* ch04_inverse_rl/sims/
cp inv.ipynb ch04_inverse_rl/sims/irl_notebook.ipynb
```

**Step 3: Copy games sims (korean, fp, NFSP)**

```bash
cp -r korean/* ch05_rl_in_games/sims/korean/
mkdir -p ch05_rl_in_games/sims/fictitious_play
cp -r fp/* ch05_rl_in_games/sims/fictitious_play/
cp NFSP-FQI.ipynb ch05_rl_in_games/sims/nfsp_fqi.ipynb
```

**Step 4: Commit**

```bash
git add .
git commit -m "feat: distribute simulations into chapter sims/ folders"
```

---

### Task 4: Archive old top-level files

**Step 1: Move old dirs and files to archive**

```bash
# Move large notebook collections to archive
cp -r qlearning archive/qlearning
cp -r ppo archive/ppo
cp -r md archive/md
cp original_sims/LICENSE archive/original_sims_meta/LICENSE
cp original_sims/README.md archive/original_sims_meta/README.md
cp ore_project.zip archive/ore_project.zip
cp rawat_rust_ore_main.pdf archive/rawat_rust_ore_main.pdf
```

**Step 2: Remove old top-level dirs (after verifying copies)**

```bash
rm -rf qlearning ppo md korean fp inv original_sims rlhf
rm ore_project.zip rawat_rust_ore_main.pdf
rm inv.ipynb NFSP-FQI.ipynb
rm -rf .ipynb_checkpoints
```

**Step 3: Commit**

```bash
git add -A
git commit -m "chore: archive old top-level dirs, remove duplicates"
```

---

### Task 5: Rewrite docs/main.tex to point to new chapter paths

**Files:**
- Modify: `docs/main.tex`

**Step 1: Update all \input paths in main.tex**

Replace the `\input{}` lines to point to new chapter locations using relative paths from `docs/`:

```latex
\input{../ch00_introduction/tex/abstract.tex}   % was 0_abstract
\input{../ch00_introduction/tex/intro}           % was 1_intro
\input{../ch01_history/tex/history}              % was 2_history
\input{../ch02_planning_learning/tex/planning_learning}  % was 3_example1
\input{../ch03_rl_structural_est/tex/rl_in_se}   % was 4_rl_in_se
\input{../ch04_inverse_rl/tex/irl}               % was 5_irl
\input{../ch05_rl_in_games/tex/marl}             % was 6_marl
\input{../ch06_bandits/tex/se_in_rl_full}        % was 7_se_in_rl (temp, will split)
\input{../ch07_applications/tex/applications}    % was 8_apps
\input{../ch08_rlhf/tex/rlhf}                   % NEW
\input{../ch09_conclusion/tex/conclusion}        % was 9_conclusion
```

Also update:
- Title to "Survey of Reinforcement Learning in Economics"
- Add RLHF section before Conclusion
- Add Bandits as separate section

**Step 2: Commit**

```bash
git add docs/main.tex
git commit -m "feat: rewrite main.tex with new chapter paths and updated title"
```

---

### Task 6: Create plan.md

**Files:**
- Create: `plan.md` (repo root)

Contents: High-level master plan describing the survey's scope, chapter outline with one-line descriptions, what each chapter's main simulation is, and current status. This is the living roadmap.

**Step 1: Write plan.md** (see Task 6 content below in implementation)

**Step 2: Commit**

```bash
git add plan.md
git commit -m "docs: add master plan for survey"
```

---

### Task 7: Create changelog.md

**Files:**
- Create: `changelog.md` (repo root)

Reverse chronological, surgical additions. Start with today's restructure entry.

**Step 1: Write changelog.md**

**Step 2: Commit**

```bash
git add changelog.md
git commit -m "docs: add changelog"
```

---

### Task 8: Create claude.md

**Files:**
- Create: `claude.md` (repo root)

Detailed project context for Claude: writing style (academic, arXiv-ready, econometrics conventions), project structure, chapter layout, simulation standards, coding conventions, references to plan.md and changelog.md.

**Step 1: Write claude.md**

**Step 2: Commit**

```bash
git add claude.md
git commit -m "docs: add claude.md project guide"
```

---

### Task 9: Create .gitignore

**Files:**
- Create: `.gitignore`

Ignore `.DS_Store`, `__pycache__`, `.ipynb_checkpoints`, `*.pyc`, LaTeX build artifacts (`*.aux`, `*.log`, `*.bbl`, `*.blg`, `*.out`, `*.toc`, `*.synctex.gz`).

**Step 1: Write .gitignore**

**Step 2: Commit**

```bash
git add .gitignore
git commit -m "chore: add .gitignore"
```

---

### Task 10: Update README.md

**Files:**
- Modify: `README.md`

Replace the single-line README with a proper project description: title, authors, chapter listing, how to build the LaTeX, how to run simulations.

**Step 1: Write README.md**

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: update README with survey structure"
```

---

## Summary of Commits

1. `feat: create chapter-based directory skeleton for RL in Economics survey`
2. `feat: distribute ORE LaTeX files into chapter structure`
3. `feat: distribute simulations into chapter sims/ folders`
4. `chore: archive old top-level dirs, remove duplicates`
5. `feat: rewrite main.tex with new chapter paths and updated title`
6. `docs: add master plan for survey`
7. `docs: add changelog`
8. `docs: add claude.md project guide`
9. `chore: add .gitignore`
10. `docs: update README with survey structure`

## Open Items (Deferred)

- **Economic Models for RL** chapter: decide placement and content split from `7_se_in_rl.tex`
- **Bandits sim**: create or identify a main simulation
- **RLHF sim**: create a DPO/RLHF demonstration notebook
- **Per-chapter paper lists**: populate `papers/` dirs with key references
- **Code cleanup**: standardize notebook style, add docstrings, consistent imports
