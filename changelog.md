# Changelog

Reverse-chronological log of structural changes to the survey repo.

---

## 2026-01-27: Initial Restructure Plan

**What:** Designed chapter-based repo structure for "Survey of Reinforcement Learning in Economics" arXiv paper.

**Chapters defined:**
- ch00: Introduction
- ch01: Historical Developments
- ch02: Planning and Learning (DP vs RL)
- ch03: RL for Structural Estimation
- ch04: Inverse RL
- ch05: RL in Games (MARL)
- ch06: Bandits & Online Learning
- ch07: Applications
- ch08: RLHF & Preference Learning
- ch09: Conclusion

**Decisions made:**
- Each chapter gets `tex/`, `papers/`, `sims/` subdirs
- RLHF added as ch08 (was cut from ORE project)
- Bandits split out as separate chapter from old "Economic Models for RL" section
- "Economic Models for RL" chapter placement deferred
- Old repo dirs (`qlearning/`, `ppo/`, `md/`, etc.) archived
- `docs/` holds combined LaTeX main file
- Sister survey ORE_main referenced but not duplicated
- Created `claude.md` with writing style, notation, simulation standards

**Files created:**
- `claude.md` -- project context and style guide
- `changelog.md` -- this file
- `docs/plans/2026-01-27-repo-restructure.md` -- detailed implementation plan

**Pending:** Execute the 10-task restructure plan (directory skeleton, file distribution, archiving, main.tex rewrite, .gitignore, README).
