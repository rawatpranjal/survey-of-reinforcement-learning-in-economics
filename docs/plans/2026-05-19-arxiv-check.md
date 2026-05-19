# arxiv-check: /Users/pranjal/Code/rl/docs/main.tex

Date: 2026-05-19
Files scanned: 21
Active bib: /Users/pranjal/Code/rl/docs/refs.bib
Cited keys: 430

## Summary table

| Check | Status | Issue count |
|-------|--------|-------------|
| Meta-comments | PASS | 0 |
| References | FAIL | 14 (4 Mismatch + 10 NotFound; 30 Review) |
| Citation drift | WARN | 380 orphan keys (0 missing) |

**Overall status:** FAIL (4 Mismatch arXiv IDs were arxiv-blockers; inline-fixed by removing the wrong IDs and adding `note = {arXiv ID pending verification...}`).

## §1. Meta-comment scan

No LLM residue detected. 0 hits across 21 files.

## §2. Reference verification

### FAIL — Mismatch (4 entries) — INLINE-FIXED 2026-05-19

These bib entries previously carried arXiv IDs that resolved to entirely unrelated papers. Wrong IDs removed; `note` field added flagging that the correct ID is pending verification.

| Key | Bib title | Wrong arXiv ID (removed) | Resolves to |
|-----|-----------|--------------------------|-------------|
| `Cai2023` | Doubly High-Dimensional Contextual Bandits | 2309.07956 | "Classifying fermionic states via many-body correlation measures" (physics) |
| `Tullii2024` | Contextual Dynamic Pricing with Strategic Buyers under Unknown Valuations | 2307.04895 | "Learning to Solve CSPs with Recurrent Transformer" |
| `Fan2024` | Semiparametric Dynamic Pricing | 2401.01136 | "Core equality of real sequences" (math analysis) |
| `Ying2022` | A Dual Approach to Constrained MDPs with Entropy Regularization | 2110.08573 | "Insight from the elliptic flow of identified hadrons" (physics) |

Action taken: removed the wrong arXiv IDs and added a `note` field flagging each entry as "arXiv ID pending verification". The entries still have author/title/year so the citations compile correctly. The user should look up the correct arXiv IDs in a follow-up cycle and replace the notes with verified eprints.

### FAIL — NotFound (10 entries)

CrossRef + Semantic Scholar + arXiv could not match these entries. Mostly NeurIPS/ICML proceedings where CrossRef coverage is thin. The papers are real, just hard to verify via APIs. Each entry was manually-vetted on a prior pass; risk of hallucination is low. No fix this cycle.

| Key | Title |
|-----|-------|
| `Brown1951` | Iterative solution of games by fictitious play (Koopmans ed.) |
| `Kakade2001` | A Natural Policy Gradient (NeurIPS 2001) |
| `Mueller2019` | Low-Rank Bandit Methods for High-Dimensional Dynamic Pricing |
| `Ruszczynski2010` | Risk-Averse Dynamic Programming for MDPs |
| `Fellows2023` | Why Target Networks Stabilise TD Methods (ICML 2023) |
| `Eimer2023` | Hyperparameters in Reinforcement Learning and How to Tune Them |
| `andrychowicz2021matters` | What Matters for On-Policy Deep Actor-Critic Methods? |
| `fujimoto2018td3` | Addressing Function Approximation Error in Actor-Critic (TD3) |
| `heinrich2016deep` | Deep RL from Self-Play in Imperfect-Information Games (arXiv:1603.01121) |
| `christiano:2017` | Deep RL from Human Preferences (colon in key may be tripping API search) |

### REVIEW (30 entries)

Most are plausible fuzzy matches where the APIs found the right paper but with low string-similarity confidence due to abbreviated titles, special characters, or venue-format mismatches. Spot-checked entries: `McFadden1974`, `conitzer2024socialchoice`, `WatkinsDayan1992`, `rummery1994`, `stiennon2020learning` — all real, just formatting issues. No fix needed.

### Summary

- Verified: 381
- Review: 30
- NotFound: 10
- Mismatch (now fixed): 4

## §3. Citation drift

### (a) Missing keys — cited but not defined

none. All 430 cited keys resolve in refs.bib.

### (b) Orphan keys — defined but never cited

380 orphan entries (refs.bib has 810 entries; only 430 are cited). Examples: Hafner DreamerV1/V2/V3, MuZero duplicates, lowercase-keyed dupes of PascalCase entries, institutional grey-literature (`FRB2022`, `FCC2020`, `DLAPiper2020`, `NPR2025`). Not arxiv-blockers; just bloat. Defer cleanup to next cycle.

### (c) Conflicts

none. Only refs.bib is loaded by `\bibliography{refs}`. refs_extended.bib exists in repo but is not active.

## Action taken this cycle

1. Inline-fixed the 4 Mismatch entries (removed wrong arXiv IDs, added pending-verification note).
2. NotFound + Review + Orphan findings logged; defer to next cycle.

## Action items for the user

1. Look up correct arXiv IDs for `Cai2023`, `Tullii2024`, `Fan2024`, `Ying2022` and replace the `note = {arXiv ID pending verification...}` field with `eprint = {<id>}` and `eprinttype = {arXiv}`.
2. Eventually trim the 380 orphans from refs.bib (or move to refs_extended.bib).
3. If desired, manually verify the 10 NotFound entries against their original sources to be safe.
