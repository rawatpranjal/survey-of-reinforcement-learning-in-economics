# Audit: multi_echelon_paradigms (two-echelon serial supply chain)

Date: 2026-07-14. Fresh cold-cache audit by an independent Opus-tier agent
against the ch12 simulation-audit checklist, after the leakage-fix revision.

## Experiment

Two-echelon serial inventory system (retailer + upstream supplier, lead time
L=1, Poisson(5) demand, backorders, installation holding + retailer backorder
penalty). Physical observation is 6-dimensional; the optimal policy needs only
the 2 echelon inventory positions. Five paradigms, 20 seeds, 500 steps:

| Paradigm | Cumulative regret | Terminal cost/period |
|---|---|---|
| Oracle (Clark-Scarf base-stock) | 0 | 6.61 |
| Decentralized (local base-stock) | 1965 | 9.84 |
| NN World Model (learned model + base-stock search) | 3879 | 8.00 |
| Naive (running-mean order) | 29485 | 106.0 |
| Model-Free DQN | 44137 | 210.8 |

Result: the learned NN world model attains the best asymptotic policy of the
learners (terminal 8.00, within 21% of the provably optimal 6.61, ahead of the
decentralized heuristic's 9.84) but pays a larger exploration transient than
the no-learning heuristic on cumulative regret. Matches Gijsbrechts (2022):
learned inventory policies match specialized heuristics rather than dominating.

## Checklist verdict (all PASS)

1. Algorithm identity — Oracle is genuine echelon base-stock by coordinate
   search on the TRUE model. World model is a real two-head NN trained on
   observed transitions; planning rolls the LEARNED net forward and never
   touches the env. DQN is a standard target-net + Huber + replay DQN.
2. Environment fidelity — Poisson(5), L=1, cost = installation holding +
   backorder penalty, 6-dim obs. Matches the tex. Conservation tests pass.
3. Data integrity — cache values byte-identical to stdout and results.tex; full
   fresh run, no stale cache. Numbers computed, not hardcoded.
4. Comparison fairness — same env seed (same demand realization), same budget,
   same terminal protocol per paradigm; regret vs oracle on the matched seed.
5. Theoretical sanity — single-stage certification is independent (closed-form
   newsvendor critical fractile) and matches sim-opt (S=13, cost 3.06 vs 3.09).
   No learner beats the oracle. Rankings theory-consistent.
6. No information leakage — no learner reads env.lam, env.p, env.h, or the
   oracle S*; each estimates mean demand only from observed demand. Verified by
   test_no_leakage_learner_source and manual trace.
7. Seeds/reproducibility — 20 seeds fixed, means and SEs reported.

## Disclosed limitation (in the prose, not hidden)

The world model carries the Clark-Scarf base-stock policy class and the echelon
sufficient statistic; the DQN does not. The comparison is therefore partly
structured-vs-unstructured, and the tex says so. The world model wins on
terminal (asymptotic) cost; the decentralized heuristic wins on cumulative
regret because it needs no exploration.

## Bullshit score: 25%

Reviewer 2 catches that the DQN finishes below even the naive-order floor
(genuine undertraining on a 25-action grid in 500 steps, not a bug) and a
couple of loose phrasings, since fixed. The substance survives: the oracle is
independently certified, no learner leaks the true parameters, the world model
genuinely learns and plans on its learned net, and every reported number is
computed and matches across cache, stdout, table, and prose.

Prior revision scored 50% (learners read the true demand rate and the world
model was initialized near the true-lambda oracle point); fixed by having every
learner estimate mean demand from observed data only.
