# Copilot review instructions

This repository is an arXiv survey, "Reinforcement Learning in Economics." It is LaTeX prose in `chXX_*/tex/` plus self-contained Python simulation scripts in `chXX_*/sims/`. Review pull requests with the priorities below, in order.

## 1. Correctness of the diff

Read the changed lines against what they claim to do. For a simulation, check that the algorithm in the code matches the algorithm named in the prose and in the cited paper (update rule, state and action spaces, transition, reward, discounting). Flag placeholder or simplified implementations that change an algorithm's character, hardcoded "expected" numbers reported as if computed, and any result that contradicts a known analytical bound.

## 2. Tests, especially end-to-end

This is the priority for simulation changes. Every simulation script splits into `compute_data(force=None)` and `generate_outputs(data)`. Demand an end-to-end test that runs the real pipeline on a small config and asserts the invariants, not a unit test of one helper. A good e2e test for a sim:

- runs `compute_data()` then `generate_outputs()` on a reduced seed count and horizon,
- asserts the declared output files are written (the `.png` figures and `.tex` tables),
- asserts the oracle or DP baseline ranks the policies correctly under good coverage (the known-good sanity check),
- asserts the cache config matches the current parameters, so stale results cannot pass,
- asserts no NaN or inf in the reported metrics.

If a PR adds or changes a sim and does not add or update such a test, say so and propose the specific test. When you can, propose the test as a concrete code suggestion.

## 3. Simulation audit checklist

Apply these to any simulation change and name the ones the diff fails: algorithm identity matches the cited paper, environment and MDP match the tex writeup, results are actually computed and not loaded from a stale cache, comparisons are apples to apples (same env instance, same seeds, same evaluation protocol), results align with known bounds, and no information leakage (the agent must not see the true reward, transition, or optimal policy during learning unless the method requires it).

## 4. Prose and citations

For `.tex` changes: no em-dashes and no colons used to splice or extend a sentence in prose (a colon before a list is fine). No `\textbf{}`. Every `\cite*{}` key must resolve in `docs/refs.bib`. Flag a citation whose key is not in the bib, and flag a claim attributed to a paper that the surrounding text does not support.

Keep review comments concrete and tied to a line. Prefer a code suggestion over a description when the fix is mechanical.
