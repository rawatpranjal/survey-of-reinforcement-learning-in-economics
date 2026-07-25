# Chapters 10b and 10c reconciliation verification

Date: 2026-07-24

Verified content commit: `904afae7106a42ef2774a368c80efc416bec949c`,
pushed to `origin/ch10-causal-reconciliation`.

## Topology

The final branch merges both descendants of `df4c74f`.

- `6ebee7c` contributes the independently checked ch10b citation, positivity,
  float-reference, and stdout fixes.
- `bf4e41e` contributes the primary-sourced ch10c rebuild and causal-bandit
  simulation corrections.
- The dirty local `main` worktree was not used or modified.

## Source and proof checks

- Lewis and Syrgkanis arXiv source `2002.07285` confirms distinct asymptotic
  normality results for the partially linear model and the SNMM extension.
- Jiang and Li arXiv source `1511.03722` states its Cramer-Rao result for
  discrete tree MDPs.
- The DTR theorem now defines the general K-stage g-formula and
  `J_M(pi)`, derives their equality by backward kernel integration, and states
  its almost-sure comparison for each fixed supported regime. It names the
  unrestricted measurable class and keeps its Bellman optimizer distinct from
  the restricted policy-learning target used later in the chapter.
- The policy-learning prose identifies the DGP-informed outcome feature map as
  an oracle correct-specification benchmark.

## Numerical verification

All six simulation entry points ran from empty caches. Every generated table and
figure was byte-identical to the previously committed artifact. The refreshed
stdout files record computation rather than cache hits. Ch10c's embedded checks
passed for hardness, arm values, budgets, Table 1 marginals, and the published
MABUC targets.

All six simulation sources pass Ruff. Regenerating the two artifacts affected
by lint-only source cleanup left their hashes unchanged.

## Build and visual verification

- Standalone ch10b builds to 41 pages. Its only undefined references are the
  expected cross-chapter links absent from the standalone wrapper.
- Standalone ch10c builds to 6 pages.
- The full manuscript builds to 305 pages with no undefined citation,
  undefined reference, LaTeX error, or minted failure.
- The DTR theorem statement, proof, oracle-benchmark disclosure, ch10c figure,
  tables, and float order were inspected as rendered pages.

## Independent gates

- A proof-blind verifier found and then rechecked two scope mismatches: the
  unrestricted Bellman optimizer now has its own notation, and its recursion
  invokes all-action positivity. The verifier's final verdict was PASS with no
  remaining correctness blocker.
- A separate shipping verifier confirmed the merge topology, cold-run evidence,
  generated artifacts, lint checks, and builds. An intermediate pass stopped
  because the verified changes were not yet committed. The exact committed
  state then passed the final gate with no remaining shipping blocker.
