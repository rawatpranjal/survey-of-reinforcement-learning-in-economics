# ch10b independent audit, 2026-07-24

**Original commit audited:** `df4c74f` on branch `ch10b-reconciled`, which was
17 commits ahead of `origin/main` and unpushed at the time.
**Original PR note:** This verdict did not apply to PR #38. Its head `f1fac64`
was four commits behind `df4c74f` and did not contain the simulation
corrections.

**Reconciliation update:** The proof gap and the policy-learning classification below were resolved
on `ch10-causal-reconciliation`, with verified content commit `904afae` pushed
to the remote branch. The original findings remain in the record, followed by
the adjudication that changed the final verdict.

**Evidence admitted:** cold re-runs of all five simulations on this machine, verbatim spans from the
authors' own arXiv LaTeX sources, a mechanical number-provenance check, and rulings from fresh agents
that were never told what conclusion to reach.

**Evidence excluded:** `ch10b_rl_for_ci/notes/*.md` and `fit_evaluation.md`. Those were written by the
sessions that wrote the chapter, so they are self-certification, not verification. No agent in this
audit was allowed to read them.

---

## Verdict

| Axis | Verdict | Evidence |
|---|---|---|
| Theorem statement | **PASS** | An adversarial reviewer given the statement without the proof found no counterexample. |
| Theorem proof | **PASS after repair** | The general g-formula and `J_M` are now defined, the kernel induction is explicit, the almost-sure claim is regime-specific, and the unrestricted Bellman target is distinct from the later restricted policy class. |
| Attributed results, 6 checked | **PASS after 2 fixes** | Both fixes applied. See below. |
| Literature numbers, 40+ checked | **PASS** | Every figure from STAR\*D, HeartSteps, the ADHD SMART, Jaman and Project STAR matches its source. |
| Number provenance | **PASS after 1 fix** | 84 of 86 numeric literals traced to an owning artifact; the two exceptions resolved. |
| Reproducibility | **PASS** | All five scripts cold-recomputed. Every table fragment and every figure byte-identical. |
| Simulation gates, 7 per script | **PASS with one oracle benchmark disclosed** | The policy-learning feature map uses the stated DGP constants. It is now labelled as an oracle correct-specification benchmark rather than a feasible nuisance learner. |
| Build and exposition | **PASS after fixes** | 41 pages, zero undefined citations, zero overfull boxes. |

The reconciled chapter has no confirmed blocker. Every number reproduces, the attribution fixes were
re-derived from primary-source LaTeX, the theorem proof now establishes every displayed identity, and
the controlled oracle specification in the policy-learning simulation is disclosed in the prose.

---

## What was wrong, and what was done about it

### 1. The SNMM asymptotic-normality result was cited as Theorem 4. It is Theorem 8. **Fixed.**

The subsection is about structural nested mean models, and it attributed asymptotic normality to
Theorem 4 of Lewis and Syrgkanis. Verified against the authors' arXiv LaTeX (`2002.07285`), the paper
carries two distinct results: `\begin{theorem}[Asymptotic Normality and Inference]\label{thm:normality}`
for the partially linear Markovian process, and, inside the section headed *Generalization to
Structural Nested Mean Models*, `\begin{theorem}[Asymptotic Normality]\label{thm:normality-snmm}`,
whose first clause reads "for a Structural Nested Mean Model". Corollary 9, which the chapter cites
correctly a few paragraphs later, is stated "Under the assumptions and definitions of Theorem 8", so
the chapter's own citation chain was internally inconsistent.

The chapter now cites Theorem 8 for the SNMM case and Theorem 4 for the partially linear
specialization the simulation actually uses. Both pointers are now right, and the simulation's own
setting is named.

### 2. Jiang and Li's lower bound holds for *discrete* tree MDPs. **Fixed.**

The chapter said "tree-structured MDPs". The authors' LaTeX (`1511.03722`, `paper.tex:269`) reads "For
discrete tree MDPs", and their Definition 1 requires discrete observations and actions, unit discount,
and terminal-only rewards. The qualifier is now restored. The rest of that passage checks out exactly:
Theorem 1 is an exact variance recursion and not a bound, Observation 1 is an equality and not an
approach, and per-decision importance sampling really is the special case with the value model set to
zero.

### 3. The theorem's positivity condition is stronger than the assumption it points back to. **Fixed.**

The theorem introduced its displayed positivity condition with "In particular,", which frames a
strengthening as a specialization.

**Provenance of this finding, stated honestly.** One reviewer raised it unprompted: its brief never
mentioned positivity, and it listed "two nonequivalent positivity conditions" under its own
objections. A second reviewer, denied the proof, *was* asked whether the two conditions differ and
whether the difference matters, so its agreement is not an independent vote and is not counted as
one. What that second reviewer contributed is a counterexample, which stands or falls on its own
logic regardless of who suggested looking, and which is checkable directly. Assumption 4 is target-policy positivity, quantified only
over actions a target regime assigns. The theorem needs positivity for *every* action, because its
optimality claim ranges over all continuation regimes.

This is not cosmetic. The proof-blind reviewer built the instance that separates them: two stages,
`P(A_1 = 1) = 1`, potential outcome `Y(a_1, ·) = 1 - a_1`. Assumption 4 holds for the target regime,
every other hypothesis holds, and the recursion returns a value of zero where the true maximum over
regimes is one, because `Q_1(h_1, 0)` conditions on a null event. The displayed condition excludes
that instance; assumption 4 alone does not. The wording now says the strong form is required and why.

### 4. A number in the prose was correct but traceable to nothing. **Fixed at the source.**

The continuous design's treatment contrast is described as changing sign at 0.347. That figure
appeared in no simulation output and in no script literal. It is in fact correct: the script computes
the same quantity at `dtr_qlearning_vs_murphy.py:679`, and solving the contrast for zero by hand gives
0.3466. The problem was that the committed `_stdout.txt` never contained it.

The cause turned out to be worse than a missing print. The committed stdout for that script was
generated by an **older version of the script committed alongside it**. Re-running produced six lines
that were absent from the committed artifact, including the oracle values, the fitted stage-1 rule,
the threshold, and the independent Monte Carlo cross-check of the Gauss-Hermite oracle
(`|diff| / MC SE = 1.20`). Script and artifact were out of sync at the same commit. The regenerated
stdout is now committed, and the number traces.

### 5. Seven floats had no prose anchor. **Fixed.**

Four of five figures and three of seven tables carried labels that were never referenced anywhere,
while the prose said "Panel A shows" and "Panel B shows" with nothing to point at. All seven now have
a reference at the sentence that discusses them. Three simulation figures also had no `\FloatBarrier`;
they do now, taking the count from three to six against twelve floats. The remaining unreferenced
labels are equations and subsection hooks, which are harmless.

---

## Policy-learning feature map, adjudicated

The initial audit classified `dtr_policy_learning.py` as information leakage because `q1_basis` uses
the true transition constants in its `correct=True` branch:

```python
mean = RHO * x + TREAT_SHIFT * A1
u = (c2 - mean) / SIGMA_ETA
prob_below = norm.cdf(u)
truncated_first_moment = mean * prob_below - SIGMA_ETA * norm.pdf(u)
cols += [A1 * x, prob_below, truncated_first_moment]
```

That code does not pass future outcomes, an oracle policy, or fitted oracle coefficients into the
learner. It supplies the analytically correct feature map for a controlled correct-specification
cell, and each fold still estimates its regression coefficients from training observations. This is
an oracle specification benchmark, not a feasible end-to-end nuisance learner. The original
"leakage" label therefore overstated the defect. The real problem was disclosure. The chapter now
states that the feature map uses the generating transition law and constants, and it also states that
the wrong outcome branch removes treatment-effect heterogeneity rather than merely dropping generic
interactions. No result number or algorithm was changed in this adjudication.

`dynamic_dml_snmm.py` passes all seven gates. `ope_estimators.py` passes all seven, with a
longhand reimplementation of every estimator agreeing to machine precision and the misspecified cell
confirmed genuinely non-nesting. Two notes there: the standard-error calibration panel is computed at
horizon 8 rather than the horizon 16 used everywhere else, because trajectory importance sampling is
not yet Gaussian at 16 (coverage 0.83 against nominal), and the caption states the cell without
stating why. And one double-robustness gate tolerates a bias of a fifth of the estimand, so that leg
is verified only loosely.

---

## The proof gap, closed in reconciliation

The audit correctly found that `J_M(π)` was undefined, the general g-formula was not displayed, and
the proof asserted their equality without the kernel calculation. The repair fixes regular
conditional versions on the standard Borel histories, writes the full iterated integral, defines
`J_M(π)` as expected terminal reward in the induced MDP, and derives the identity by backward
integration. The identification induction now names the consistency, sequential-ignorability, and
tower-property step at every stage. The optimization statement is narrowed to each fixed supported
regime, with the almost-sure null set allowed to depend on that regime, and equality is proved for the
measurable tie-broken optimizer. The unrestricted Bellman optimizer is distinguished from the later
restricted policy-learning target, and the recursion invokes all-action rather than target-policy
positivity. The proof no longer intersects null sets over an uncountable policy class.

---

## Reproducibility

Every script was re-run cold in a fresh worktree with no cache present. All five recomputed from
scratch, and **every generated table fragment and every figure came back byte-identical**. The only
files that changed were the stdout logs, and in four of five cases the only changes were the absolute
output paths.

Runtime, measured here for the first time (11 cores, 18 GB, threads capped, `nice`d):

| Script | Cold runtime |
|---|---|
| `dtr_dags.py` | 0.8 s |
| `dtr_policy_learning.py` | 13.2 s |
| `ope_estimators.py` | 16.2 s |
| `dtr_qlearning_vs_murphy.py` | 499.5 s |
| `dynamic_dml_snmm.py` | 668.3 s |

The full ch10b suite is about 20 minutes cold. There was never a compute problem here.

---

## A systemic issue worth fixing beyond this chapter

Every converted source under `ch10b_rl_for_ci/papers/` has had its display equations stripped by the
PDF-to-markdown conversion and replaced with the literal placeholder `formula-not-decoded`:

| Source | Lines | Stripped equations |
|---|---|---|
| `lewis2021dml.md` | 1802 | 314 |
| `kitagawa2018who.md` | 1728 | 253 |
| `sakaguchi2024dynamicpolicy.md` | 1166 | 146 |
| `xie-2019-marginalized-is-ope.md` | 897 | 106 |
| `jiang-2016-doubly-robust-ope.md` | 488 | 39 |
| `schulte2014qlearning.md` | 479 | 32 |
| `liao-2021-long-term-ope-mobile-health.md` | 1057 | 0 |

No theorem's actual inequality is readable from those files. Any prior verification pass that claimed
to check a displayed result against "the full text" could not have done so, because the displayed
result is not in the file. Prose statements of rates survived, which is why every claim in this
chapter could still be checked, but that was luck rather than design.

The authors' LaTeX for four of these is now downloaded and was used for the two citation fixes above.
Landing them permanently under `papers/` is the durable fix.

---

## What this audit did not check

Attribution of the Athey and Wager rate, the Thomas and Brunskill MAGIC construction, and the Kallus
and Uehara claim that the non-Markov efficiency bound is exponential in the horizon. The causal-bandit
papers and ch10c were checked in the separate ch10c claim-source ledger and cold-run verification.

`fit_evaluation.md` is retained as an archived planning memo. Its status banner
now routes readers to the split ch10b and ch10c sources and their current
claim-source ledgers.

---

**Reconciled bullshit score: 20%.** Every result reproduces, the theorem proof is complete at the
scope it states, primary-source corrections are in place, and the oracle specification benchmark is
disclosed. No individual simulation remains above the shipping threshold.
