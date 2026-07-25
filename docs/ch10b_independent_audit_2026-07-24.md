# ch10b independent audit, 2026-07-24

**Commit audited:** `df4c74f` on branch `ch10b-reconciled` (17 ahead of `origin/main`, unpushed).
**This verdict does not apply to PR #38.** Its head `f1fac64` is four commits behind `df4c74f` and
does not contain the simulation corrections. Anyone reading #38 today is reading pre-correction
simulations.

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
| Theorem proof | **OPEN, one gap** | The second block, `J_M(π) = g-formula(π)`, is asserted in one sentence and never derived. |
| Attributed results, 6 checked | **PASS after 2 fixes** | Both fixes applied. See below. |
| Literature numbers, 40+ checked | **PASS** | Every figure from STAR\*D, HeartSteps, the ADHD SMART, Jaman and Project STAR matches its source. |
| Number provenance | **PASS after 1 fix** | 84 of 86 numeric literals traced to an owning artifact; the two exceptions resolved. |
| Reproducibility | **PASS** | All five scripts cold-recomputed. Every table fragment and every figure byte-identical. |
| Simulation gates, 7 per script | **ONE FAILS** | `dtr_policy_learning.py` fails gate 6, information leakage. Details below. |
| Build and exposition | **PASS after fixes** | 40 pages, zero undefined citations, zero overfull boxes. |

The chapter is in good shape on every axis but one. Nothing in the prose is a false claim, every
number reproduces, and the theorem survived an adversarial reading. The blocker is in a simulation:
the "correctly specified" arm of the policy-learning study is built from the true data-generating
constants, which the chapter does not disclose and which flatters the method that tops its results
table.

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

## The blocker: `dtr_policy_learning.py` fails the leakage gate

**Not fixed. This is a call for the author, not backend tidying, and the repo rule is that a single
"no" on any gate halts the ship until you have seen the verdict.**

`q1_basis` at `dtr_policy_learning.py:311-319` builds the stage-1 regression basis, in its
`correct=True` branch, out of the true data-generating constants:

```python
mean = RHO * x + TREAT_SHIFT * A1
u = (c2 - mean) / SIGMA_ETA
prob_below = norm.cdf(u)
truncated_first_moment = mean * prob_below - SIGMA_ETA * norm.pdf(u)
cols += [A1 * x, prob_below, truncated_first_moment]
```

`RHO = 0.6`, `TREAT_SHIFT = -0.6` and `SIGMA_ETA = 0.5` are the generating constants
(`dtr_policy_learning.py:64-66`). Line 129 draws the state as
`s21 = RHO * x + TREAT_SHIFT * A1 + SIGMA_ETA * noise`, and the oracle's own continuation value at
lines 163 to 165 is the same expression. So this is not a correctly specified functional form fitted
to data. It is the true continuation-value function, evaluated at the true parameters, handed to the
regression as a regressor. I read the code myself rather than taking the verdict on trust, and it
holds.

This basis feeds DM and AIPW in every cell of the headline sample-size sweep
(`dtr_policy_learning.py:452,456,461`), and the method it most flatters, plug-in `Q`, is the one that
tops `dtr_policy_learning_results.tex` at 0.9996 of oracle value. The chapter tells the reader the
methods "use the same cohorts, threshold search, and two-fold cross-fitting" and says nothing about a
basis wired to the DGP. A reader who takes "correct nuisances" to mean correctly specified and
estimated cannot reproduce that number.

**The fix is small.** Estimate the three constants per fold from the training rows, by regressing
`s21` on `(x, A1)` and taking the residual standard deviation, instead of reading them off
`DGP_PARAMS`. Then re-run `sweep_n` and `misspec` and re-emit the table. At 13 seconds a run the
recompute is free. My expectation is that the ranking survives, because those constants are cheap to
estimate at these sample sizes, but that is a guess and the point of the gate is not to guess.

Two smaller things worth doing in the same pass. The four misspecification regimes draw different
cohorts (`dtr_policy_learning.py:530-532`), so the ablation is unpaired on 40 seeds; pointing all four
at one seed stream costs nothing and makes the comparison paired. And the "wrong" outcome model does
not merely drop interactions, it removes all treatment-effect heterogeneity, so the learned stage-2
gain is a constant and the threshold search degenerates to treating everyone by construction. That
outcome is guaranteed before the simulation runs rather than discovered by it, and the chapter
describes the cell as only removing interactions.

`dynamic_dml_snmm.py` passes all seven gates. `ope_estimators.py` passes all seven, with a
longhand reimplementation of every estimator agreeing to machine precision and the misspecified cell
confirmed genuinely non-nesting. Two notes there: the standard-error calibration panel is computed at
horizon 8 rather than the horizon 16 used everywhere else, because trajectory importance sampling is
not yet Gaussian at 16 (coverage 0.83 against nominal), and the caption states the cell without
stating why. And one double-robustness gate tolerates a bias of a fifth of the estimand, so that leg
is verified only loosely.

---

## The one thing still open in the mathematics

The proof of Theorem `thm:dtr_rl_equivalence` is sound where it argues, and the statement survived a
dedicated attempt to break it. But the theorem asserts two blocks and the proof really only
establishes one.

The second block claims `J_M(π) = V(π) = g-formula(π) = E[Y^π]`. Of those four terms, `V(π) = E[Y^π]`
is a definition restated, `J_M(π)` is never defined in the chapter, and the K-stage g-formula is never
written down (only the two-stage case is displayed). The proof discharges the whole block with one
sentence: "Unrolling its transition and terminal reward kernels gives the observed-data g-formula."
That unrolling is a standard tower-property induction, but it is asserted rather than performed.

Two smaller points from the same review: the stage-K to stage-k identification induction is delegated
to a citation rather than carried out, and the maximum over continuation regimes sits inside an
almost-sure statement whose null set is per-regime, over an uncountable class. Both are repairable
under the hypotheses already stated, by fixing regular conditional probability kernels once and
arguing pointwise.

**This is left for the author.** Completing a proof is mathematical content, not backend tidying, and
the fix changes what the theorem is claiming to have shown.

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
and Uehara claim that the non-Markov efficiency bound is exponential in the horizon. The nine
causal-bandit papers that `fit_evaluation.md` recommends, which now belong to ch10c rather than here.
ch10c itself, which is materially weaker than ch10b and has no verification artifacts at all.

`fit_evaluation.md` is stale: it still describes causal bandits and adaptive experimentation as
sections 4 and 5 of ch10b, and its line references point at lines that no longer exist. It should be
refreshed or retired.

---

**Bullshit score: 35%.** Per-script: `ope_estimators` 20%, `dynamic_dml_snmm` 25%,
`dtr_policy_learning` 75%, `dtr_dags` capped at 25% as diagram-only, `dtr_qlearning_vs_murphy`
pending. The chapter's prose makes no false claim, every number reproduces byte-for-byte, and the
theorem survived a reviewer who was denied its proof. What pulls the score up is one simulation whose
correctly-specified arm knows the answer, and the fact that the chapter's own text does not say so.
Above 50 on any single script means that script must be fixed before the chapter ships, and
`dtr_policy_learning` is there.
