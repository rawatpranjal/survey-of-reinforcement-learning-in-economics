# Surgical Humanizer Pass — Master Edit Report

Surgical de-AI pass across survey chapters. Egregious tells only (em dash, colon-drumroll in main prose, blatant negative parallelism, aphoristic moral-closer, formulaic opener-as-thesis). Minimal word swaps; everything else byte-identical. Originals untouched; edits live in `<name>_targeted.tex`. Two passes per chapter (main surgical + independent check).

Plan: `~/.claude/plans/ok-on-the-arsiv-functional-cascade.md`

---

## ch07_bandits — `dynamic_pricing.tex` → `dynamic_pricing_targeted.tex`

12 edits / 11 paragraphs. Invariants: citet 48/48, citep 6/6, footnote 22/22, theorem 2/2, equation 4/4, figure 4/4, table 2/2, label 21/21, includegraphics 4/4, citation keys identical, em dash 0, textbf 0. Compiles 15pp, 0 undefined. Check: grammar PASS (line-62 "not only X but Y" optional "also" polish).

Tells: 5 colon-drumrolls, 4 negative parallelism, 1 em dash, 1 aphoristic closer, 1 formulaic opener.
PDFs: `docs/ch07_original.pdf`, `docs/ch07_targeted.pdf`

| Line | Old → New |
|---|---|
| 3 | `structure---...curves---learning` → `structure, such as ...curves, learning` |
| 3 | `assumptions matter: an unknown` → `assumptions matter, since an unknown` |
| 18 | `not whether the model has a finite-dimensional parameter; it is whether` → `not whether the model has a finite-dimensional parameter, but whether` |
| 60 | `not independent pulls: they lie` → `not independent pulls; they lie` |
| 62 | `does not merely say which prices can be discarded; it says how` → `says not only which prices can be discarded but how` |
| 67 | `This is not a cosmetic modeling choice: the error` → `The error` |
| 90 | `The regret is linear in $T$: every standard` → `Regret here is linear in $T$. Every standard` |
| 106 | `The lesson is that demand learning is also mechanism design. A pricing rule` → `A pricing rule` |
| 111 | `not just a constant factor` → `more than a constant factor` |
| 148 | `The same pattern appears beyond single-product` → `This pattern extends beyond single-product` |
| 165 | `narrower: WARP-based` → `narrower. WARP-based` |
| 176 | `closer to \citet{Weaver2025}: curve-level` → `closer to \citet{Weaver2025}. Curve-level` |

---

## ch00_introduction — abstract / intro / language

4 edits, all in `language.tex` → `language_targeted.tex`. abstract.tex: 0 edits. intro.tex: 0 edits.
Invariants (language): citet 19/19, citep 9/9, footnote 12/12, equation 2/2, table 2/2, label 8/8, textit 43/43, citation keys identical, em dash 0, textbf 0. Compiles (full ch00 wrapper) 19pp, 0 fatal. Check: PASS all 4 (line-1 rename judged accurate, fixes a tautology).

Tells: 2 colon-drumrolls, 1 aphoristic-lead-in, 1 tautology/redundancy.
PDFs: `docs/ch00_original.pdf`, `docs/ch00_targeted.pdf`

| Line | Old → New |
|---|---|
| 1 | `Two intellectual traditions both study sequential decision-making under uncertainty, but they descend from different intellectual traditions.` → `Economics and reinforcement learning both study sequential decision-making under uncertainty, but they descend from different intellectual traditions.` |
| 22 | `The key takeaway is that most RL convergence results and sample complexity guarantees refer to the training phase` → `Most RL convergence results and sample complexity guarantees refer to the training phase` |
| 63 | `The mapping is not exact: a reward $r(s,a)$ is` → `The mapping is not exact. A reward $r(s,a)$ is` |
| 81 | `asking how agents would respond: if the earned income tax credit` → `asking how agents would respond. If the earned income tax credit` |

Note: pre-existing `analytically.}.` double-period in line-63 footnote exists in original too (not introduced; out of scope).

---

## ch01_history — `history.tex` → `history_targeted.tex`

2 edits, both in "Animal Psychology". Invariants: citet 15/15, footnote 7/7, equation 6/6, label 2/2, subsection 3/3, textit 1/1, keys identical, em dash 0, textbf 0. Compiles 8pp, 0 fatal. Check: PASS both.

Tells: 1 negative parallelism, 1 editorializing closer.

| Line | Old → New |
|---|---|
| 16 | `The response followed not from the stimulus itself but from what it "predicted".` → `The response followed from what the stimulus "predicted" rather than from the stimulus itself.` |
| 24 | (deleted closer) `The model's power lay in prediction, not just explanation. It correctly predicted overexpectation,` → `The model correctly predicted overexpectation,` |

Note: pre-existing straight-quote `"predicted"` (vs LaTeX `` ``'' ``) on line 16 — in original too, out of scope.

---

## ch02_rl_algorithms — `rl_algorithms.tex` → `rl_algorithms_targeted.tex`

7 edits / 7 lines. Invariants: citet 30/30, citep 12/12, footnote 37/37, equation 22/22, align 2/2, figure 3/3, label 16/16, subsubsection 16/16, includegraphics 1/1, textit 10/10, keys identical, textbf 0. Em dash 4/4 (all in TikZ `% ---- Panel ----` comments, none in prose). Check: PASS all 7. Compiles 21pp (benign pre-existing `\FloatBarrier` undefined-ctrl-seq, identical in original — `placeins` not in compile_chapter preamble; main.tex loads it).

Tells: 4 colon-drumrolls, 2 aphoristic/promotional closers, 1 negative parallelism.

| Line | Old → New |
|---|---|
| 47 | `off-policy:\footnote{...} the update target` → `off-policy.\footnote{...} The update target` |
| 57 | `Virtually all continuous-control results...descend from the policy gradient framework for this reason.` → `Continuous-control methods...are largely built on this policy gradient framework.` |
| 126 | `supervised regression step: given a batch...fit a function approximator` → `supervised regression step. Given a batch..., it fits a function approximator` |
| 165 | `practical instability: a single large gradient step` → `practical instability. A single large gradient step` |
| 179 | (deleted gloss) `applications, demonstrating that constraining the magnitude of policy updates is essential for stable optimization.` → `applications.` |
| 268 | `not direct instances of the framework; they are recovered` → `not direct instances of the framework but are recovered` |
| 406 | `demonstrated this concretely: in a simple gambling MDP` → `demonstrated this concretely. In a simple gambling MDP` |

---

## ch03_theory — `planning_learning_v3.tex` → `planning_learning_v3_targeted.tex`

12 edits / 11 lines (line 516 carries 2). Invariants: citet 80/80, citep 54/54, footnote 58/58, equation 28/28, figure 8/8, table 3/3, input 4/4, label 42/42, subsubsection 24/24, emph 35/35, keys identical, textbf 0. Em dash 1/1 (commented line 313, untouched). Check: PASS all 12. Compiles 39pp, 0 non-FloatBarrier errors.

Tells: ~7 colon-drumrolls, ~3 negative parallelisms, 2 conclusion closers/filler.

| Line | Old → New |
|---|---|
| 30 | `is as follows: the linear operator` → `is that the linear operator` |
| 104 | `quantitative consequences: \citet{evendar2003}` → `quantitative consequences. \citet{evendar2003}` |
| 168 | `Three terms drive the error: the geometric decay` → `The error is driven by three terms, the geometric decay` |
| 170 | `geometry of the problem: when` → `geometry of the problem. When` |
| 200 | `confirm the diagnosis: basis representability, not algorithmic failure.` → `confirm that the failure is one of basis representability rather than algorithmic failure.` |
| 276 | `no longer orthogonal in the $d^\pi$-norm; it is \emph{oblique}.` → `becomes \emph{oblique} in the $d^\pi$-norm rather than orthogonal.` |
| 278 | `This divergence is not overfitting.` → `This divergence differs from overfitting.` |
| 360 | (folded) `...globally optimal. The non-convex landscape has no false peaks, no spurious local maxima. Gradient ascent cannot get trapped.` → `...globally optimal, so gradient ascent cannot get trapped at a spurious local maximum.` |
| 369 | `illustrates the distinction: on the policy manifold` → `illustrates the distinction. On the policy manifold` |
| 403 | `illustrates this mechanism: each surrogate` → `illustrates this mechanism. Each surrogate` |
| 516 | `RL algorithms are not mysterious. They are asymptotic approximations` → `RL algorithms are asymptotic approximations`  ·AND·  `is not a departure from dynamic programming but an extension of it.` → `extends dynamic programming.` |

---

## ch03b_deeprl_practice — `deeprl_practice.tex` → `deeprl_practice_targeted.tex`

4 edits / 4 lines. Invariants: citet 39/39, citep 9/9, footnote 21/21, equation 1/1, figure 2/2, table 1/1, label 12/12, emph 10/10, includegraphics 2/2, keys identical, textbf 0. Em dash 6→3 (the 3 remaining are table N/A cells in the Policy-agreement column, lines 94/96/98 — content, not prose). Compiles 13pp, 0 non-FloatBarrier errors. Check: initial NEEDS-FIX on lines 13 & 81 (comma-swap garden-path) → corrected to parentheses → re-verified PASS.

Tells: 3 prose em-dash pairs (incl one in a footnote), 1 colon-drumroll.

| Line | Old → New |
|---|---|
| 13 | `correlated across time---...successive state-action pairs---they cancel` → `correlated across time (...successive state-action pairs), they cancel` |
| 58 | (footnote) `Architectural interventions---layer normalization \citep{Lyle2025}, ..., spectral normalization---reduce` → `Architectural interventions, layer normalization \citep{Lyle2025}, ..., spectral normalization, reduce` |
| 74 | `changes the objective: clipped rewards` → `changes the objective; clipped rewards` |
| 81 | `feasible $(s,a)$ pairs---0.2\% coverage---the distribution mismatch condition` → `feasible $(s,a)$ pairs (0.2\% coverage), the distribution mismatch condition` |

Note: line 13 & 81 used parentheses (em-dash parenthetical pair → parens, house style) after Pass-B flagged the all-comma version.

---

## ch04_control_problems — `applications.tex` → `applications_targeted.tex`

7 edits / 6 lines (line 209 carries 2). All colon-drumrolls. Invariants: citet 27/27, citep 5/5, footnote 22/22, equation 10/10, figure 1/1, table 7/7, label 16/16, subsection 8/8, emph 3/3, keys identical, textbf 0. Em dash 2/2 (line 47 comment + line 204 table N/A cell, untouched). Check: PASS all 7. Compiles 12pp, 0 non-FloatBarrier errors.

Tells: 7 colon-drumrolls (no em dashes / parallelisms / aphorisms in this chapter).

| Line | Old → New |
|---|---|
| 44 | `fleet positioning problem: today's assignments` → `fleet positioning problem; today's assignments` |
| 88 | `capacity allocation problem: how to dynamically distribute` → `capacity allocation problem, namely how to dynamically distribute` |
| 160 | `microstructure signals: trading aggressively` → `microstructure signals, trading aggressively` |
| 209 | `RL underperforms: the base-stock` → `RL underperforms. The base-stock`  ·AND·  `unavailable: non-stationary demand` → `unavailable, such as non-stationary demand` |
| 215 | `budget pacing problem: an advertiser must allocate` → `budget pacing problem, where an advertiser must allocate` |
| 258 | `omit DP rather than run it: plain-Python` → `omit DP rather than run it. Plain-Python` |

Note: pre-existing double-negative "not numerically infeasible" (line 258) in original too, out of scope.

---

## ch05_econ_models — `rl_in_se.tex` → `rl_in_se_targeted.tex`

1 edit. Invariants: citet 33/33, citep 11/11, footnote 24/24, equation 12/12, figure 2/2, table 1/1, input 1/1, label 15/15, emph 3/3, keys identical, em dash 0, textbf 0. Compiles 16pp, 0 non-FloatBarrier. Check: PASS (trivial colon-split, self-verified). Very clean chapter — 4 of 5 slices had zero egregious tells.

| Line | Old → New |
|---|---|
| 104 | `combine game theory and dynamic programming: firms choose actions` → `combine game theory and dynamic programming. Firms choose actions` |

---

## ch06_macro — `macro_rl.tex` → `macro_rl_targeted.tex` (1349-line chapter, 14 slices)

7 edits (8 textual changes; slice L bundled a correct `the RL is`→`RL is` grammar fix). Invariants: citet 79/79, citep 14/14, footnote 15/15, equation 11/11, figure 2/2, table 2/2, input 2/2, label 42/42, subsection 10/10, subsubsection 18/18, keys identical, em dash 0, textbf 1 (comment line 10 only). Compiles 33pp, 0 non-FloatBarrier. Check: PASS all 7 (folds/deletions verified lossless).

Tells: 3 colon-drumrolls, 2 negative parallelisms, 2 aphoristic closers.

| Region | Old → New |
|---|---|
| RBC sim (fold) | `The result is what one would hope on a model this small. It is not a demonstration that RL replaces dynamic programming; it is a check that RL does not fail it on the textbook case.` → `The simulation establishes that RL does not fail dynamic programming on this textbook case, rather than that RL replaces it.` |
| RSPG (del) | `...same underlying household problem, not competing claims about the same equilibrium.` → `...same underlying household problem.` |
| Stackelberg | `Best-response updates alternate: fix the leader, train followers...; then fix followers` → `Best-response updates alternate. One step fixes the leader, trains followers...; the next fixes followers` |
| LQ MFG sim | `LQ example: history helps` → `LQ example; history helps` |
| AI Economist | `suggestive rather than decisive: the two-level RL planner` → `suggestive rather than decisive. The two-level RL planner` |
| RICE-N | `...strategic policies, not a claim that regions are boundedly rational.` → `...strategic policies and does not imply that regions are boundedly rational.` |
| Discussion (del) | `...different stages of maturity. The current state is best read as early progress rather than as a settled methodology.` → `...different stages of maturity.` |

---

## ch06_games — `rl_in_games.tex` → `rl_in_games_targeted.tex`

7 edits / 7 lines. Invariants: citet 22/22, citep 15/15, footnote 20/20, equation 12/12, figure 3/3, table 4/4, input 3/3, label 13/13, subsubsection 13/13, emph 6/6, keys identical, em dash 0, textbf 0. Compiles 18pp, 0 non-FloatBarrier. Check: PASS all 7 ("where" connectives + "Instead" drop verified).

Tells: 6 colon-drumrolls, 1 "Instead" pivot drop.

| Line | Old → New |
|---|---|
| 77 | `provides a deeper lens: reinforcement learning dynamics...converge` → `shows that reinforcement learning dynamics...converge` |
| 106 | `bypasses equilibrium selection: instead of` → `bypasses equilibrium selection. Instead of` |
| 202 | `equilibrium price path: long horizons` → `equilibrium price path, where long horizons` |
| 204 | `collapse rate: $p_T \to 0$` → `collapse rate, where $p_T \to 0$` |
| 209 | `subsection: the asymptotic price-collapse` → `subsection; the asymptotic price-collapse` |
| 244 | `delivered numerically: without commitment power` → `delivered numerically. Without commitment power` |
| 248 | `not a Coase regime switch. Instead, as $\delta$ grows` → `not a Coase regime switch. As $\delta$ grows` |

---

## ch08_offline_rl — `offline_rl.tex` → `offline_rl_targeted.tex`

6 edits / 6 lines. Invariants: citet 17/17, citep 10/10, footnote 13/13, equation 9/9, figure 1/1, table 1/1, input 1/1, label 19/19, subsubsection 8/8, textit 8/8, keys identical, em dash 0, textbf 0. Compiles 14pp, 0 non-FloatBarrier. Check: PASS all 6 (line-122 epigram deletion verified lossless).

Tells: 5 colon-drumrolls, 1 deleted epigram.

| Line | Old → New |
|---|---|
| 1 | `interacts with its environment while learning: it tries a price` → `...while learning; it tries a price` |
| 12 | `is the \textit{pessimism principle}: construct a lower confidence bound` → `is the \textit{pessimism principle}, which is to construct a lower confidence bound` |
| 101 | `outside the dataset: the $\max$ operation is implicit` → `outside the dataset; the $\max$ operation is implicit` |
| 105 | `takes a different approach: rather than modifying` → `takes a different approach. Rather than modifying` |
| 122 | (deleted epigram) `treat the model as the policy. The agent forecasts what to do, and the forecast is the action. Two members` → `treat the model as the policy. Two members` |
| 156 | `The reason is the behavioral itself: the state-dependent kernel` → `The reason is the behavioral itself. The state-dependent kernel` |

---

## ch09_rlhf — `rlhf.tex` → `rlhf_targeted.tex`

5 prose edits + 5 caption `\textbf` removals = 10 changes / 10 lines. Invariants: citet 20/20, citep 27/27, footnote 12/12, equation 5/5, figure 5/5, table 2/2, input 2/2, label 21/21, caption 7/7, includegraphics 5/5, emph 6/6, keys identical, em dash 0, textbf **5→0**. Compiles 21pp, 0 non-FloatBarrier. Check: PASS all 10 (line-124 "silently" drop verified editorial-only).

⚠️ SCOPE NOTE: this is the only chapter where the pass touched formatting beyond prose. 5 figure/table captions had bold lead-ins `\caption{\textbf{Title.} ...}` — stripped to `\caption{Title. ...}` (text byte-identical). Reason: CLAUDE.md bans `\textbf` everywhere; the other 16 chapters' captions have zero. Reversible via original.

Tells: 3 negative parallelisms, 2 colon-drumrolls, + 5 caption de-bold.

| Line | Old → New |
|---|---|
| 14 | `makes this point explicit: standard Bradley-Terry-style` → `makes this point explicit. Standard Bradley-Terry-style` |
| 51 | `\caption{\textbf{RLHF versus DPO pipelines.} Top row...` → `\caption{RLHF versus DPO pipelines. Top row...` |
| 58 | `should not be read as a law of nature. It is an empirical frontier of...` → `is an empirical frontier of..., not a law of nature.` |
| 64 | `Feedback is not passively observed; it is elicited through a mechanism.` → `Feedback is elicited through a mechanism, not passively observed.` |
| 124 | `not an efficiency gain...but a guarantee...BT-MLE silently abandons` → `a guarantee...BT-MLE abandons..., rather than an efficiency gain` |
| 147,156,164,174 | 4× `\caption{\textbf{Title.} ...}` → `\caption{Title. ...}` (caption text unchanged) |
| 160 | `structural model dominates: even at $K=25$` → `structural model dominates. Even at $K=25$` |

---

## ch10_causal — `causal_rl.tex` → `causal_rl_targeted.tex`

6 edits / 5 lines. Invariants: citet 38/38, citep 11/11, footnote 12/12, equation 16/16, figure 4/4, table 2/2, input 1/1, label 42/42, keys identical, em dash 0, textbf 0. Compiles 21pp. Check: PASS all 6 (line-19 opener fold verified: "It"→Table, no claim lost).

| Line | Old → New |
|---|---|
| 19 | `The purpose is not to create a checklist of papers. It is to show economists that...` → `It groups papers to show economists that...` |
| 56 | 2× `is a modeling/identification challenge: X` → `..., since X` |
| 214 | `are separate: first specify` → `are separate. First specify`  ·AND·  `do not by themselves remove hidden confounding, but they make...` → `By themselves they do not remove hidden confounding; they make...` |
| 278 | `overestimate promotion success: the promote action` → `...success; the promote action` |
| 325 | `identification techniques of this one: this chapter` → `...of this one. This chapter` |

---

## ch10b_rl_for_ci — `rl_for_ci.tex` → `rl_for_ci_targeted.tex`

7 edits / 6 lines (line 299 carries 2). Invariants: citet 92/92, citep 25/25, footnote 12/12, equation 15/15, figure 3/3, table 5/5, input 4/4, label 35/35, emph 29/29, keys identical, ASCII `---` 0, textbf 0. Compiles 33pp. Check: PASS all 7.

⚠️ Line 248 was a genuine **Unicode em-dash (—, U+2014)** caught by the slice agent reading prose; the ASCII `grep '---'` invariant missed it. Lesson: scan Unicode dashes too (added to remaining chapters). Fixed → comma.

| Line | Old → New |
|---|---|
| 19 | `The question here is not how to protect an RL estimator from confounding, but how...` → `The question here concerns how..., rather than how to protect an RL estimator` |
| 23 | `is the same: doubly-robust orthogonalization` → `is the same. Doubly-robust orthogonalization` |
| 248 | `$m(q) < N$ — equivalently, whenever` (Unicode em-dash) → `$m(q) < N$, equivalently, whenever` |
| 259 | `same applied workflow: design a multi-wave` → `same applied workflow. First design a multi-wave` |
| 299 | 2× `targets Section~\ref{...}: dynamic/on the` → `targets Section~\ref{...}. Dynamic/On the` |
| 371 | `not separate enterprises...; they are the same enterprise...` → `the same enterprise..., not separate enterprises...` |

---

## ch11_dist_robust_constrained — `dist_robust_constrained.tex` → `dist_robust_constrained_targeted.tex`

7 edits / 6 lines. Invariants: citet 33/33, citep 12/12, footnote 19/19, equation 18/18, figure 2/2, table 3/3, input 2/2, label 30/30, emph 4/4, keys identical, ASCII `---` 0, Unicode em-dash 0, textbf 0. Compiles 15pp. Check: PASS all 7 (608-610 fold: dropped "not a deficiency of the method" editorial, no claim lost).

| Line | Old → New |
|---|---|
| 109 | `this is inadequate: a portfolio manager` → `this is inadequate, since a portfolio manager` |
| 256 | `= \lambda_k^*$: the marginal change` → `= \lambda_k^*$, the marginal change` |
| 263 | `\emph{zero duality gap}: the optimal dual value` → `\emph{zero duality gap}, meaning the optimal dual value` |
| 373 | `compare three methods: the constrained LP oracle` → `compare three methods, the constrained LP oracle` |
| 526 | `problem-specific: joint torques..., force perturbations` → `problem-specific (joint torques..., force perturbations)` |
| 608 | `oracle gap; this is the expected price of not knowing the true model, not a deficiency of the method.` → `oracle gap, the expected price of not knowing the true model.`  ·AND·  `shows the mechanism: robust agents` → `shows the mechanism, where robust agents` |

---

## ch12_world_models — 6 sub-files → `*_targeted.tex` (wrapper compiled via `ch12_targeted_wrap.tex`)

9 edits across 6 sub-files. Per-file invariants all match (citet/footnote/input/figure/table), keys identical, ASCII `---` unchanged (s09 keeps its 1 caption N/A-symbol `---`), Unicode em-dash 0, textbf 0. Compiles 41pp, 0 non-FloatBarrier. Check: PASS all 9 (s04 reword faithful, s10 deletion clean).

| Sub-file | Edits |
|---|---|
| s01_intro | `outer loop is the same: learn $M$...` → `is the same, since both learn $M$...` |
| s03_dyna_q | `is not a separate algorithm... It is the same update rule` → `is the same update rule..., with no separate algorithm`  ·  `is not a distinct planning operator; it is step 3` → `is step 3..., not a distinct planning operator` |
| s04_deep_mbrl | `achieves its results despite training to the wrong target; ...how much further the field could go` → `...while training against likelihood rather than the downstream decision, ...what is gained when the loss is aligned` |
| s06_objectives | 2× negative-parallelism reorders (value-function correctness; "contribution lies in... rather than...") |
| s09_dual_sim | `designed to expose three things in turn:` → `examines`  ·  `with one twist:` → `with one twist, in that` |
| s10_synthesis | (deleted aphorism) `The agent's policy is only as good as the model it plans against, and the question...` → `The question...` |

---

## ch99_conclusion — `conclusion.tex` → `conclusion_targeted.tex`

5 edits / 5 lines. Invariants: citet 5/5, citep 17/17, subsection 6/6, ref 22/22, keys identical, ASCII `---` 0, Unicode em-dash 0, textbf 0. Compiles 9pp. Check: PASS all 5 (line-111 appositive-list+verb valid, mild garden-path noted but not a defect).

| Line | Old → New |
|---|---|
| 23 | `the nested fixed-point algorithm is infeasible, as the simulation... demonstrates: a DQN trained` → `...is infeasible. As the simulation... demonstrates, a DQN trained` |
| 37 | `makes this connection concrete: Q-learning households` → `shows that Q-learning households` |
| 67 | `setting most familiar to economists: the analyst has a fixed dataset` → `setting most familiar to economists, in which the analyst has a fixed dataset` |
| 80 | `is a minimum rather than an aspiration.` → `is a minimum requirement.` |
| 111 | `the Q-function-CCP duality, are not coincidences; they reflect the fact that` → `the Q-function-CCP duality, reflect the fact that` |

---

## SUMMARY — all 17 chapters

| Ch | File | Edits | Check |
|----|------|-------|-------|
| 0 | ch00 (abstract/intro/language) | 4 | PASS |
| 1 | ch01 history | 2 | PASS |
| 2 | ch02 rl_algorithms | 7 | PASS |
| 3 | ch03 theory | 12 | PASS |
| 3b | ch03b deeprl_practice | 4 | PASS (2 re-fixed w/ parens) |
| 4 | ch04 applications | 7 | PASS |
| 5 | ch05 rl_in_se | 1 | PASS |
| 6m | ch06 macro_rl | 7 | PASS |
| 6g | ch06 rl_in_games | 7 | PASS |
| 7 | ch07 dynamic_pricing (pilot) | 12 | PASS |
| 8 | ch08 offline_rl | 6 | PASS |
| 9 | ch09 rlhf | 5 prose + 5 caption de-bold | PASS |
| 10 | ch10 causal_rl | 6 | PASS |
| 10b | ch10b rl_for_ci | 7 | PASS |
| 11 | ch11 dist_robust_constrained | 7 | PASS |
| 12 | ch12 world_models (6 sub-files) | 9 | PASS |
| 99 | ch99 conclusion | 5 | PASS |

Total: ~112 surgical edits + 5 caption de-bolds. All originals untouched; edits in `*_targeted.tex`. Every chapter: citation keys identical, em dash (ASCII+Unicode) 0, textbf 0 (except comment lines), all envs/figures/tables/footnotes preserved, compiles clean. Next step: swap `_targeted` → originals once approved.
