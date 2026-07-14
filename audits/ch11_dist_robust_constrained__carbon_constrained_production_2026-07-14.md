# Audit: ch11_dist_robust_constrained/sims/carbon_constrained_production.py

**Date:** 2026-07-14
**Type:** DELTA (prior full audit 2026-05-19 @ 25%; polish 2026-05-20 @ 10%; changed 2026-05-22 in commit `9238bab`, never re-audited)
**Auditor stance:** hostile journal referee, evidence-only, read-only.

## Delta summary

Commit `9238bab` (2026-05-22, "polish: ch11 carbon 5->10 seeds") changed exactly:
- `carbon_constrained_production.py:61` — `SEEDS = [0,1,2,3,4]` -> `[0,1,2,3,4,5,6,7,8,9]` (5 -> 10 seeds). One line.
- `carbon_constrained_production_table.tex` — both QL rows renumbered to the 10-seed values.
- `dist_robust_constrained.tex` — results paragraph numbers, one "five seeds" -> "ten seeds", table caption and figure caption "five" -> "ten seeds".
- Regenerated `_stdout.txt` (now 10 seeds) and `_convergence.png` (legend "10 seeds").

The change is the exact upgrade the 2026-05-20 polish flagged as its only residual (5 seeds short of the CLAUDE.md minimum of 10). It closes that gap. It also leaves one stale "five seeds" behind (Finding 1).

## Files read (this turn, end to end)

- `/Users/pranjal/Code/rl/ch11_dist_robust_constrained/sims/carbon_constrained_production.py`
- `/Users/pranjal/Code/rl/ch11_dist_robust_constrained/sims/carbon_constrained_production_stdout.txt`
- `/Users/pranjal/Code/rl/ch11_dist_robust_constrained/sims/carbon_constrained_production_table.tex`
- `/Users/pranjal/Code/rl/ch11_dist_robust_constrained/sims/carbon_constrained_production_convergence.png` (viewed)
- `/Users/pranjal/Code/rl/ch11_dist_robust_constrained/tex/dist_robust_constrained.tex` (lines 238–437, the consuming subsection)
- `/Users/pranjal/Code/rl/sims/sim_cache.py`
- `git show 9238bab` (full diff), `git show 9238bab^:...tex`
- Prior audits `..._2026-05-19.md` and `..._polish_2026-05-20.md` (read at step 6 only)

## Step 3 — thesis statement (what this sim is evidence FOR)

(i) Theoretical claim of the surrounding subsection: a constrained MDP admits a Lagrangian/LP-dual formulation (Eq. `cmdp_lagrangian`, `carbon_cmdp`) in which the optimal dual variable is the *shadow price* of the constraint (envelope theorem, `partial V*/partial q_k = lambda_k*`), and the CMDP can be solved by primal-dual dual ascent on the multiplier (Eq. `dual_update`), with zero duality gap (Paternain 2019). It is the applied capstone of the chapter's Section on constrained RL.

(ii) What the sim is used as evidence for: that naive Lagrangian dual ascent, run on a concrete economic CMDP (a factory choosing production level and dirty/clean energy under a carbon budget), (a) drives an unconstrained-optimal-but-budget-violating policy back into feasibility, (b) converges the learned multiplier `lambda` toward the analytical LP shadow price `lambda* = 1.20`, and (c) does so with little overshoot, so the PID damping of Stooke (2020) is not needed here. The sim is the demonstration that the shadow-price/dual-ascent theory operates as advertised on a real problem.

## Criteria verdicts

### (a) CORRECTNESS — PASS (one disclosed caveat)

- Primal is tabular Q-learning on the Lagrangian one-step reward `r - lambda*c` (line 265–266); dual is projected gradient ascent `lambda <- [lambda + eta*(avg_cost_inf - d)]_+` every 1000 episodes (line 276–281). This is the single-Q-table two-timescale Lagrangian primal-dual named in the prose (Eq. `dual_update`), matching Tessler (2019) RCPO in form. It implements what it claims.
- LP oracle is the textbook Altman occupation-measure LP (lines 182–224); the shadow price is read from `result.ineqlin.marginals[0]` (line 217). Correct.
- No information leakage: model-free QL consumes only sampled `(s,r,c)`; `build_matrices` / VI / LP are used for the oracle and budget only, never inside `run_q_learning`.
- Caveat (not a bug, and disclosed in prose): the learned `lambda` settles at 1.396 (~16% above `lambda* = 1.20`) while the greedy policy's cost is 28.4, strictly below the budget 31.35. A positive multiplier with a slack constraint is a soft violation of complementary slackness, i.e. the primal-dual has not reached the exact saddle. The mechanism is that the dual update measures cost on the epsilon-greedy *behavior* policy (`cost_buffer`, line 272, exploration floor 0.05), which emits more than the greedy policy, so `lambda` is calibrated slightly high and over-constrains the greedy policy. Prose calls the agent "slightly too conservative" (tex 403), which is honest. Not a correctness failure.

### (b) PRESENTATION / NUMBERS — FAIL (one internal contradiction; one minor overstatement)

Number-consistency sweep, stdout <-> table <-> figure <-> prose, all traced:

| Quantity | stdout | table.tex | prose (tex) | figure |
|---|---|---|---|---|
| LP return / cost / lambda* | 186.4 / 31.35 / 1.20 | 186.4 / 31.35 / 1.20 | 186 / 31 / 1.20 | dashed 186 / dashed 1.20 |
| Unconstr. QL return / cost | 257.81±3.21 / 99.08±2.62 | 257.8±3.2 / 99.08±2.62 | 258±3 / 99±3 | orange ~255 |
| Lagr. QL return / cost | 178.43±4.04 / 28.44±3.60 | 178.4±4.0 / 28.44±3.60 | 178±4 / 28±4 | green ~178 |
| Lagr. lambda (final) | 1.3963±0.0019 | 1.40±0.00 | ~1.40 | green plateau ~1.40 |
| lambda peak | 1.4052±0.0019 (max 1.4139) | — | 1.405±0.002 | slight peak ~1.40 |

Table and figure are fully consistent with the 10-seed stdout. Two prose defects:

1. **Internal contradiction (see Finding 1).** tex:377 still says the QL variants are "run over five seeds," while the table caption (413), figure caption (425), and the results sentence (395, "Across the ten seeds") all say ten. The delta updated three of the four "five seeds" and missed this one.

2. **"seventeen percent" overstates.** tex:400 — the final `lambda` "sits about seventeen percent above `lambda*`." The true mean 1.3963 gives (1.3963-1.20)/1.20 = 16.4%, which rounds to sixteen; only the rounded display value 1.40 gives 16.67% -> seventeen. The pre-delta text (5-seed, 1.39) correctly said "sixteen." Low-severity overstatement.

The `lambda`-trajectory claim the polish rewrote (peak 1.405±0.002, within one percent of the settling value) is verified against the current 10-seed stdout: peak mean 1.4052, settle 1.3963, gap 0.64% < 1%. Correct. The discredited "overshoots to 3.2" is gone.

### (c) CHAPTER FIT — PASS

The sim demonstrates precisely the step-3 claim. Unconstrained QL reaches 258 (near DP 273) but emits 99 (3x the budget of 31); Lagrangian dual ascent drives cost to 28 (feasible) and the multiplier to a positive value tracking the LP shadow price; the near-monotone `lambda` path substantiates the "little overshoot, PID not needed here" point tied to Stooke (2020). Direct evidence for shadow-price/dual-ascent theory on an economic CMDP.

### (d) EFFICIENCY / STANDARDS — PASS

- Caching: `load_results`/`save_results` keyed on `CONFIG` (includes `seeds` and `version:13`). `sim_cache.config_hash` is MD5 over the JSON config, so the 5->10 change altered the key and invalidated the old 5-seed cache. stdout line 120 prints "Cache saved" (a full compute, not "Loaded from cache"), so the published artifacts genuinely come from a 10-seed run. Output mtimes (15:39) are after the .py edit (15:10), same day — regenerated, not stale.
- Flags: `add_cache_args` gives `--data-only` / `--plots-only` (lines 543–552). Compliant.
- Seeds: 10 (>= 10 minimum), mean±SE reported in stdout, table, and figure bands. Compliant.
- stdout format: parameter header, per-seed lines, summary block, output paths, no opinion words. Compliant.
- Colors via `plot_style` (`COLORS`, `BENCH_STYLE`, `FIG_DOUBLE`); no hardcoded hex. Compliant.

## 7-point checklist

1. **Algorithm identity** — PASS. Lagrangian QL on `r-lambda*c` + projected dual ascent (Tessler 2019 form); LP oracle = Altman occupation-measure LP with dual from HiGHS marginals.
2. **Environment/MDP fidelity** — PASS. 18 states (inv 0–8 x 2 regimes), 8 actions (prod 0–3 x dirty/clean); price 10, prod cost {2,5}, hold 1, emission {3.0,0.5}, gamma 0.95 (py lines 28–41) all match tex 344–370.
3. **Data integrity** — PASS. Numbers computed (not hardcoded); cache correctly invalidated by the seed change; stdout shows a fresh 10-seed compute feeding the table and figure.
4. **Comparison fairness** — PASS. Same env, same 10 seeds, same H=100, shared `_eval_det` with fixed eval RNGs (42 periodic, 99 final, 5000 episodes) for both QL variants. LP-vs-QL truncation asymmetry (exact infinite-horizon vs H=100 MC) is disclosed in the footnote at tex 378–388.
5. **Theoretical sanity** — PASS (with the disclosed caveat). Constraint binds at oracle (cost 31.35 = d); `lambda* = 1.20 > 0`; unconstrained policy violates 3x; Lagrangian QL approaches LP optimum and respects budget. The `lambda` settling ~16% high with a slack greedy constraint is the epsilon-greedy dual-estimation artifact noted in (a); prose reports the conservatism plainly.
6. **Information leakage** — PASS. Model-free QL sees only samples; oracle/model used only for comparison and the budget.
7. **Seed & reproducibility** — PASS. Seeds fixed [0..9], 10 seeds, mean±SE everywhere. This is the item the delta fixed.

## Prior-audit open-item disposition

From **2026-05-19** (25%):
1. Single-seed reporting (the bullshit driver) — **RESOLVED.** Now 10 seeds with mean±SE in stdout, table, and figure bands.
2. "lambda overshoots to 3.2" prose claim, figure-only / stdout-uncorroborated — **RESOLVED.** Removed in the 05-20 polish; current prose (tex 393–399) says "rises from zero and settles at lambda ~ 1.40 ... nearly monotone: peak 1.405±0.002, within one percent of the settling value," and that peak is corroborated by current stdout line 119.
3. LP-vs-QL truncation-bias asymmetry undisclosed — **RESOLVED.** Footnote at tex 378–388 discloses it.

From **2026-05-20** polish (10%):
- Residual "5 seeds, not the stated 10" — **RESOLVED** by this delta (`9238bab`): now 10 seeds, and every number carries an SE.

New item introduced by the delta:
- Stale "five seeds" at tex:377 — **REGRESSED/NEW** (see Finding 1). The delta upgraded the counts everywhere except this one sentence.

## Findings (severity-ordered)

**Finding 1 (medium — internal contradiction, delta-introduced).**
`dist_robust_constrained.tex:377` still reads "Each Q-learning variant is run over five seeds; the table reports means and standard errors," on the same page as the table caption (413) "over ten seeds," the figure caption (425) "means over ten seeds," and the body sentence (395) "Across the ten seeds." Commit `9238bab` changed the other three occurrences of "five seeds" to "ten" and missed this one (confirmed: `git show 9238bab^` had "five seeds" at 377, 395, 413, 425; the current file has "ten" at 395/413/425 but "five" survives at 377). A referee reading top-to-bottom hits "five seeds" then "ten seeds" eighteen lines later. Fix: `five` -> `ten` at line 377. (Aside: line 377 also uses a semicolon splice, against the project no-semicolon-in-prose rule; both are cured by the same rewrite. Pre-existing, not delta-caused.)

**Finding 2 (low — rounding overstatement).**
`dist_robust_constrained.tex:400` says the final `lambda` sits "about seventeen percent" above `lambda*`. The true 10-seed mean 1.3963 is 16.4% above 1.20 (rounds to sixteen); "seventeen" holds only off the rounded display value 1.40 (16.67%). The pre-delta prose said "sixteen." Minor; "about sixteen percent" is the faithful figure.

**Finding 3 (informational — not a defect, disclosed).**
The learned `lambda` (1.40) exceeds `lambda*` (1.20) by ~16% and the Lagrangian policy holds cost at 28.4 < budget 31.35, a soft complementary-slackness tension from measuring the dual constraint signal on the epsilon-greedy behavior policy rather than the greedy policy. The prose already frames this as the agent being "slightly too conservative," so no action is required; noted so a future reader does not mistake it for a convergence bug.

No finding rises to the 50% halt threshold. The substance is intact and improved over the prior audits; the only live defect is a one-word stale seed count.

**Bullshit score: 25%** — Reviewer 2 circles the "five seeds" at tex:377 sitting on the same page as three "ten seeds" and the 10-seed table, and quibbles the "seventeen percent" rounding; the algorithm, environment, numbers, and headline finding all survive revision, so it is a snarky-comment defect, not a substance defect. Rounded up from the polish's 10% because a self-contradicting seed count in the published prose is exactly the number-consistency slip a hostile referee pounces on.
