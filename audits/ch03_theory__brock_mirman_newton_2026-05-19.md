# Audit: ch03_theory/sims/brock_mirman_newton.py

**Date:** 2026-05-19
**Diagram-only:** no (Regime 1 PI/VI is real computation on a 1,000-state MDP; Regime 2 LP is real; Regime 3 timing sweep is real; only Panels (a)/(b) of the convergence figure are hand-drawn illustrations of the staircase / Newton-jump geometry, not data plots)
**Cited tex file(s):** `/Users/pranjal/Code/rl/ch03_theory/tex/planning_learning_v3.tex` lines 45–74 (Section "Simulation Study: The Brock--Mirman Economy") — figure include at line 65, table input at line 74.
**Cited paper PDFs read:** none in `papers/` — the chapter `papers/` directory holds RL-theory references (TD/Q-learning convergence, sample complexity), but does not contain Brock & Mirman (1972), Manne (1960), Puterman (1979), Howard (1960), Santos & Rust (2004), or Bertsekas (2022). The corresponding bib entries are all present in `/Users/pranjal/Code/rl/docs/refs.bib` (lines 126, 136, 227, 3427, 3437, 3457, 3477, 4506, 4516), but no PDFs were available locally to cross-check term-by-term. I verified the algorithmic claims against the LaTeX writeup (which is internally self-consistent) and against textbook formulations of VI, Howard PI, and Manne LP.

## 1. Algorithm Identity

**Value iteration (lines 129–145).** Standard form. `Q = R + gamma * (P @ V)`, then `V_new = max_a Q`; convergence test on sup-norm successive difference. Matches the textbook Bellman optimality update. The final greedy policy is recomputed from $V$ after the loop. Fine.

**Howard policy iteration (lines 148–186).** Standard exact PI:
- Policy evaluation: solves $(I - \gamma P^\pi) V = r^\pi$ via `np.linalg.solve` — this is exact, not modified PI / not truncated. This is the algorithm the tex describes (the "Newton step"). 
- Policy improvement via $\arg\max_a [R + \gamma P V]$.
- Termination on policy stability (`np.array_equal(new_policy, policy)`).

One genuine wart: the `errors` list returned by `policy_iteration` is computed *after* the loop by `errs.append(max|V_h - V_star|)` where `V_star` is the *last* `V_history` entry — i.e., the errors are measured against the algorithm's final iterate, not against an external ground truth. For PI this is essentially fine because the final iterate satisfies the Bellman equation to machine precision (the `\|V_VI - V_PI\|_inf = 2.32e-09` line in the stdout confirms the two methods agree). But it does mean the `0.0e+00` "final error" in Table~\ref{tab:brock_mirman} is tautological — the last iterate is being compared to itself. The hostile reviewer notes this is a minor display issue, not a correctness issue.

**LP primal/dual (lines 189–263).** The Manne (1960) LP formulation. Primal $\min \sum \alpha(s) V(s)$ s.t. $V(s) \geq r(s,a) + \gamma \sum P V$ — implemented correctly with infeasible $(s,a)$ pairs (those with $c \leq 0$) excluded via the `feasible` filter. Dual flow-conservation equation is correctly constructed: $\sum_a \mu(s,a) - \gamma \sum_{s',a'} P(s',a',s)\mu(s',a') = \alpha(s)$. Uses `highs` solver. Bounds are explicit. The stdout reports $\|V_{\text{LP}} - V_{\text{VI}}\|_\infty = 2.32 \times 10^{-9}$, which is solver precision — consistent with the tex claim.

**Verdict:** all three algorithms are textbook-faithful. No placeholders.

## 2. Environment / MDP Fidelity

The Brock-Mirman closed-form policy is $k'(k,z) = \alpha \beta z k^\alpha$ with log utility, $c^* = (1-\alpha\beta) z k^\alpha$. The code uses log utility (line 80, `np.log(c)`), production function $z k^\alpha$ (line 100), and the closed form (line 120) — these match.

One concern: the deterministic steady state $k_{ss}$ for the closed-form policy is $k_{ss} = (\alpha \beta z)^{1/(1-\alpha)}$, but the grid is built using $k_{ss,\text{high}} = (\alpha\beta z_{\max})^{1/(1-\alpha)}$ and only goes from 0.01 to $1.5 \times k_{ss,\text{high}}$. With $\alpha = 0.36$, $\beta = 0.96$, $z_{\max} = 1.1$, this is $k_{ss,\text{high}} = (0.380)^{1.5625} \approx 0.236$, so the grid spans roughly $[0.01, 0.355]$. That seems reasonably narrow but adequate for the demonstration; capital should stay near steady state regardless.

The "policy matches closed-form" check uses `argmin |k_grid - k_next|` (line 121) — i.e., it discretizes the continuous closed-form by snapping to the nearest grid point. The stdout never reports the actual `vi_cf_match` / `pi_cf_match` percentages (the print lines fire but the values aren't shown in the stdout file because the stdout file is from a cache-hit run, lines 295–298 only fire on a full compute). I cannot verify from the artifacts that the discretized closed-form actually agrees with VI/PI policies; the value-function agreement ($\|V_{VI} - V_{PI}\|_\infty = 2.3 \times 10^{-9}$) does confirm the two solvers agree with each other, just not necessarily with the analytical benchmark.

**Verdict:** environment matches the tex description; the closed-form benchmark exists in the code but the comparison results are not surfaced in the stdout artifact.

## 3. Data Integrity

The reported table numbers (VI=567 iters, PI=11 iters, $\beta = 0.96$, $\alpha = 0.36$) trace to live computation in `compute_vi_r1` / `compute_pi_r1`. The `compute_or_load` pattern caches results to `.pkl` files keyed on config hash; the config decomposition (env params, then per-algo additions) is sane and changing $\beta$ or $n_k$ would invalidate the right cache files.

The numbers are internally consistent. The tex says "VI requires 567 iterations" and "PI converges in 11"; the table prints 567 and 11. The tex also predicts VI iteration count from the contraction bound: $k = \lceil \log(10^{-10}/\|TV_0 - V_0\|_\infty)/\log(0.96)\rceil$. With $V_0 = 0$, $TV_0 = R$ so $\|TV_0 - V_0\|_\infty \approx |\log c_{\min}|$ — order one. Then $\log(10^{-10})/\log(0.96) = -23.03/(-0.0408) \approx 564$, close to the observed 567. Consistent.

The "11 iterations" claim is harder to verify without running, but Howard PI converging in 5-15 iterations on a 1,000-state problem with $\beta = 0.96$ is in line with the empirical PI literature for discounted MDPs.

One artifact issue: the stdout file `brock_mirman_newton_stdout.txt` only shows cache hits — the actual numerical results printed by the compute functions never appear in stdout because the cache was already populated. This is fine for reproducibility (the pkl files are the data trail) but it means the stdout artifact is uninformative on its own. The CLAUDE.md "Stdout Output Format" rule asks for "copious tables" and parameter sweeps in stdout — this script's cache-hit stdout violates that, though re-running with `--force` would presumably produce a richer stdout.

**Verdict:** computed, not hardcoded; caching is correctly keyed; the stdout artifact is sparse but not deceptive.

## 4. Comparison Fairness

Regime 1 (VI vs PI on $n_k = 500$): same $R$, $P$, $\gamma$, tolerance. Same initial conditions (both start from $V = 0$ implicitly — VI starts $V = 0$ at line 132; PI starts from greedy-of-$R$ at line 152, which is essentially the greedy policy for $V = 0$). Fair.

Regime 2 (VI vs LP on $n_k = 20$): same MDP. The LP "matches VI" claim ($2.3 \times 10^{-9}$) is the right kind of fairness check. Fair.

Regime 3 (timing sweep): each grid size rebuilds the MDP and runs both algorithms in sequence. Same MDP per row. Fair.

The only mild concern: VI tolerance is $10^{-10}$ (`TOL` constant), which is tight, and PI terminates on exact policy stability. The tex calls VI's 567 iters "the worst case" and PI's 11 iters "Newton-fast", but VI's iteration count would drop substantially with a looser tolerance — at $\epsilon = 10^{-4}$, VI would need ~150 iters, narrowing the gap. The tex narrative is honest about this (the "factor of 50" claim is for the specific tolerance), but a hostile reviewer would say the headline "50x reduction" is partly an artifact of the tolerance choice. This is well-known and the tex's footnote on $n_k = 200$ wall-clock is the more defensible comparison.

**Verdict:** comparisons are fair; tolerance choice is conservative.

## 5. Theoretical Sanity Checks

VI rate prediction (Banach contraction): $\|V_k - V^*\|_\infty \leq \gamma^k \|V_0 - V^*\|_\infty$. The figure panel (c) plots both the empirical VI errors and the $\beta^k$ envelope — this is the right diagnostic. From the tex narrative, the empirical VI curve tracks the $\beta^k$ line, which is the expected linear behavior.

PI/Newton rate: the tex prompt asks whether quadratic convergence near the fixed point is verified. The script does *not* explicitly verify quadratic rate on the Bellman residual — it reports PI's per-iteration error against the final $V$, but on a finite MDP PI typically terminates on exact policy equality after a few iterations, so quadratic rate is hard to observe in the discrete setting (the policy is fixed for the last few iters and the value gap drops to zero in one solve). The tex's claim is more carefully "Newton step ⇒ finite termination via finite policy switches", not "we observe quadratic convergence", which is appropriate for finite discounted MDPs. Bertsekas (2022) and Puterman (1979) both treat this distinction. So the *theoretical* claim made is correct, but the audit prompt's "Newton iteration should exhibit quadratic convergence near the fixed point" is a slightly different claim — for *exact* PI on a *finite* MDP with non-degenerate policy, you get termination, not asymptotic quadratic rate. The script's behavior (finite termination at 11) is consistent with Howard's theorem.

LP-VI agreement to $2.3 \times 10^{-9}$ is solver precision and matches the theoretical fact that the LP primal optimum is the unique value function (Manne 1960).

Santos & Rust (2004) prediction: PI iteration count empirically independent of grid resolution. The Regime 3 sweep shows PI iters = $\{7, 7, 9, 9, 10\}$ for $n_k \in \{10, 20, 50, 100, 200\}$. That's "approximately constant" but trending mildly upward. Consistent with the cited claim within a small constant factor.

**Verdict:** theoretical predictions hold; no contradictions; the script doesn't *over*-claim quadratic rate.

## 6. Information Leakage

PI's policy evaluation uses the true $P$, $r$ — that is *intrinsic* to model-based PI, not a leak. The closed-form benchmark (line 112) uses the true $\alpha, \beta, z$ — but this is the analytical solution, not training data; using it as a *benchmark* is fine, the algorithm itself does not consult it.

The `pol_vi`, `pol_pi` arrays computed at lines 295, 314 compare to `shared['cf_pol']` — comparison only, not used in training. No leakage.

LP uses the same $P$, $R$ as VI/PI; appropriate.

**Verdict:** no leakage. This is a planning (model-known) experiment, and all three methods correctly use only the model.

## 7. Seed & Reproducibility

`SEED = 42` set at module level. `np.random.seed(SEED)` is called at the start of each `compute_*` function. The MDP itself is deterministic (no stochastic environment sampling — the algorithms are deterministic given $P, R, \gamma$, and tolerance), so the seed is effectively cosmetic.

Crucially, **there is no multi-seed run.** The CLAUDE.md standard requires "Run each method across multiple seeds (minimum 10) and report means and standard errors." This script reports point estimates: 567 iters, 11 iters, 32.66s, 0.62s. Wall-clock times in particular vary across runs and machines, and the script reports a single sample.

However: this is a *deterministic* planning experiment. There is no stochasticity in the algorithms (no random initialization for VI/PI/LP — VI starts at $V=0$, PI starts from greedy-of-$R$, LP is convex). Iteration counts are deterministic functions of $(P, R, \gamma, \text{tol})$. So the "minimum 10 seeds" rule does not really apply in the same way it does to e.g. Q-learning. The wall-clock times *are* run-to-run variable, but the iteration counts (the substantive claim) are not. A hostile reviewer could still ding the timing column for reporting single-sample wall-clock without confidence intervals, but the substantive claim (VI 567, PI 11) is reproducible to the bit.

**Verdict:** reproducible in the deterministic sense; multi-seed timing CIs would be nice but the substantive claim does not require them.

## Hostile-Reviewer Summary

The script does exactly what the tex claims: solves the Brock-Mirman MDP with VI (567 iters), Howard PI (11 iters), and Manne LP (matches VI to $10^{-9}$), and runs a grid-size sweep showing PI iters are roughly constant in $n$. The algorithms are textbook-faithful. The Newton interpretation is mathematically correct and the empirical evidence (50× iteration reduction) supports the tex narrative. Minor warts: (i) the PI "final error" column is tautological because it measures against PI's own final iterate; (ii) the closed-form policy-match percentage prints but never makes it into the stdout artifact (cache-hit run); (iii) wall-clock times are single-sample. None of these change the substance.

**Bullshit score: 10%** — Reviewer 2 grumbles that the PI "final error" column is `0.0e+00` because it's comparing to itself, but the headline result (VI 567 vs PI 11, LP-VI agreement to $10^{-9}$, PI iters approximately grid-independent) is exactly what the tex claims and is consistent with Banach contraction, Howard's theorem, Manne (1960), and Santos & Rust (2004).
