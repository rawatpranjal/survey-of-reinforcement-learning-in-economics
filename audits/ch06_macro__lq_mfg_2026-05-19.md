# Audit: ch06_macro/sims/lq_mfg.py

**Date:** 2026-05-19
**Diagram-only:** no (the script itself does no training; it aggregates and renders an external grid. See §3.)
**Cited tex file(s):** `/Users/pranjal/Code/rl/ch06_macro/tex/macro_rl.tex` (subsection `sec:macro:mfg_sim`, lines 588-667; table `tab:macro:lqmfg-results`; figure `fig:macro:lqmfg-curves`).
**Cited paper PDFs read:** `/Users/pranjal/Code/rl/ch06_macro/papers/RSPG.pdf` (Wibault et al., 2026, Recurrent Structural Policy Gradient for Partially Observable Mean Field Games); extracted preliminaries in `papers/extracted/RSPG.md`.

## 1. Algorithm Identity

The artifact does NOT implement the algorithm. `lq_mfg.py` is a pure post-processing script: it loads `mfax_lq_grid_results.json`, aggregates means / SEMs across seeds, picks the best learning rate, and renders a table + 2-panel figure. The actual training is delegated to the public MFAX repository (Wibault et al., commit `9acc1eb`) via the thin subprocess wrapper `mfax_lq_run_grid.py`, which invokes `mfax/algos/hsm/algos/spg.py` and `mfax/algos/hsm/algos/rspg.py` with the paper's own configs (`linear_quadratic_spg.yaml`, `linear_quadratic_rspg.yaml`).

Identity claims to verify against:
- The paper's Algorithm box (RSPG, §3.3/§4 in RSPG.md) requires: (i) analytic mean-field update via the operator $\Phi^\pi$ over the finite state simplex; (ii) Monte-Carlo sampling of common-noise paths; (iii) policy gradient with a recurrent encoder over the history of shared observations; (iv) exploitability evaluated by best-response backward induction against the analytic mean-field path. Tex matches this description (lines 419-466).
- The wrapper passes `--task linear_quadratic --state-type indices --normalize-obs --normalize-states --common-noise --num-envs 8 --num-iterations 200 --anneal-lr --max-grad-norm 1.0 --eval-frequency 20`. These are the documented MFAX flags for the paper's LQ POMFG benchmark.
- Because the heavy lifting is outsourced to the upstream repo at a pinned commit, the algorithm identity claim is mostly an attestation, not a verification, by this repo. The chapter is honest about this: tex line 620-628 says "The reported results come from the public MFAX implementation."

Concerns the hostile reviewer should raise:
- The `lq_mfg.py` summariser cannot detect a silent identity drift in the upstream commit. There is no checksum of the SPG/RSPG source files, only `commit = 9acc1eb`. If MFAX changes its update rule and someone re-pulls, this artifact will silently mutate. Acceptable for a chapter artifact, brittle for a benchmark.
- The wrapper notes (`mfax_lq_run_grid.py`, docstring) admit local patches were applied: "Python 3.11 may require default_factory compatibility fixes in MFAX dataclasses" and "MFAX scripts must print Return in the Iteration lines for this parser." A patch to printing is benign; a patch to `default_factory` defaults is potentially load-bearing if the patched class is a config used during training. The patch is not in this repo and cannot be inspected. A snarky reviewer will demand the diff.

No claim is made in the prose that the implementation reproduces the closed-form LQ Riccati MFE. (For a partially observable LQ MFG with discrete states and finite-horizon, there is no clean Riccati anchor in any case; the canonical anchor in §1 of the audit prompt applies to continuous LQ MFGs, not the MFAX discrete-state LQ instance.) That is correct restraint, not a hole.

## 2. Environment / MDP Fidelity

The tex section describes the environment as: $s \in \{0,\ldots,98\}$, $a \in \{-3,\ldots,3\}$ (7 actions), $z \in \{-1,+1\}$ (binary common noise), $T = 30$. Reward equation 36 (line 607-611):
$R_t(s,a,\mu,z) = -c_a a^2 + q a (\bar{s}_{t+1}-s) - (\kappa/2)(\bar{s}_{t+1}-s)^2$
with terminal reward $-c_{\text{term}}(\bar{s}_T - s)^2/2$. The text states reward is evaluated using the post-transition population mean, consistent with the MFAX implementation.

Verification status:
- The lq_mfg.py script itself does not instantiate any environment. State / action / horizon claims rely on the upstream MFAX `tasks/linear_quadratic`. The chapter's only handle on these numbers is the JSON: `config = {'task': 'linear_quadratic', 'state_type': 'indices', 'discount_factor': 0.99, ...}`. The JSON does not record $|S|=99$, $|A|=7$, $z=\{-1,+1\}$, $T=30$, $c_a$, $q$, $\kappa$, or $c_{\text{term}}$. A reader cannot verify any of those numbers from the chapter's artifacts; they must take Wibault et al.'s defaults on faith.
- The tex section parenthetically grounds $|S|=99$ via "in the public MFAX implementation," matching the stdout line `num_states=99`. Good. The other numbers in the tex have no in-repo trace.
- $\gamma=0.99$ is verified by the JSON `discount_factor` field.

Reviewer pain points:
- The hyperparameters $c_a, q, \kappa, c_{\text{term}}$ are named in the reward equation but never given numerical values in tex or stdout. A hostile reviewer asks for them. They are pinned by the upstream `linear_quadratic_spg.yaml` config, but you do not show them.
- The horizon $T=30$ in tex disagrees in spirit with the infinite-discount $\gamma=0.99$ in the run config. The paper formalises infinite horizon (RSPG.md line 37) and adds a finite-horizon variant. A reviewer will ask whether $T=30$ is a hard truncation, a discount-equivalent horizon, or the paper's actual finite-horizon variant. The tex does not adjudicate.

## 3. Data Integrity

The numbers in the table and figure flow strictly from the JSON. I verified:
- `mfax_lq_grid_results.json` exists, 220 KB, with `n_results = 60`. Two algos x three learning rates x ten seeds = 60. All `returncode == 0` (the loader rejects failed jobs at line 71-77).
- Each run has 11 curve points at iterations `[0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]`. All `final.iteration == 200`. No suspicious truncation.
- The reported table numbers (SPG @ LR=1e-2: $86.64 \pm 16.29$; RSPG @ LR=1e-3: $60.37 \pm 4.11$) are reproduced in `lq_mfg_stdout.txt` and the `.tex` file. No hardcoded "expected" values; everything is computed in `aggregate_runs` → `write_table`.
- The LR-selection rule (`select_learning_rates`, lines 124-131) picks the LR minimising mean exploitability per method. Final-exploitability-by-LR in stdout (SPG: 1e-4=1824, 1e-3=153, 1e-2=86.64; RSPG: 1e-4=1316, 1e-3=60.37, 1e-2=797) confirms selection is correct.

What the integrity audit cannot do:
- It cannot certify that the JSON itself was produced by an unbiased run. The wrapper `mfax_lq_run_grid.py` is the only authoritative source, and to rerun it the auditor would need `/tmp/mfax` (not present), `/tmp/mfax-venv` (not present), and the local patches mentioned in the wrapper docstring. The artifact is reproducible only in the "we ran this once, here are the JSON bytes" sense, not in the "anyone can re-derive these numbers" sense.
- The wrapper's stderr is merged into stdout (subprocess `stderr=subprocess.STDOUT`) and is preserved in `mfax_lq_grid_stdout.txt` (83 KB), so a forensic re-reader can at least inspect the training trace.

## 4. Comparison Fairness

The only comparison in the table is SPG vs RSPG, both with their best LR from the same grid `{1e-4, 1e-3, 1e-2}`, both with 10 seeds (matched seed set $\{0,\ldots,9\}$), both with the same 200 iterations, same num_envs, same eval frequency, same task, same common-noise toggle, same `--anneal-lr` and `--max-grad-norm 1.0`. That is a fair comparison.

Concerns:
- Best-LR-per-method is fair as a hyperparameter-tuned comparison but biased toward whichever method has the wider sweet spot on this grid. RSPG's best LR is 1e-3; at 1e-2 it degrades to exploitability 797. SPG's best LR is the boundary point 1e-2 (so it might still be improving at higher LR, untested). The tex does not flag this. A reviewer can argue both ways: SPG might be at a boundary, RSPG is at an interior minimum. Mention this in a footnote or extend the SPG LR grid.
- Expected returns ($-1021$ vs $-1186$) are reported with the disclaimer "should not be read as a welfare ranking across methods, since each method induces a different mean-field path" (tex line 638-641). This is the correct caveat. RSPG having lower expected return but lower exploitability is consistent (exploitability is a local best-response gap, not a welfare measure).
- Training time is reported as `train_time` (the model's internal counter) with SEM. Wall clock is recorded separately in the JSON but not in the table. This is fine.

## 5. Theoretical Sanity Checks

The standard sanity anchor for an LQ MFG is the continuous Riccati closed form, with Nash gap $\to 0$. That does NOT apply here, because:
- This is a discrete-state, discrete-action, partially observable LQ MFG. There is no clean Riccati solution and no in-paper analytical $V^\star$ to compare to.
- The relevant theoretical anchor is Wibault et al.'s own Theorem on exploitability decay (paper RSPG §4-5, summarised in RSPG.md lines 23-31 and in the chapter at lines 487-498).

Observed sanity facts in the data:
- Exploitability is positive (good; it is defined as a non-negative best-response gap).
- RSPG < SPG in mean exploitability ($60$ vs $87$), which matches the paper's headline claim that recurrent memory helps under partial observability.
- Variance of RSPG ($\pm 4.11$) is much tighter than SPG ($\pm 16.29$), which is consistent with HSM low-variance + recurrent stabilisation.

What is suspicious:
- The absolute exploitability scale ($\sim 60$-$90$) sits near the upper end of the reward scale ($-1000$ range). The paper does not assert exploitability of zero, and the chapter does not claim convergence to a Nash equilibrium, only "lower exploitability". So this is not a contradiction, but the chapter does not give the reader a yardstick for "how good is 60?" A reviewer can ask: what is exploitability of a uniformly random policy? Of the SPG baseline at convergence? Without an anchor, the numbers float free.
- Exploitability of SPG decreases monotonically with LR from 1e-4 to 1e-2 (1824 → 153 → 87). A reviewer asks: have you saturated? Or is SPG still in the descending part of its LR curve at the grid edge?
- RSPG exploitability is non-monotone in LR (1316 → 60 → 797), which is the canonical "interior optimum" signature. Good.

No theoretical contradictions are visible. The chapter is appropriately modest about what is being claimed.

## 6. Information Leakage

The RSPG agent in the paper observes (i) its own state $s_t$ and (ii) the shared observation $o_t = \bar{s}_t$ (population mean), but NOT the full distribution $\mu_t$, NOT the common-noise realisation $z_t$, and NOT the calendar time $t$. This is the partial-observability statement of the chapter (tex line 600-602).

The script `lq_mfg.py` cannot leak anything because it does not run the agent. The leakage question reduces to "does MFAX's RSPG implementation respect the paper's observation model?" The chapter cannot answer this from its own artifacts; it relies on the upstream repo. A hostile reviewer will demand a verified reduction: a small audit script that dumps the observation tensor seen by the policy at each step and confirms it has no $z$ field, no $\mu$ field, no $t$ field. None is provided here.

Less load-bearing concerns:
- The structural mean-field update is "analytic", i.e. uses the known transition kernel $T$. The paper is explicit that this is by design (HSMs assume white-box access to individual dynamics). The chapter (line 446-449) is also explicit. This is not leakage; it is a feature of the HSM class.
- Exploitability evaluation uses backward induction against the analytic mean-field path induced by the final policy (tex line 626-630). This uses the kernel $T$, which is allowed at evaluation time.

## 7. Seed & Reproducibility

- Seeds: $\{0,1,\ldots,9\}$, $N=10$. Meets the $\geq 10$ threshold in CLAUDE.md.
- Means and SEMs are reported in table, stdout, and figure (shaded SEM band on the learning curves).
- SEM formula uses `ddof=1`. Correct.
- The single-seed SEM short-circuit (`_sem` returns $0$ for $n \le 1$) is unreachable here because each (algo, lr) bin has 10 seeds; harmless defensive code.
- Reproducibility from cache is fully deterministic: rerunning `python3 lq_mfg.py` against the same JSON yields byte-identical table and figure.
- Reproducibility from scratch is NOT verified: requires `/tmp/mfax` checkout at commit `9acc1eb`, a JAX venv, local patches for Python 3.11, and ~24-29 seconds of training per (algo, seed, lr) combo. The wrapper documents this; the chapter does not. A reader who wants to rerun has to read `mfax_lq_run_grid.py` to discover the environment.

## Hostile-Reviewer Summary

The script itself (the post-processing layer) is clean: aggregation is correct, the JSON has 60 jobs with no failures, table and figure are arithmetic transformations of the JSON, no leakage from a post-hoc renderer is possible, seeds and SEMs are reported per CLAUDE.md. The chapter prose is appropriately modest, never claims Riccati convergence, names the source repo and commit, and flags that expected returns are not a welfare ranking.

The genuine weakness is upstream-attestation. The substantive algorithmic work is outsourced to the MFAX repository at one pinned commit, and (i) the chapter cannot verify the algorithm identity beyond "we ran the file the authors named," (ii) the precise reward coefficients $c_a, q, \kappa, c_{\text{term}}$ are not anywhere in this repo, (iii) the local patches that allowed the run on Python 3.11 are not committed here, and (iv) the chapter cannot independently certify the observation model used in MFAX's RSPG matches the paper. A snarky reviewer will write: "the authors aggregated 60 runs from someone else's repo and called it a chapter contribution." A charitable reviewer will accept that this is an honest reuse with the right caveats. Both will note: extend the SPG LR grid past 1e-2 to confirm interior optimum, and give the reward coefficients somewhere.

**Bullshit score: 25%** — Reviewer 2 catches the missing reward coefficients, the SPG LR-grid boundary, and the un-checksummed upstream patches. The substance survives revision: the aggregation is honest, seeds and SEMs are reported, RSPG < SPG is robust across LRs at the best operating point, and the chapter never overclaims a Riccati-style anchor it cannot deliver.
