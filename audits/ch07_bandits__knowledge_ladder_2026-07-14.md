# Audit: ch07_bandits/sims/knowledge_ladder.py

**Date:** 2026-07-14
**Type:** DELTA (promoted from a 15%-polished sim because the committed stdout dates 2026-03-09 while the .py's last commit is 2026-03-23, so the data-integrity question is whether the published numbers come from the current code).
**Diagram-only:** no (Monte Carlo, 10 seeds x T=200,000 x 6 algorithms).

**Delta summary.** The 2026-03-23 commit (`4cdbac6`) to `knowledge_ladder.py` is a pure plumbing refactor plus a cosmetic label change. It wraps the top-level script body into `compute_data()` / `generate_outputs(data)`, adds `sim_cache` (CONFIG dict, `load_results`/`save_results`), adds `argparse` with `--data-only`/`--plots-only`, and switches hard-coded output paths to `os.path.join(OUT_DIR, ...)`. It also rewrites the `alg_labels` dict to append rate annotations (`Thompson -- $O(\sqrt{KT})$` instead of `Thompson Sampling`). It touches **no** algorithm class, **not** the demand model, and **not** `run_experiment` (the compute loop, seeds, and regret accounting). Consequence: pre-change and post-change numbers are identical, so the published figure/table/prose numbers reflect the current code. The committed stdout's *numbers* are current; only its *header labels* are stale (they show the pre-03-23 labels). Data Integrity passes; the delta is benign.

**Files read (full):**
- `/Users/pranjal/Code/rl/ch07_bandits/sims/knowledge_ladder.py`
- `/Users/pranjal/Code/rl/ch07_bandits/sims/knowledge_ladder_stdout.txt`
- `/Users/pranjal/Code/rl/ch07_bandits/sims/knowledge_ladder_results.tex`
- `/Users/pranjal/Code/rl/ch07_bandits/sims/knowledge_ladder_diagnostics.tex`
- `/Users/pranjal/Code/rl/ch07_bandits/sims/knowledge_ladder_regret.png` (viewed)
- `/Users/pranjal/Code/rl/ch07_bandits/tex/dynamic_pricing.tex` (lines 41-171)
- `git diff 54ae7cf 4cdbac6 -- knowledge_ladder.py` (and the same range for the two .tex and the .png)
- Prior audits (Step 6 only): `audits/ch07_bandits__knowledge_ladder_2026-05-19.md`, `audits/ch07_bandits__knowledge_ladder_polish_2026-05-20.md`

---

## Step 3 -- thesis statement (what this sim is evidence FOR)

(i) **Theoretical claim.** The `\subsection{Revealed Preference and Partial Identification}` argues that economic structure (WARP: a buyer who purchases at price $p$ would purchase at any $p' < p$) turns a binary purchase into partial identification of each segment's valuation interval $[p_s^{\min}, p_s^{\max}]$, which can prune dominated prices and, per Misra (2019), drive pricing-bandit regret down toward $O(\log T)$. The chapter frames a "ladder" from no structure ($\varepsilon$-greedy, $\Theta(T)$) up to WARP-based partial ID (UCB-PI, $O(\log T)$).

(ii) **What the sim is used for.** Six algorithms ordered by the structural knowledge they exploit are run on the Misra (2019) demand environment to trace how cumulative regret responds to increasing structure. The consuming prose (line 161) explicitly scopes it as "a WARP and partial-identification demonstration, not a horse race against GP-UCB, GP-TS, or monotone GP demand-curve methods," and (line 165) narrows the conclusion to: the variance-tuned WARP method (UCB-PI-tuned) wins in finite sample, while the clean monotone ladder does **not** hold at $T=200{,}000$ (the finite-sample order inverts the asymptotic-rate order).

---

## Criteria

### (a) Correctness -- PASS with disclosed caveats
- **Update rules** are recognizable instantiations of their names (re-verified against the code; prior audit checked term-by-term vs Auer 2002 and Misra 2019): UCB1 `values + sqrt(2 log(t+1)/n)` (line 157); Beta-Bernoulli price-scaled TS `argmax p_k*theta` (lines 176-178); explore-then-commit LTE with `T_explore = 0.05*T` (line 128); WARP `p_min/p_max` update and dominance test `ub_profits <= max_lb` (lines 205-256); UCB-V-style tuned bonus `2*p_k*delta_hat*sqrt((log(t+1)/n)*min(0.25, V_kt))` (lines 285-291).
- **Ad-hoc `delta_hat`** (lines 232-233): `gamma_hat = delta_max - mean_delta; delta_hat = delta_max + gamma_hat` (= `2*delta_max - mean_delta`). Not a published formula, but **disclosed verbatim** in the tex footnote at line 163 ("I am not aware of a published source for this exact form ... a transparent reference point"). Code matches the disclosure exactly. Acceptable because it is flagged, not attributed to the paper.
- **Regret vs theory:** no method beats the oracle; the rate labels do *not* empirically settle over the horizon (UCB-PI's $R/\log T$ quadruples), but the prose concedes this in full. Minor: tuned bonus uses `log(t+1)` where Misra Eq. (8) uses `log(n+S)` (immaterial at large $t$ with $S=1{,}000$).

### (b) Presentation / numbers -- numbers PASS, presentation GAP
- **Number-consistency sweep (complete).** Every published number traces to a generated artifact and all four surfaces agree:
  - `results.tex` checkpoint means == stdout checkpoint means, cell by cell: eps 160/624/1,158/2,263; LTE 1,005/1,165/1,363/1,771; UCB1 719/2,609/4,263/6,734; TS 290/628/844/1,136; UCB-PI 789/2,266/3,244/4,503; tuned 338/541/640/780. All match.
  - `diagnostics.tex` == stdout rate-diagnostic table, cell by cell (e.g. 200K row: 0.0113 / 0.5 / 15.1 / 2.5 / 368.9 / 63.9). All match.
  - Prose (line 165) claims all verify: "tuned lowest by T=200,000" (780 = min); "plain UCB-PI worse than TS and LTE" (4,503 > 1,136, > 1,771); "$R/\log T$ for plain UCB-PI quadruples" (85.7 -> 368.9 = 4.30x); "only TS's $R/\sqrt{T}$ close to stable" (2.9->2.5); "UCB1's $R/\sqrt{T}$ grows" (7.2->15.1); env footnote (K=100, \$0.01-\$1.00, S=1,000 equal weights, delta=0.1, $v_s\sim U(0.1,0.9)$, buy iff $v_i\ge p$, 10 seeds, T=200,000) all match the code (lines 32-37, 71-72, 361).
  - Figure end-point ordering matches the table (UCB1 highest, UCB-PI second, then eps crossing above LTE, TS, tuned lowest).
- **Gap 1 -- orphaned tables.** `knowledge_ladder_results.tex` and `knowledge_ladder_diagnostics.tex` are generated but `\input` **nowhere** (`rg` over all `*.tex` returns zero references; the distinctive values 6,734 / 4,503 / 2,263 / 368.9 appear in no `.tex`). The document shows **only the figure** (line 169). Yet the prose refers to "cumulative regret at four checkpoints" and "the rate-diagnostic columns" -- content the reader cannot see. Also, line 165 attributes "reports cumulative regret at four checkpoints" to the *figure*, which is a continuous log-log curve, not a checkpoint table. This violates the CLAUDE.md "one consolidated table of all results" standard (no table reaches the page).
- **Gap 2 -- stale stdout.** The committed stdout header labels are from the pre-03-23 code (`UCB-PI (WARP)`, `Thompson Sampling`, no rate suffix; stdout lines 25, 48-53). Current `alg_labels` (lines 300-307) carry rate suffixes, so re-running current code would print a different header. Numbers are unaffected; this is a process lapse against "always update _stdout.txt after any sim change."

### (c) Chapter fit -- PASS (narrowed claim, honestly)
The sim does **not** demonstrate a clean monotone knowledge ladder: by final regret the order is tuned (780) < TS (1,136) < LTE (1,771) < eps (2,263) < UCB-PI (4,503) < UCB1 (6,734), so Levels 2 and 4 are worse than Level 0. The `.py` docstring still carries the aspirational monotone framing (Level 0..5 with clean rate assignments), but the **published prose was rewritten** (line 165) to concede the inversion and to restrict the takeaway to "WARP-based elimination helps relative to plain UCB1, but good finite-sample performance depends on the variance-tuned implementation." The chapter demonstrates the narrower claim it actually makes.

### (d) Efficiency / standards -- PASS (at floor), minor deviations
- `sim_cache` integrated (CONFIG dict invalidates on K/T/N_SEEDS/delta change), `--data-only`/`--plots-only` present (lines 662-672). Added by the 03-23 refactor, so caching/flags are current.
- Seeds = 10 (exactly the CLAUDE.md floor); SE reported in stdout and as +/-2 SE bands in the figure. LTE relative SE ~13.7% (242 on mean 1,771) argues for more seeds.
- Stdout format follows the standard (config header, results tables, output paths). Deviation: no sim table reaches the compiled document (see Gap 1); stdout is stale (Gap 2).

---

## 7-point checklist

1. **Algorithm identity -- PASS.** Six recognizable bandits; ad-hoc `delta_hat` disclosed in the tex footnote and matched by code.
2. **Environment fidelity -- PASS.** Code (lines 32-37, 71-72, 361) matches Misra (2019) Section 4 and the tex footnote exactly.
3. **Data integrity -- PASS (delta central question).** The 03-23 change altered no computation. `results.tex`/`diagnostics.tex` are byte-identical across `54ae7cf..4cdbac6` (empty diff); stdout numbers == tex numbers. The PNG re-rendered (404,162 -> 400,245 bytes) only because the legend labels changed; plotted data is identical. Published numbers reflect current code. Not hardcoded (`regret_arrays[name][:, idx].mean()`).
4. **Comparison fairness -- PASS.** Common-random-numbers: one `segment_ids` stream and one `valuation_offsets` stream per seed (lines 320-321); every algorithm sees the same `(segment_id, v_i)` at each $t$; per-algorithm RNGs isolate internal randomization (lines 324-329).
5. **Theoretical sanity -- PARTIAL, flagged.** Rate-diagnostic columns mostly fail to stabilize at T=200,000 and the ladder order is non-monotone; both are stated in the prose (line 165). No method beats the oracle.
6. **Information leakage -- PASS.** `segment_id`/`purchased` reach only the methods the model entitles (UCB-PI/tuned); no algorithm reads `OPTIMAL_ARM` or the true model; `V_L`, `V_H`, `segment_weights` are known-by-assumption per the paper; `DELTA_TRUE` is not used by any algorithm.
7. **Seed / reproducibility -- PASS at floor.** `np.random.seed(42)`; N=10; per-seed and per-algorithm RandomStates; SE reported. N=10 is the minimum; LTE's ~14% relative SE would benefit from more.

---

## Prior-audit open-item disposition

From `2026-05-19` (25%) / `2026-05-20` polish (15%):
- **Homegrown `delta_hat` estimator -- RESOLVED (disclosed).** Tex line 163 footnote states the exact formula and its provenance; code unchanged, which the polish brief explicitly chose (option B). Still an unusual estimator, but transparently flagged.
- **Rate-diagnostic columns do not stabilize -- RESOLVED (broadened).** Tex line 165 now walks every column ("only TS's $R/\sqrt{T}$ close to stable; ... $R/\log T$ for plain UCB-PI quadruples").
- **Non-monotone ladder / legend orders by claimed rate -- RESOLVED (flagged).** Tex line 165 states the finite-sample order inverts the asymptotic order.
- **10 seeds is a floor; LTE relative SE ~14% -- STILL OPEN.** Unchanged (N_SEEDS=10; LTE SE 242 on 1,771).
- **Figure legend still orders by claimed rate -- STILL OPEN, but flagged in prose and in the figure caption ("legend entry includes its theoretical regret rate").** Acceptable.
- **No regression** on any prior item.

**New (not in prior audits):** the two result tables are orphaned (never `\input`), so the document contains no sim table while the prose references checkpoint values and diagnostic columns the reader cannot see (Gap 1); and the committed stdout header labels are stale relative to current code (Gap 2).

---

## Findings, severity-ordered

1. **(Presentation, moderate, NEW) Orphaned result tables.** `knowledge_ladder_results.tex` and `knowledge_ladder_diagnostics.tex` are generated but `\input` nowhere; the compiled chapter shows only `knowledge_ladder_regret.png`. The prose (line 165) cites "cumulative regret at four checkpoints" and "the rate-diagnostic columns," and even attributes checkpoint reporting to the figure, but no such table appears. Fix: `\input` at least the checkpoint table (or drop the checkpoint/column phrasing). Evidence: `rg` over all `*.tex` finds no reference to either filename; `dynamic_pricing.tex:169` is the only sim output included.
2. **(Process, minor, NEW/delta) Stale stdout.** `knowledge_ladder_stdout.txt` (committed 2026-03-09) predates the .py's 2026-03-23 label change; its header rows (lines 25, 48-53) show the old `alg_labels` without rate suffixes. Numbers are unaffected (identical computation). Fix: regenerate stdout from current code. Evidence: `git diff 54ae7cf 4cdbac6 -- knowledge_ladder.py` shows the `alg_labels` rewrite; stdout still carries the old labels.
3. **(Correctness/exposition, minor, carried) Homegrown `delta_hat` and non-settling rate labels.** Disclosed in the tex; substance defensible. Carried from prior audits, no action required.
4. **(Standards, minor, carried) N=10 seeds at the floor; LTE relative SE ~14%.** More seeds would tighten LTE. Carried.

**Bullshit score: 20%** -- Data integrity is clean: the 2026-03-23 change is pure plumbing plus cosmetic labels, tables are byte-identical across the commit, and every published number traces and cross-checks, so the delta is benign; Reviewer 2 nonetheless catches that the prose cites four-checkpoint values and rate-diagnostic columns while the document ships only a figure (both result tables are orphaned) and that the committed stdout's header labels are stale, but the substance survives a one-`\input` revision.
