# Audit: ch05_econ_models/sims/nfxp_ccp_td.py

**Date:** 2026-05-19
**Diagram-only:** no
**Cited tex file(s):** `/Users/pranjal/Code/rl/ch05_econ_models/tex/rl_in_se.tex` (Simulation Study: DDC Estimation at Scale, §sec:ddc_estimation_sim, lines 192–214)
**Cited paper PDFs read:**
- `/Users/pranjal/Code/rl/ch05_econ_models/papers/AdusumilliEckardt2022_td_learning_ddc.md` (the canonical Adusumilli–Eckardt 2022 paper that the TD-CCP variants ostensibly implement; §§1–3.2 read in full)
- Rust 1987 / Hotz–Miller 1993: not in `papers/` as PDFs (only via reference)
- Aguirregabiria–Mira 2002 PMLE: not in `papers/`

## 1. Algorithm Identity

**NFXP (estimate_nfxp, lines 274–302).** Outer L-BFGS-B over (RC, θ₁, θ₂); inner `_vi_solve` does Bellman iteration `EV ← logaddexp(-cost + γ P_keep EV, -RC + γ EV[0])` with tol 1e-8 and max 5000 iters. Matches Rust (1987) NFXP. The likelihood is the binary logit derived from `v0 - v1`. Sparse P_keep and the trick `idx_zero` for the replace branch are correct. **Identity: NFXP, faithful.**

**CCP (estimate_ccp, lines 305–367).** Estimates p̂(a=1|s) from cell frequencies, clipped to [0.01, 0.99]. Builds the conditional transition operator `F = diag(p̂₀) P_keep + p̂₁ · e₀ᵀ` and solves the linear system `(I - γF) EV = r̄(θ) + H(p̂)` for each θ via prefactored sparse LU (n_states ≤ 10000). Then `v₀, v₁` and PMLE. This is the Hotz–Miller / Aguirregabiria–Mira inversion under logit. **Identity: CCP / Aguirregabiria–Mira PMLE, faithful.**

**TD-CCP Linear (estimate_td_ccp_linear, lines 392–462).** Decomposes the EV function into θ-linear pieces, EV(s; θ) = θ₁·ev₁(s) + θ₂·ev₂(s) + RC·ev_RC(s) + ev_H(s). Each ev_k satisfies a Bellman recursion under the *observed* policy with flow vector matching the θ-coefficient's term. Each is solved by the standard linear semi-gradient TD fixed-point equation,

  Â w = b̂, with Â = (1/n)·Σ φ(sₜ)·(φ(sₜ) − γφ(sₜ₊₁))ᵀ, b̂ = (1/n)·Σ φ(sₜ)·flow(sₜ),

via `np.linalg.lstsq`. This matches Adusumilli–Eckardt eq. (3.3)–(3.5) (the population form replacing E by Eₙ; verified against §3.1 of the paper).

Two subtleties a hostile reviewer would raise. (i) The paper approximates *h(a,s)* (a function of both action and state) and *g(a,s)* directly; here the action dependence is folded into the θ-linear decomposition (the keep-branch flow vs. replace), and the basis is state-only `φ(s)` (polynomial in x with per-component features when K>1). This is an algebraically equivalent reformulation under binary action with one-period reset on `replace`, but it is not the paper's algorithm as written. (ii) **The locally robust correction (eq. (4.x) of the paper, the main theoretical contribution that yields √n-convergence) is omitted entirely.** The PMLE is the naive plug-in. The tex acknowledges using "semi-gradient TD with polynomial basis" but does not say "plug-in PMLE without local-robustness", which is what the code actually does. **Identity: half-faithful — TD step yes, locally robust PMLE no.**

**TD-CCP Neural (estimate_td_ccp_nn, lines 465–565).** Same θ-linear decomposition, but each `ev_k` is fit by a two-layer MLP via 20 outer AVI passes × 30 inner SGD epochs. The bootstrap target is computed with `torch.no_grad()` and refreshed once per AVI iter (target-network AVI per eq. (3.10) of the paper). Then PMLE plug-in. Same caveats as TD-CCP Linear: (i) algorithm reformulated via θ-linear decomposition rather than fitting h(a,s) directly; (ii) no locally robust correction. The architecture also takes only the per-component normalized values `m_k/M` as features — fine for K up to 4 but does not capture cross-component interactions beyond what the MLP learns. **Identity: spirit of AVI yes, paper's exact PMLE no.**

## 2. Environment / MDP Fidelity

`MultiComponentBusEngine` extends Rust (1987) Zurcher to K independent wear components, each in {0,…,M−1} (M=20), with aggregate normalized wear x(s) = Σₖ mₖ/M entering cost `c(s;θ) = θ₁ x + θ₂ x²`. Transition for `keep` is a per-component independent increment drawn from `trans_probs=[0.4, 0.4, 0.2]` over {1,2,3}, capped at M−1. For `replace`, state resets to (0,…,0). Reward is `-c(s;θ)` for keep and `-RC` for replace. Logit errors and γ=0.95. This matches the tex (§sec:ddc_estimation_sim, "multi-component extension of Rust 1987" with K∈{1,…,4}, M=20, N=500, T=100).

`build_transition_matrix` builds a sparse |S|×|S| matrix by enumerating Kᵏ increment combos per state — correct, and the capping behaviour (`min(m+δ, M-1)`) is consistent with a reflecting/absorbing upper boundary, which is the standard Rust convention. csr_matrix sums duplicate (i,j) entries from capping, which the code comments correctly.

One soft flag: **the rewriting from M=90 mileage bins (Rust 1987) to M=20 per component is not justified in the tex.** A reviewer would want the bin choice motivated, or at minimum noted as a sensitivity. Fidelity to Rust's specific Zurcher data is not claimed, only fidelity to the model class.

## 3. Data Integrity

`compute_data` (line 639) calls `load_results(CACHE_DIR, ...)` to short-circuit if the pickle is present; otherwise calls `run_experiment_2()` which does actual VI, panel simulation, and four estimators per seed × per K.

The cache `nfxp_ccp_td.pkl` contains keys `['exp2', '_config_hash']` and was written 2026-03-16 (Unix mtime 1773637745). The stdout file `nfxp_ccp_td_stdout.txt` shows real per-seed log lines (`Seed 4/5 NFXP RC= 4.976 th1=1.849 th2=4.130 (172.1s)`, etc.) matching the table — NFXP K=4 timing avg ≈ 172.6s ↔ stdout 165.6s, 172.1s, plus three other seeds. Numbers in `nfxp_ccp_td_results.tex` line-up with what the cache would produce; the table is not hardcoded.

**Critical reproducibility break.** The committed script does **not run** as written. Lines 22–24 import

```python
from sims.sim_cache import compute_or_load, add_component_args, parse_force_set
```

but the body calls `load_results`, `save_results`, `add_cache_args` (lines 640, 648, 739, 751) — none of which are imported. I verified this by running `python3 ch05_econ_models/sims/nfxp_ccp_td.py --plots-only`:

```
NameError: name 'add_cache_args' is not defined
```

The pickle and the table date from an older, working version of the script. Anyone attempting to reproduce or update the numbers from the current `HEAD` (`4cdbac6`) cannot, without first fixing the imports. The stdout file itself is also degenerate: it contains the *header banner of a recent (failed/cached-load) attempt* on lines 1–12, then a separate dump of the *tail of an earlier real run* on lines 13–32 (no `EXPERIMENT 2` header, no full per-seed trace for K=1..3, just the K=4 seed-4 and seed-5 lines). This is a clear sign that the stdout was not regenerated end-to-end after the script was modified.

So: the *numbers shown in the table and figure are real* (they came from a previous working run), but the *artifact as committed is broken* and the stdout is a frankenstein.

## 4. Comparison Fairness

Same data per (K, seed): `env.simulate_panel(v0, v1_vec, N=500, T=100, seed=seed)` is called once per seed and the resulting `sim_states, sim_actions` are passed to all four estimators. Same target parameters (RC=5.0, θ₁=2.0, θ₂=4.0). Same env_config and precomputed sparse P_keep.

Same optimizer (L-BFGS-B, maxiter=200, bounds=[(0.1, 20.0)]³, starting point [4.0, 1.5, 3.0]) for all four PMLEs. Same NFXP tolerance (vi_tol=1e-8). Each method's wall-clock includes its own inner cost (VI for NFXP, sparse LU for CCP, lstsq for TD-Linear, AVI MLP training for TD-Neural), which is the fair definition.

CCP gets a SPARSE early-exit if state coverage `(count_s >= 5).sum() / n_states < 0.1`. At K=3 (|S|=8000) and K=4 (|S|=160k) with only N×T = 50k observations, this triggers (coverage ≈ 0.4% at K=4 per stdout). The table correctly reports CCP failure as dashes at K=3, 4. This is not unfair to CCP — it is the known curse-of-dimensionality failure mode the figure is designed to demonstrate. **Fair.**

One soft asymmetry: **TD-CCP Linear and TD-CCP Neural cheat slightly by being given the θ-linear structure of the flow.** They get the analytical separation of `flow_1 = -p̂₀·x`, `flow_2 = -p̂₀·x²`, `flow_rc = -p̂₁`, `flow_H = H`, which is only available because the analyst knows the parametric form `c(s;θ) = θ₁x + θ₂x²`. Adusumilli–Eckardt's actual h(a,s) approach does not require this decomposition. NFXP and CCP also use this parametric form, so the comparison is consistent; but a reader expecting an apples-to-apples nonparametric h-fit will be surprised. The tex does not mention the θ-linear decomposition trick anywhere — a reviewer would call this an undocumented modification.

## 5. Theoretical Sanity Checks

True params: RC=5.0, θ₁=2.0, θ₂=4.0.

| Method | K=4 RC RMSE | RC bias |
|---|---|---|
| NFXP | 0.070 | −0.036 |
| TD-CCP Neural | 0.069 | −0.029 |
| TD-CCP Linear | 0.337 | −0.333 |

NFXP and TD-CCP Neural deliver the consistent estimator that Rust (1987) Theorem and Adusumilli–Eckardt Theorem 5 promise. The RC bias of about −0.04 at K=4 with 5 seeds, 500 agents × 100 periods, is the right order of magnitude (≈ σ/√(NT) for a logit binary choice with this signal-to-noise; 0.04/5 ≈ 1% relative). NFXP's θ₂ RMSE drops from 0.618 at K=1 to 0.113 at K=4 — counterintuitive at first glance, but reasonable: more components → more total wear states sampled → more information about the curvature.

CCP is consistent in theory (Hotz–Miller 1993 Thm) but fails operationally when state coverage is sparse. The table correctly shows CCP succeeding at K=1, K=2 with bias and RMSE close to NFXP (RC RMSE 0.103 and 0.085 vs NFXP 0.102 and 0.077), then failing at K=3, K=4. **Consistent with theory.**

TD-CCP Linear's RC bias *grows monotonically with K* (−0.083, −0.211, −0.251, −0.333). The tex attributes this to "basis misspecification at higher K". The basis is degree-2 polynomial in (x, m₁/M, …, m_K/M). For K=4, this gives 1 + 1 + 1 + 4 + 4 = 11 features over a 160k state space. Under-parametrized basis → biased TD fixed-point → biased PMLE. The story is internally consistent.

But there is an alternative reading that the chapter does **not** consider: the bias may come from the **θ-linear decomposition itself**. If the *true* EV is not separable as θ₁·ev₁(s) + θ₂·ev₂(s) + RC·ev_RC(s) + ev_H(s) under the observed policy (because the observed policy depends on θ through the data, but the analyst freezes the observed CCPs once and never re-solves), then the linear PMLE is misspecified even with an arbitrarily rich basis. The chapter does not warn the reader that this is a one-shot inversion that does not iterate (i.e., it is not Aguirregabiria–Mira NPL). A hostile reviewer in Econometrica would flag this. The tex glosses it by saying TD-CCP Linear is "TD-CCP Linear (semi-gradient TD with polynomial basis)" without mentioning the θ-linearity reformulation.

TD-CCP Neural's RC RMSE matches NFXP almost exactly at K=2..4. Given that the neural net is trained from scratch on the same panel as TD-Linear and uses the same θ-linear decomposition trick, this is evidence that the bias in TD-Linear is from basis misspecification, not from the decomposition itself — a partial defense of the chapter's reading. **Theoretical sanity: passes for NFXP and CCP; passes for TD-Neural empirically; TD-Linear's failure mode is plausibly explained but not rigorously diagnosed.**

## 6. Information Leakage

Estimators receive (states, actions) only — no access to true (RC, θ₁, θ₂). Starting point [4.0, 1.5, 3.0] differs from truth [5.0, 2.0, 4.0] by 20-25% on each component; bounds [0.1, 20.0]³ are wide enough that the optimizer is not boxed near truth. The `precomp` object passed to each estimator contains only `(n_states, x_vec, P_keep, idx_zero)` — the sparse transition matrix is structural knowledge (Rust assumed known transition probabilities of the increment distribution `[0.4, 0.4, 0.2]`). This is consistent with the standard DDC convention where the transition density of the discrete mileage increment is treated as nonparametrically estimable from data; here it is built from the true `trans_probs`, so all four estimators get the *truth* of K(·|·).

This is a mild concession but it is the standard convention in the Rust literature (transition probs are often estimated nonparametrically and treated as known by the structural step). One caveat: **TD-CCP estimators' main marketed advantage is precisely that they don't need K(·|·)**. The code passes K(·|·) to them anyway, via the precomputed `P_keep` used at the PMLE step `v0 = -cost + γ P_keep ev_k`. This silently undercuts the main contrast the tex draws between "NFXP/CCP needs transitions" and "TD-CCP doesn't". A hostile reviewer in Econometrica would call this out: if you want to demonstrate the transition-free advantage, you must not pass `P_keep` to TD-CCP at PMLE time. The estimators get the *sample successors* anyway via `s_t1_arr`, which is what Adusumilli–Eckardt actually use, but the PMLE step using `P_keep.dot(ev_k)` is a structural-info shortcut.

**No leakage of θ; structural leakage of K(·|·) into TD-CCP that contradicts the chapter's framing.**

## 7. Seed & Reproducibility

`exp2_seeds = 5` (line 48 of CONFIG). Project standard is N≥10 (per `CLAUDE.md` Study Design section). Five is below threshold.

**No SE reporting.** Table reports `RC Bias` (mean − truth) and three RMSEs — no standard errors. With 5 seeds, SE on RMSE is ≈ RMSE / √(2·5) ≈ 30% relative; reporting RMSE to three decimals (0.337, 0.253, …) is significantly more precision than the data support. CLAUDE.md says "report means and standard errors" — this requirement is violated.

The seeds are set explicitly per seed iteration (`env.simulate_panel(..., seed=seed)`) using `np.random.RandomState(seed)`. PyTorch seeds are **not** set anywhere; the TD-CCP Neural training (Adam SGD over batched data, `torch.randperm`) is non-deterministic across runs. So even if you fix the panel data via `seed`, two reruns of TD-CCP Neural will produce different θ̂. The reported timings (33–45s per seed) and RMSE (0.069 at K=4) are *one realization* of this non-determinism. This is a moderate reproducibility flag — not fatal because we average 5 seeds and the panel data are the same, but the within-seed RMSE component from MLP training stochasticity is not isolated.

**Reproducibility verdict.** Five seeds, no SEs, MLP not seeded, *and* the current script does not run at all due to broken imports (§3). Anyone attempting to reproduce the table by running the script today gets a NameError on the first line of `main()`. The numbers in the table are real (they came from a previous working version), but the audit trail to reproduce them is severed.

## Hostile-Reviewer Summary

What a Reviewer 2 in *Journal of Econometrics* says after one read:

1. "The script you submitted does not execute. NameError on `add_cache_args`. I cannot reproduce a single number in your Table." (§3, §7.)
2. "The TD-CCP methods you implement are not the methods of Adusumilli–Eckardt. You introduce a θ-linear decomposition of the EV that is not in the paper, drop the locally robust PMLE correction that is the paper's main theoretical contribution, and pass the true transition matrix `P_keep` to the PMLE step — silently undercutting the very advantage your chapter claims." (§1, §6.)
3. "Five seeds is below your own stated standard of ten. No standard errors are reported. You quote RMSE to three decimals when SE on the RMSE estimate is roughly ±30% relative." (§7.)
4. "Your reference to Adusumilli–Eckardt (2022) in `refs.bib` adds a third author 'Tate, G.' who does not exist on the paper, and gives a title ('Differentiable Temporal-Difference Learning') that differs from the actual title ('Temporal-Difference Estimation of Dynamic Discrete Choice Models')." (citation-fidelity flag; the script itself is silent on this, but a reviewer reading the chapter and the bib together will catch it.)
5. "Your tex says 'NFXP scales from 0.2s at K=1 to 179s at K=4', but your table says 172.6s. Pick one." (minor, §5.)
6. "The PyTorch training in TD-CCP Neural is not seeded; the numbers in the table are a single realization of a stochastic training procedure." (§7.)

The substance of the K=4 RMSE comparison (NFXP 0.070, TD-Neural 0.069, TD-Linear 0.337) is right and would survive revision. The scaling story (NFXP 0.2s → 172s; TD-Neural ≈ 30–44s flat; CCP fails at K≥3) is right and is the only thing the figure is asked to show. But the audit trail is fragile, the algorithm-identity claim for the TD-CCP variants is shakier than the tex admits, and **the script-as-committed does not run at all** — so a sceptical reader cannot independently confirm any of it. This is squarely in the "Reviewer 2 catches it and writes a snarky paragraph; substance survives revision" zone, pushed up by the non-running script and the citation-fidelity bib bug.

**Bullshit score: 50%** — A reviewer with one afternoon will find: (i) the artifact does not execute, (ii) the TD-CCP variants are reformulated/simplified vs. the paper they cite without disclosure, (iii) the locally robust correction (the paper's main theorem) is missing, (iv) P_keep is fed to PMLE for TD-CCP undercutting the framing, (v) the cited bib entry has a hallucinated co-author. The numbers themselves survive scrutiny, but the path from script to numbers does not. Major revise.
