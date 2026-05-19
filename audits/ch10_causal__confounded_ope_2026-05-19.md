# Audit: ch10_causal/sims/confounded_ope.py

**Date:** 2026-05-19
**Diagram-only:** no
**Cited tex file(s):** `ch10_causal/tex/causal_rl.tex` (§3.1 backdoor OPE, §3.4 IV/Wald, §3.5 front-door, §3.6 proximal; simulation §3.7 "Simulation Study: Confounded Retail Pricing MDP", lines 259–284)
**Cited paper PDFs read:** `liao2024_iv_rl.md` (Liao 2024, IV-aided value iteration, source of the IV-CMDP formulation); `bennett2021proximal.md` (Bennett & Kallus, proximal RL, source of the bridge-function approach); `1338_Causal_Reinforcement_Lear.md` (Deng et al. causal RL survey, contextual). Not present in `papers/`: Zhang & Bareinboim 2016, Namkoong 2020, Kallus–Zhou 2020 (which the task brief suggested checking). The script's stated target task is point identification under unmeasured confounding, not sensitivity bounds, so Kallus–Zhou/Namkoong are tangential here; their absence from `papers/` is not a correctness problem for the audited code, only for the chapter's coverage.

## 1. Algorithm Identity

Five estimators implemented (plus oracle). Verifying each against the corresponding equation in the tex:

- **Naive (`naive_ope`).** Empirical $\widehat{P}(s'\mid s,a)$ from counts, plug into Bellman with target policy $\pi(\text{promote}\mid s)=1$. This is exactly the biased baseline the chapter calls out in Lemma "naive_bias". Correct.
- **Backdoor (`backdoor_ope`).** Implements $\widehat{P}(s'\mid s, \text{do}(a)) = \sum_z \widehat{P}(s'\mid s,a,z)\,\widehat{P}(z\mid s)$ from counts. Matches Equation (backdoor_estimator) and Theorem (backdoor_mdp) in the tex. Correct — but note this is the *non-augmented* backdoor estimator, not the doubly-robust variant the prose advertises ("The doubly robust variant combines the fitted action-value function with backdoor-adjusted propensities…", line 138). The DR variant is mentioned but not implemented. Reviewer would call this out as a minor mismatch between prose and code.
- **Front-door (`frontdoor_ope`).** Implements $\widehat{P}(s'\mid s, \text{do}(a)) = \sum_m \widehat{P}(m\mid a)\sum_{a'} \widehat{P}(s'\mid s,m,a')\,\widehat{P}(a'\mid s)$, matching Equation (frontdoor). Correct. The implementation marginalises over $a'$ using $\widehat{P}(a'\mid s)$ from the logged data (not $\widehat{P}(a'\mid s, m)$), which is the textbook Pearl form when $A\to S'$ goes through $M$ only.
- **IV / Wald (`iv_ope`).** Implements the per-state Wald estimator $\hat\beta = (\widehat{P}(s+1\mid s,IV{=}1) - \widehat{P}(s+1\mid s,IV{=}0)) / (\widehat{P}(A{=}0\mid s, IV{=}1) - \widehat{P}(A{=}0\mid s, IV{=}0))$, then $\widehat{P}(s{+}1\mid s, \text{do}(a{=}0)) = \widehat{P}(s{+}1\mid s, IV{=}0) + \hat\beta \cdot (1 - \widehat{P}(A{=}0\mid s, IV{=}0))$. This matches Equation (wald) and the recovery formula in the tex (line 210). It is a *binary-action linearised* shortcut, not the full Liao et al. (2024) IV-aided value iteration (which solves a conditional moment restriction via a primal-dual reformulation; see Liao §3, "Algorithm 1"). The simplification is honest — the tex labels it "Wald estimator", not "IVVI" — but a hostile reviewer will flag the gap between the cited algorithm and what runs. Weak-instrument fallback (`|first_stage| < 0.01` → naive average) is reasonable and prevents division blowups.
- **Proximal (`proximal_ope`).** Solves a 2×2 linear bridge equation $\widehat{P}(s{+}1\mid w_1, s, a) = \sum_{w_2}\widehat{P}(w_2\mid w_1,s,a)\,h(w_2,s,a)$ then averages $h$ against the marginal $\widehat{P}(W_2\mid S{=}s)$. Matches the textbook discrete bridge-function form (Bennett & Kallus Assumption 2 for binary proxies). The choice of marginal-over-$S$ (not conditioned on $A$, see line 537 comment "Use marginal P(W2|S), NOT P(W2|S,A)") is defended in code — a careful reader who knows the bridge formula will recognise this is the right move because conditioning on $A$ would reintroduce the confounded path. Fallbacks for small cells (`n < 5`) and singular $A$-matrix (`|det| < 1e-10`) are reasonable.

Overall: each estimator is what it claims to be, with two prose-vs-code gaps (DR backdoor mentioned but not run; full IVVI not implemented, replaced by a binary-action Wald simplification). The chapter is honest about the IV simplification but says "doubly robust" in the prose without backing it.

## 2. Environment / MDP Fidelity

DGP in code (lines 53–58, 182–246) versus tex (lines 265–274):

- $N=5$ states with state 4 absorbing — matches "engagement funnel" prose.
- $Z\sim\text{Bernoulli}(0.5)$, $U\mid Z{=}1\sim\text{Bernoulli}(0.9)$, $U\mid Z{=}0\sim\text{Bernoulli}(0.1)$ — matches the tex's footnote.
- Behavioral policy $\mu(\text{promote}\mid s, U, IV) = 0.55 + \rho \cdot 0.25 \cdot (2U-1) + 0.15 \cdot (IV-0.5)$, clipped to $[0.01, 0.99]$ — matches Equation in §3.7 word-for-word.
- $M\mid A{=}\text{promote}\sim\text{Bernoulli}(0.8)$, $M\mid A{=}\text{hold}\sim\text{Bernoulli}(0.2)$ — matches footnote.
- $W^{(1)}, W^{(2)}$ proxy probabilities $(0.85, 0.15)$ and $(0.75, 0.25)$ — match footnote.
- Transitions $P(s{+}1\mid s, M, Z)$: $(M{=}1, Z{=}1)\!=\!0.9$, $(M{=}1, Z{=}0)\!=\!0.5$, $(M{=}0, Z{=}1)\!=\!0.4$, $(M{=}0, Z{=}0)\!=\!0.15$ — these are in the code (`P_TRANS`) but NOT in the tex (only the marginalised interventional probability 0.615 is reported). A reviewer would want the four transition values stated or tabulated in the tex; right now they exist only in code. Minor.
- Rewards: $-1$ at $s<4$, $0$ at $s=4$, $\gamma=0.9$ — match the tex footnote.
- Front-door / backdoor / IV conditions: the code makes $S'$ depend on $(M, Z)$ only and never on $A$ or $U$ directly. This satisfies the front-door assumption ($A\to S'$ goes through $M$), the backdoor assumption ($Z$ blocks $A\leftarrow U\to S'$ since $U$ only affects $S'$ via... actually, $U$ does NOT affect $S'$ at all in this DGP — it only affects $A$). The IV satisfies exclusion ($IV$ enters $\mu$ only, not transitions). All four identification strategies are simultaneously valid by construction. The tex acknowledges this is intentional (line 274). Honest.

The "confounding" in this DGP is unusual: $U$ confounds $A$ but $U$ has *no direct path to $S'$*. Confounding bias arises because $U$ correlates with $Z$ (and $Z$ does affect transitions), so $U$ acts as a backdoor variable through $Z$. This is the path $A \leftarrow U \leftarrow Z \to S'$ (and the un-blocked-in-naive path $A \leftarrow U \to Z \to S'$ is reversed; really it's $A \leftarrow U$, $U \leftarrow Z$, $Z \to S'$, which gives $A \leftarrow U \leftarrow Z \to S'$ — a backdoor path through $U$ and $Z$). The tex's DAG description (line 638–642 in script's stdout) is consistent. Confounding strength $\rho$ scales how much $U$ shifts $\mu$, so $\rho=0$ should give zero bias for all estimators (verified in stdout via `summary[0.0]`). Reviewer-careful, but consistent.

## 3. Data Integrity

`compute_data()` actually runs the DGP per-seed, fits each estimator from the trajectories, and stores per-seed values. The cache layer (lines 608–611) loads pre-computed results when present, which is why the stdout shows "Loaded from cache". The bias/SE/RMSE numbers in the LaTeX table come from `summary[rho][name]` built from the per-seed `results` arrays (lines 715–727). No hardcoded values masquerading as results. Numbers shown in the figure are consistent with what's in stdout (e.g. naive bias ≈ +0.32 at $\rho=1$, panel (a); backdoor/front-door bias ≈ 0 at all $\rho$). One caveat: with `--plots-only`, the script asserts the cache exists, otherwise computes — so if config drifts and the cache invalidates, the user will be forced to recompute, which is correct. No data-integrity red flag.

## 4. Comparison Fairness

Per (rho, seed), `generate_trajectories(rng, rho)` is called once and feeds *all five* estimators with the same trajectories (lines 678–706). Same $N\!=\!2000$ trajectories, same $T_\text{max}\!=\!50$, same seeds (`BASE_SEED + seed_idx` from 42 to 61). Same target policy (always promote) and same Bellman solve for all model-based estimators. Apples-to-apples on this front.

One subtlety the reviewer might flag: the IV-strength panel (panel c) refreshes the random generator with the *same seeds* used for the main loop (`BASE_SEED + seed_idx`, lines 800–804), but with a different `iv_coeff` argument that changes the behavioral policy. So the trajectories at $\rho=1$, $iv\_coeff=0.15$ in panel c are *not* identical to those used for the main loop's IV column at $\rho=1$ (which uses the default `MU_IV_COEFF=0.15`). They should be — the seeds match — and indeed the default `iv_coeff` is also `0.15`, so panel c at the middle setting *is* the same data as the main IV column at $\rho=1$. The other two settings (0.05, 0.30) get different trajectories, as intended. Fair.

## 5. Theoretical Sanity Checks

- At $\rho=0$ (no confounding), all five estimators should be ~unbiased. The figure shows them all near zero at $\rho=0$. The stdout summary at $\rho=0$ shows naive, backdoor, front-door, IV, proximal all with bias $|b|\lesssim 0.02$ (read off panel a). Pass.
- Naive bias growth with $\rho$: positive and roughly linear in $\rho$, reaching ≈ +0.32 at $\rho=1$. Sign: positive. Direction is consistent — confounding makes promote more likely when $U=1$, which correlates with $Z=1$ (favorable conditions), so observational transitions overestimate `P(advance | promote)` and value-estimation overstates $V^\pi$. Bias propagates through the Bellman recursion. Tex prose (line 278) gives the same explanation. Pass.
- Backdoor and front-door: zero bias at all $\rho$. Theorem (backdoor_mdp) and Equation (frontdoor) both predict consistency given the satisfied criteria. Pass.
- IV: low bias, higher variance. Panel (c) shows variance shrinking with instrument strength, consistent with classic weak-instrument theory. The 0.05 instrument shows a few extreme outliers (one near $-5.0$, one near $-2.6$). This is the Wald ratio's pathology under weak first stage — exactly what theory predicts. Pass with caveat: the weak-instrument fallback (`abs(first_stage) > 0.01`) is fairly loose and lets in noisy ratios. A stricter threshold would mute the outliers and arguably present a fairer picture, but the current behavior is theoretically *correct* (and dramatic outliers actually make the point about IV variance more visceral). Pass.
- Proximal: low bias, moderate variance. Panel (b) shows proximal RMSE around 0.12–0.14, well above backdoor/front-door (~0.03) but well below naive at high $\rho$. This is what proximal RL theory predicts — semiparametric efficiency under weak proxies is suboptimal vs an actually-observed backdoor variable. Pass.

Reviewer-2 questions that arise:
- Why does proximal RMSE *decrease* slightly with $\rho$ (panel b, around $\rho=0.2$ to $\rho=0.4$)? At higher $\rho$, the link $A\leftarrow U$ is stronger, so the proxies $W$ have more signal about $U$ via $A$'s observational distribution. Plausible but not articulated in the tex.
- Front-door's success here is mechanical: the DGP literally has $A\to M\to S'$ with no $A\to S'$ direct edge. A skeptical reader would call this engineered. The tex acknowledges the engineering ("This enables all four identification strategies simultaneously", line 274), so this is not a "gotcha" — but the reader should understand front-door is being awarded an exact-fit scenario.

## 6. Information Leakage

The oracle (`compute_oracle_value`) uses the true parameters `P_TRANS`, `P_M1_GIVEN_A0`, `P_Z1`, `STEP_COST`, `GOAL_REWARD`, `GAMMA`. That is the oracle's prerogative.

The five sample-based estimators take only `data` (the flattened trajectory arrays: $s, a, z, iv, m, w_1, w_2, s', r$). They never see $U$. They never use `P_TRANS`, `P_M1_GIVEN_A0`, or any DGP parameter. Verified by reading each estimator function — they all build counts from `data` and invert. The Bellman solve uses the chapter's known reward function ($-1$ step cost, terminal $0$), which is the standard OPE setup where the reward function is known and the transition is what's being estimated. No leakage on transitions, which is what the chapter is about.

Population-level verification (`print_population_verification`) uses true DGP parameters, but its outputs go to stdout only; they do not feed any estimator. Clean.

## 7. Seed & Reproducibility

- Seeds fixed: `BASE_SEED = 42`, per-seed seed = `BASE_SEED + seed_idx`. Reproducible.
- $N = 20$ seeds, ≥10 threshold met.
- Mean and SE reported in the LaTeX table (Bias ± SE) and in stdout summary. Pass.
- One minor: the IV-strength panel uses 20 seeds at fixed $\rho=1$; same seed base. Reproducible.
- Cache key includes the full `CONFIG` dict (lines 93–120) with a `version: 1` field, so config drift invalidates cleanly. Pass.

## Hostile-Reviewer Summary

The DGP is artificial in a way that lets every identification strategy work simultaneously — that is by design and the tex says so. Within that engineered setting, the five estimators are correctly implemented per the equations they cite, the comparison is fair (one dataset, all five estimators, fixed seeds), the oracle does not leak into the sample estimators, and the bias/variance patterns match theory. Two prose-vs-code mismatches: (i) the chapter mentions a doubly-robust backdoor variant that is not in the code, and (ii) the chapter cites Liao 2024's IV-aided value iteration but the code runs a simpler binary-action Wald estimator (this is flagged honestly in the prose, just not always in cross-references). The transition table $P(s{+}1\mid s, M, Z)$ lives in the code but not the tex, which a reader trying to reproduce the simulation from the chapter alone would notice. Outlier behavior in panel (c) at IV coefficient 0.05 is theoretically correct (weak-instrument pathology) but visually startling and would benefit from a one-line explanation in the caption.

**Bullshit score: 20%** — Reviewer 2 catches the DR-backdoor-mentioned-but-not-implemented mismatch and the missing transition-table in the prose, writes a snarky comment about Wald-vs-IVVI naming, and complains that "every estimator unbiased by construction" is a tautological demo. The substance — correct implementations, fair comparison, no leakage, sensible numbers — survives revision. Rounded up from ~15% because two prose-vs-code mismatches are present and a reproducibility-from-prose-only reader would have to read the code to recover all DGP details. Below the 25% bar but not by much.

**Path:** `/Users/pranjal/Code/rl/audits/ch10_causal__confounded_ope_2026-05-19.md`
