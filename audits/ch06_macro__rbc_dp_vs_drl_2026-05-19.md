# Audit: ch06_macro/sims/rbc_dp_vs_drl.py

**Date:** 2026-05-19
**Diagram-only:** no
**Cited tex file(s):** `/Users/pranjal/Code/rl/ch06_macro/tex/macro_rl.tex` (sec `macro:simulation`, table `tab:macro:rbc-results`, figure `fig:macro:rbc-curves`, lines 332–392; also referenced from `macro:solver_single_agent` at lines 184–212)
**Cited paper PDFs read:** `papers/AS_IMF.pdf` (Atashbar & Shi 2023, IMF WP/23/40 — the explicit anchor paper for this sim); `papers/extracted/AS_IMF.md`. Schulman2017 (PPO) and Lillicrap2015 (DDPG) are **not** in `ch06_macro/papers/` — only `LillicrapDDPG2016` is in `docs/refs.bib`. No Kydland-Prescott1982 or KingPlosserRebelo1988 PDFs in the chapter's `papers/` directory.

## 1. Algorithm Identity

**VFI (lines 213–260).** Standard discrete-state Bellman iteration on a $400 \times 41$ tensor grid: $V_{k+1}(K,A) = \max_{K'} \{\log(W-K') + \beta \mathbb{E}_{A'|A}[V_k(K',A')]\}$ with Tauchen-discretised AR(1) for log A. Implementation is vectorised; argmax over $K'$ is done with `np.argmax`; convergence tolerance $10^{-5}$, max $800$ iterations. This is textbook VFI and is correct. Linear interpolation in $K$ at policy-extraction time is reasonable; nearest-neighbour in $A$ is a minor approximation but acceptable.

**KPR (lines 130–185).** Solves a quadratic in $\eta_k$ from a log-linearisation of the FOCs around the deterministic steady state, picks the saddle-path-stable root by the Blanchard-Kahn eigenvalue criterion, and derives $\eta_a$ analytically. Form $\tilde c_t = \eta_k \tilde k_t + \eta_a \tilde a_t$. This is a log-linear approximation, not the full King-Plosser-Rebelo (1988) solution method (KPR's contribution is balanced-growth-path consistency under permanent technology; here it's just first-order perturbation around the deterministic steady state). Calling it "KPR" is at best a loose label. Atashbar–Shi (the anchor reference) does not use this approximation, and the tex calls it a "Blanchard-Kahn log-linearisation (KPR)," which is a defensible if non-standard naming. Mild miscredit, not a methodological error.

**PPO (lines 324–423).** Clipped surrogate $\min(r_t A_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t)$ with $\epsilon = 0.2$, GAE($\lambda=0.95$) with $\gamma = \beta = 0.96$, separate actor (Gaussian, learned mean, **scalar** log_std parameter) and critic, $4$ epochs of minibatch SGD per update, advantage normalisation. The clip + GAE structure matches Schulman et al. (2017) exactly. Notable simplifications: entropy coefficient = 0 (the policy can collapse — no entropy floor), no learning-rate annealing, no value-function clipping (PPO2 in the original code does this; vanilla PPO does not require it). The action is unbounded $\mathbb R$ and squashed through a sigmoid to a savings fraction (line 297). The log-density for the *true* squashed action would need a Jacobian correction; the code uses `dist.log_prob(a)` on the pre-squash sample, which is consistent with itself (the policy effectively works in pre-squash space and the squash is part of the environment from the actor's view). Self-consistent but unusual.

**DDPG (lines 429–560).** Deterministic actor + Q-critic + soft Polyak-updated targets ($\tau = 0.005$), replay buffer (50k), Gaussian exploration noise ($\sigma = 0.1$) added to the actor output, $1000$-step uniform-random warmup. Critic loss: $(Q(s,a) - (r + \beta(1-d) Q_t(s', \mu_t(s'))))^2$. Actor loss: $-Q(s, \mu(s))$. Polyak average is correct. This matches Lillicrap et al. (2016) faithfully. The action is again unbounded $\mathbb R$ post-actor, squashed via sigmoid into a savings fraction.

**Verdict:** Algorithm identities are essentially correct. The "KPR" label is loose (it's standard Blanchard-Kahn log-linearisation, not the KPR balanced-growth construction) but the tex parenthetically clarifies this. PPO's missing entropy regularisation and DDPG's vanilla form are conventional simplifications.

## 2. Environment / MDP Fidelity

Parameters from script (lines 32–35): $\beta = 0.96$, $\alpha = 0.36$, $\delta = 0.10$, $\rho = 0.95$, $\sigma_\varepsilon = 0.007$, horizon $T = 200$. Tex (lines 343–344) reports identical values. Steady state computed in code: $K^* = ((1/\beta - 1 + \delta)/\alpha)^{1/(\alpha-1)} = 4.294$, $C^* = 1.260$. Tex reports $K^* = 4.29$, $C^* = 1.26$. Matches.

Transition: $K_{t+1} = (1-\delta) K_t + A_t K_t^\alpha - C_t$ in tex; code (line 110–117) computes $Y = A K^\alpha$, $W = Y + (1-\delta) K$, $K' = W - C$. Equivalent.

Productivity: $\log A_{t+1} = \rho \log A_t + \varepsilon_{t+1}$, $\varepsilon \sim N(0, \sigma^2)$. Tex matches code (line 115). Reward $r = \log C$, utility $u(C) = \log C$. Matches.

There is **no labour choice** (consistent with the basic RBC variant Atashbar–Shi 2023 uses; many RBC models include labour, but the writeup is clear that this is the inelastic-labour case). The state is $(K, A)$, action is $C$. Square brackets close.

**One discrepancy:** The tex (line 357) states evaluation initial conditions $(K_0, A_0) \sim \mathcal{U}[0.5, 8] \times \mathcal{U}[0.95, 1.05]$. The script matches at lines 42–44. Consistent. Note that $K_0$ spans roughly $0.12 K^*$ to $1.86 K^*$, which is wide; the agent has to handle a broad starting distribution. This is fine but worth noting — DDPG variance probably reflects difficulty over the wide $K_0$ support more than fundamental DDPG instability.

**Verdict:** Environment matches tex exactly. The MDP is a faithful textbook stochastic RBC with no labour-leisure choice.

## 3. Data Integrity

`compute_data()` (lines 747–760) calls `compute_or_load` for `shared`, `KPR`, `VFI`, `PPO`, `DDPG` components. Cache pickles exist (`cache/rbc_dp_vs_drl__{KPR,VFI,PPO,DDPG,shared}.pkl`). I did not unpickle but the script structure is correct: each compute_* function returns real numerics derived from training/iteration, and `generate_outputs` reads `mean_return`, `se_return`, learning curves, and policy values from those results.

Numbers in the reported table (KPR 45.88 ± 0.587, VFI 45.86 ± 0.589, PPO 45.82 ± 0.843, DDPG 45.17 ± 1.476; MSE 0.0004 / 0.0000 / 0.0087 / 0.0365) are computed inside `generate_outputs` from `data` arrays, not hardcoded. The MSE is computed against the VFI policy on 1000 random test states (line 785). VFI's self-MSE is 0.0 by construction (it's the reference) — correctly reported as 0.0000.

Stdout file matches the printed format and reports cache hits; numbers reproduce.

**One housekeeping bug:** Line 674, `'capital_traj_first': sol['policy_C']` is labelled as a capital trajectory but stores the VFI policy matrix. It's an unused placeholder field, not consumed downstream. Cosmetic, no impact on results.

**Verdict:** Numbers come from the actual computation. No hardcoding detected. One mislabelled cache field, harmless.

## 4. Comparison Fairness

Shared evaluation set (lines 624–630): 30 episodes, all four methods evaluated on the same `(K_0, A_0, \{\varepsilon_t\})` tuples, fixed at script start with `eval_rng = np.random.default_rng(99991)`. This is correct and the tex's claim that "cross-method differences are not driven by evaluation noise" (line 359) is supported by the code.

Training budgets are **not** matched between PPO and DDPG. PPO: $100 \text{ updates} \times 1024 \text{ steps} = 102{,}400$ environment steps per seed. DDPG: $60{,}000$ steps per seed. That's a ~1.7x advantage to PPO in environment samples. The wall-clock column in the table is empty for both RL methods ("$-$"), so the reader cannot judge compute parity. A hostile reviewer would catch this: "Why does PPO get 70% more samples than DDPG?" Defensible (PPO is on-policy, DDPG is off-policy with a replay buffer, so step-counts aren't apples-to-apples), but the asymmetry should be acknowledged in the writeup.

Same RBC environment (same params, same `RBCEnv` class). Same seed range $[0, 10)$ for both methods. Both RL methods squash through the same sigmoid action wrapper. Eval set is identical. Network architecture is identical ($64$-$64$ MLP). This is fair.

**Verdict:** Fair on environment, evaluation, network size, and seeds. Not fair on training-step budget, but the asymmetry is methodologically defensible — should be disclosed in the table footnote.

## 5. Theoretical Sanity Checks

VFI on a $400 \times 41$ grid for log-utility Cobb-Douglas Aiyagari-style RBC should be very close to the oracle. KPR (log-linear) being within $0.001$ policy MSE of VFI and $0.022$ in mean return is the expected agreement near the steady state — the log-linearised solution is exact at order zero in the perturbation around $K^*$ and the eval set has $K_0 \in [0.5, 8]$ around $K^* = 4.29$, so policies should agree closely. They do.

PPO mean return $45.82$ vs VFI $45.86$: gap $0.04$, within VFI's own evaluation SE ($0.589$). PPO MSE $0.0087$ on the policy: a typical 10–20% relative error compared to the consumption level near steady state. Reasonable for PPO on a smooth control problem.

DDPG mean return $45.17$ vs VFI $45.86$: gap $0.69$, just outside VFI's $1\sigma$ but well within DDPG's own SE of $1.476$. MSE $0.0365$, 4x PPO's MSE. Consistent with the known DDPG sensitivity to initialisation; the tex acknowledges this.

The chapter's anchor result (Atashbar & Shi 2023, IMF WP/23/40) reports that DDPG converges to a steady-state policy close to the analytical optimum in both deterministic and stochastic variants of the same basic stochastic RBC, and explicitly documents training-instability cases. Direction of the current sim matches the published finding. There is no theoretical bound that PPO should be better or worse than DDPG here; both should approach the oracle, which they do.

**One mild concern:** The "rule of thumb" that DRL converges to the analytical optimum is satisfied; the more interesting question is whether the gap of $0.04$ for PPO is the algorithm's residual or the eval-set noise. Given that VFI itself has SE 0.589 across the same 30 episodes (the SE here is across episodes, not across method-induced randomness — so it's the cross-initial-condition spread), this is not a sharp test. A defensible reviewer remark is that the SE column for VFI and KPR reflects evaluation-set spread, while PPO and DDPG's SEs *also* include training-seed noise. The table footer should note this asymmetry — the reader may misread these as comparable error bars.

**Verdict:** All methods approach the oracle. KPR/VFI gap is theoretically expected. PPO/DDPG gaps are within reported SE bands and consistent with the literature.

## 6. Information Leakage

PPO actor sees $(K, A)$ only (line 338, `obs_t = torch.tensor([K, A])`). DDPG actor sees $(K, A)$ only (line 484–485). Neither sees rewards directly, neither sees the closed-form policy, neither references the VFI solution during training. Both rebuild $W$ from $(K, A)$ via the `wealth` function which uses environment params, but wealth $W = A K^\alpha + (1-\delta) K$ is part of the agent's budget constraint and is known to the household in any RBC formulation — this is not leakage.

The sigmoid wrapper converts unbounded actor output to a savings fraction in $(0,1)$ (line 297), then $c = (1 - \text{sav}) \cdot W$. The wrapper uses $W$ (computable from observation) and is part of the action-encoding, not a policy hint.

`consumption_from_action` clips $c$ to $[10^{-4}, W - 10^{-4}]$. This is a feasibility safeguard, not leakage.

**Verdict:** No leakage. Both DRL methods are properly model-free given $(K, A)$ alone.

## 7. Seed & Reproducibility

Both DRL methods use $N\_SEEDS = 10$ (line 39) and report mean across seeds with cross-seed SE. KPR and VFI use 30 eval episodes with the same eval set and report cross-episode SE. Tex caption correctly distinguishes these two flavours of SE. Per-seed `np.random.default_rng(seed)` and `torch.manual_seed(seed)` are set inside each training function (lines 325–326, 457–458) — proper.

Eval-set RNG (`eval_rng = np.random.default_rng(99991)`) is fixed at line 624, so the eval distribution is reproducible. Test-state RNG (`np.random.default_rng(0)`) at line 619 is fixed.

Caching protects against config drift via the `compute_or_load` API; SHARED_CONFIG / PPO_CONFIG / DDPG_CONFIG are hashed into the cache key. Edit a hyperparameter and the corresponding cache invalidates.

**One asymmetry worth flagging:** the table reports SE for KPR/VFI as cross-episode standard error over $30$ eval episodes (homogeneous-evaluation noise), while PPO/DDPG SEs are reported as cross-seed standard error of per-seed mean returns over $10$ seeds. These are conceptually different objects but rendered with the same units in the same column. The tex caption (line 383) does distinguish them ("Standard error across 10 training seeds for PPO and DDPG, across 30 evaluation episodes for KPR and VFI"). Acceptable, but a hostile reviewer would ask for a unified bootstrap SE across the $30 \times 10$ tuples.

**Verdict:** Reproducible, seeded, $N \geq 10$. Mixed-SE convention in the same table is a minor presentation issue, not a methodological flaw — and the caption discloses it.

## Hostile-Reviewer Summary

The simulation is structurally sound. VFI/KPR/PPO/DDPG implementations are textbook and faithful to their defining equations. Environment matches the tex. Same evaluation set across methods, multiple seeds, fixed RNGs, hash-keyed caching. No information leakage. Results agree qualitatively and quantitatively with the chapter's anchor reference (Atashbar–Shi 2023). The chapter prose is appropriately understated — "It is not a demonstration that RL replaces dynamic programming; it is a check that RL does not fail it on the textbook case."

Reviewer-2 catches: (i) PPO gets 102k steps vs DDPG's 60k — sample-budget asymmetry undisclosed in the table; (ii) "KPR" is a loose label for what is really first-order Blanchard-Kahn log-linearisation around the deterministic steady state, not the KPR (1988) balanced-growth construction; (iii) the SE column mixes cross-seed and cross-episode notions of SE; (iv) PPO's missing entropy coefficient could be flagged as a non-standard ablation if held to a high bar; (v) `capital_traj_first` field is mislabelled in the cache (cosmetic). None of these change the conclusion. The wall-clock column has a real $13.5$ s for VFI and `$-$` for the others — adding wall-clock for the RL methods would make the comparison crisper and address the implicit "why aren't you reporting compute?" question.

**Bullshit score: 20%** — Reviewer 2 nicks the sample-budget asymmetry, the loose "KPR" label, and the mixed-SE convention, but the methodology is honest, the algorithms match their definitions, and the result reproduces a published finding on the same problem. Round up from 15% to 25% only if the reviewer also dings the missing PPO entropy term as a faithfulness issue; otherwise, score holds at 20%. The chapter's own conservative framing ("a check that RL does not fail it") would absorb most of the surviving criticism in revision.
