# Audit: ch10_causal/sims/identification_dags.py

**Date:** 2026-05-19
**Diagram-only:** yes (cap 25%)
**Cited tex file(s):** `ch10_causal/tex/causal_rl.tex` (Figure 4 / `fig:identification_dags` at L149–154; Figure 5 / `fig:simulation_dag` at L267–271; surrounding prose at L143–227 and L259–266)
**Cited paper PDFs read:** `ch10_causal/papers/bennett2021proximal.pdf` (referenced for proximal), `ch10_causal/papers/liao2024_iv_rl.pdf` (referenced for IV). Pearl 2009 (`pearl2009causality`) is cited for front-door — not in `papers/`, but the criterion is textbook canonical.

## 1. Algorithm Identity (does each DAG depict what its label claims?)

The script renders three small DAGs (`identification_dags.png`) and one larger DGP DAG (`simulation_dag.png`). Verified panel-by-panel:

**(a) Front-door.** Nodes: $U_t$ (unobserved, top), $A_t$ (left), $S_{t+1}$ (right), $M_t$ (bottom). Edges drawn: $U \dashrightarrow A$, $U \dashrightarrow S'$, $A \to M$, $M \to S'$. No direct $A \to S'$. This is exactly Pearl's front-door criterion: the mediator $M$ intercepts every directed path from $A$ to the outcome, while $U$ confounds $A$–$S'$ from outside the $A \to M \to S'$ chain. Matches the chapter's Equation~\ref{eq:frontdoor} at L221.

**(b) Instrumental variables.** Nodes: $Z_t$ (left, observed), $U_t$ (top, unobserved), $A_t$ (centre), $S_{t+1}$ (right). Edges: $Z \to A$, $U \dashrightarrow A$, $U \dashrightarrow S'$, $A \to S'$. No $Z \to S'$ direct edge (exclusion restriction), no $Z \to U$ (independence). This is the canonical IV DAG. Consistent with the tex at L197–204 (relevance $Z \to A$, exclusion $Z \not\to S'$ except via $A$).

**(c) Proximal causal inference.** Nodes: $U_t$ (top, unobserved), $A_t$ (left), $S_{t+1}$ (right), $W_t^{(1)}$ (bottom-left), $W_t^{(2)}$ (bottom-right). Edges: $U \dashrightarrow A$, $U \dashrightarrow S'$, $U \dashrightarrow W^{(1)}$, $U \dashrightarrow W^{(2)}$, $A \to S'$. The proxies are pure children of $U$ with no other parents. This matches the conditional-independence requirement in the proximal causal inference literature ($W^{(1)} \perp\!\!\!\perp W^{(2)} \mid U$, neither proxy directly affects the other or the outcome). Consistent with Bennett et al. 2021 as described at L183–193. Minor stylistic note: many proximal papers split the proxies into "treatment-side" $W^{(1)}$ (a parent of $A$ or pre-treatment) versus "outcome-side" $W^{(2)}$ (post-treatment), and some renderings show $A \to W^{(2)}$. The simulation here uses the simplest "both proxies are pure children of $U$" formulation, which is what `confounded_ope.py` actually samples (L230–232 of that script: $W^{(1)}, W^{(2)} \sim \text{Bernoulli}(p \mid U)$, no $A$ dependence). So the DAG matches the simulator. Not a misrepresentation.

**Caption alignment.** Caption at L152 reads: "(a) Front-door criterion with mediator $M_t$. (b) Instrumental variables with exogenous instrument $Z_t$. (c) Proximal causal inference with proxies $W_t^{(1)}, W_t^{(2)}$." Matches diagram labels exactly.

Backdoor is not drawn (the chapter intros backdoor earlier in §subsec:alternative_identification and the figure caption explicitly says "three point-identification strategies", so this is intentional — backdoor uses $Z_t$ alone, depicted in the simulation DAG by the $Z \to S'$ edge).

## 2. Environment / MDP Fidelity (simulation DAG vs SCM in confounded_ope.py)

Cross-checked `simulation_dag.png` against the DGP in `ch10_causal/sims/confounded_ope.py:generate_trajectories` (L182–246) and config L33–50.

Required edges per SCM:

| Edge | In simulation DAG? | In `confounded_ope.py`? |
|------|--------------------|--------------------------|
| $Z \to U$ | yes (dashed, $Z \to U$) | yes — `p_u1 = P_U1_GIVEN_Z1 if z==1 else P_U1_GIVEN_Z0` (L216) |
| $Z \to S'$ | yes (curved $Z \to S'$ edge) | yes — `P_trans` indexed by $(s, m, z)$ (L54–60, L240) |
| $U \to A$ | yes (dashed $U \to A$) | yes — `mu_promote = MU_BASE + rho*MU_DELTA*(2u - 1) + ...` (L221) |
| $U \to W^{(1)}$ | yes (dashed) | yes — L230 |
| $U \to W^{(2)}$ | yes (dashed) | yes — L232 |
| $\text{IV} \to A$ | yes (solid $\text{IV} \to A$) | yes — `mu_promote += _iv_coeff*(iv - 0.5)` (L221) |
| $S_t \to A_t$ | yes (solid) | weakly — behavioural policy uses state via clipping/absorbing logic, but `mu_promote` formula does not depend on $s$ explicitly. The state still gates "alive" trajectories (absorbing at $s=4$), so the edge is defensible but conservative. Not a misrepresentation. |
| $S_t \to S_{t+1}$ | yes (curved $S_t \to S'$) | yes — `P_trans[(s, m, z)]` indexed by current state |
| $A \to M$ | yes | yes — `p_m1 = P_M1_GIVEN_A0 if a==0 else P_M1_GIVEN_A1` (L226–227) |
| $M \to S'$ | yes | yes — `P_trans` indexed by $m$ (L54–60) |

Edges NOT in the DAG but also NOT in the simulator: $A \to S'$ directly (correct — front-door requires that $A$ affect $S'$ only via $M$; in the simulator, transitions are $P(s'|s, m, z)$ with no direct $a$ argument, see `compute_true_transition_*` at L127–150).

Two minor observations, not errors:

1. The DAG shows $Z \to U$ as **dashed** (because $U$ is unobserved), but $Z$ itself is observed. The convention used in this script is "dashed = edge touches an unobserved variable", which is unconventional — some readers expect "dashed = unobserved edge" only if both endpoints are latent. Caption clarifies "dashed edges involve unobserved variables", so it's internally consistent. Hostile-reviewer flag, not a bug.

2. The simulation DAG does not draw an $\text{IV} \to S'$ exclusion ban explicitly (it's the absence of an edge that conveys exclusion). This is standard DAG-drawing practice and matches the IV panel in `identification_dags.png`.

## 3. Data Integrity

N/A — diagram-only; no Monte Carlo numbers are produced. `identification_dags_stdout.txt` exists (the script prints only "Saved: ..." paths).

## 4. Comparison Fairness

N/A — no comparison.

## 5. Theoretical Sanity Checks

N/A — diagram-only. The DAG structure itself is the theoretical claim, and it matches Pearl 2009 (front-door, IV) and Tchetgen-Tchetgen / Bennett (proximal) canonical forms.

## 6. No Information Leakage

N/A.

## 7. Seed / Reproducibility

N/A — no randomness in the figure-generation code. Re-running yields identical PNGs.

## Hostile-Reviewer Summary

The diagrams are clean, accurately labelled, and structurally faithful to (i) Pearl's textbook DAGs for front-door / IV / proximal and (ii) the actual sampling code in `confounded_ope.py`. The simulation DAG correctly enumerates every $U, Z, \text{IV}, A, M, W^{(1)}, W^{(2)}, S, S'$ edge that the simulator instantiates. Minor stylistic nits (dashed-edge convention is unconventional; $S_t \to A_t$ edge is drawn even though the policy formula doesn't read $s$ explicitly) do not rise to a substantive error. No claim in the caption contradicts the picture.

**Bullshit score: 5%** — Under the 25% diagram-only cap. A hostile reviewer might quibble over the dashed-edge convention or the $S_t \to A_t$ edge in the DGP DAG, but neither overturns any reported result and the panels match Pearl/Bennett DAGs term-for-term.
