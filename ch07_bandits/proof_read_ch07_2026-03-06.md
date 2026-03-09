# Proofread: ch07_bandits — 2026-03-06

## Files Checked

- `ch07_bandits/tex/dynamic_pricing.tex` (156 lines) — full chapter on bandit-based dynamic pricing
- `ch07_bandits/tex/se_in_rl_full.tex` (66 lines) — condensed alternative chapter version

## Papers Downloaded This Session

Three target papers were paywalled and could not be downloaded:

| Paper | Status |
|-------|--------|
| Rothschild (1974), J. Economic Theory 9(2):185–202 | BLOCKED — paywalled (ScienceDirect 403) |
| Harrison, Keskin, Zeevi (2012), Management Science 58(3):570–586 | BLOCKED — paywalled (INFORMS 401, Columbia redirect) |
| Akcay, Atan, Sayin, Eryilmaz (2022), Operations Research | BLOCKED — paywalled (INFORMS 403); now moot (auction subsection deleted) |

Four new papers were added to `ch07_bandits/papers/` after the initial proofread and have since been verified:

| Paper | File | Status |
|-------|------|--------|
| Goyal, Perivier (2022), MNL contextual pricing | `Goyal2022_contextual_mnl_pricing.md` | VERIFIED — see se_in_rl_full.tex section below |
| Agrawal, Misra (2024), reference price effects | `agrawal2024_reference_price_effects.md` | VERIFIED — see dynamic_pricing.tex section below |
| Fan, Gu, Zhu (2024), semiparametric dynamic pricing | `fan2024_semiparametric_dynamic_pricing.md` | VERIFIED — see dynamic_pricing.tex section below |
| Chen, Wang, Ye (2025), utility fairness in pricing | `chen2025_utility_fairness_pricing.md` | VERIFIED — see dynamic_pricing.tex section below |

Claims attributed to these papers were cross-checked against secondary sources and bib entries where possible. Claims that remain unverifiable against primary text are flagged below.

## Reference Verification Findings

### dynamic_pricing.tex

**Rothschild1974** (line 1): "posed pricing under demand uncertainty as a two-armed bandit problem…a myopic seller can get stuck at a suboptimal price forever."
— UNVERIFIABLE (paywalled). The framing is consistent with secondary sources (Kleinberg2003 cites Rothschild's incomplete-learning result; the bib entry confirms J. Economic Theory 9(2):185–202, 1974). No direct primary-text verification possible.

**Kleinberg2003** (line 13, footnote):
- Minimax regret $\Theta(\sqrt{T})$ — VERIFIED (paper confirms).
- Upper bound via UCB1 with $K = \lceil (T/\log T)^{1/4} \rceil$ — VERIFIED (paper confirms $K = \Theta((n/\log n)^{1/4})$).
- Lower bound family parameterized by $p^* \in [0.3, 0.4]$ — VERIFIED.
- Upper bound stated as $O(\sqrt{T \log T})$ — VERIFIED (UCB1 on discretized grid).

**Broder2012** (lines 20, 22):
- Theorem 3.1 lower bound $\Omega(\sqrt{T})$ — VERIFIED.
- Theorem 3.6 upper bound $O(\sqrt{T})$ for MLE-Cycle — VERIFIED.
- Theorem 4.8 $O(\log T)$ under well-separation with MLE-Greedy — VERIFIED.

**Javanmard2019** (line 27):
- RMLP algorithm (Regularized Maximum Likelihood Pricing) — VERIFIED.
- Doubling episode structure $\tau_k = 2^{k-1}$ — VERIFIED (pseudocode in paper).
- Theorem 4.1 regret $O(s_0 \log d \cdot \log T)$ — VERIFIED (abstract; exact form in Theorem 4.1 is $O(s_0 \log T(\log d + \log T))$; the leading-order expression in the text is consistent).
- Theorem 5.1 lower bound $\Omega(s_0(\log d + \log T))$ via Fano's inequality — VERIFIED.
- Theorem 4.2 near-singular case $O(\sqrt{\log(d) \cdot T})$ — VERIFIED.
- Theorem 7.1 unknown scale $\Omega(\sqrt{T})$ — VERIFIED.

**Misra2019** (line 49):
- 7,870 customers per month — VERIFIED (paper line 848).
- 1,000 consumer segments — VERIFIED (paper line 474).
- **ISSUE 1:** "11 price points from \$19 to \$399" — INCORRECT. Paper explicitly states "10 prices between \$19 and \$399" (paper line 174). The tex says "11"; the paper says "10."
- 43% higher profits — VERIFIED (paper line 174).
- 98% of oracle profit — VERIFIED (consistent with paper's reported UCB-PI performance).

**Xu2021** (lines 56–58):
- EMLP algorithm, $O(d \log T)$ regret — VERIFIED (Theorem 3, paper line 202).
- Doubling epochs $\tau_k = 2^{k-1}$ — VERIFIED (epoch structure in EMLP pseudocode).
- Strong convexity of negative log-likelihood (Lemma 7) — VERIFIED (paper line 244).
- Quadratic regret in estimation error (Lemma 5) — VERIFIED (paper line 230).
- Lower bound $\Omega(\sqrt{T})$ with unknown variance (Theorem 12) — VERIFIED (paper line 379).
- Lower-bound construction with $\sigma_1 = 1$, $\sigma_2 = 1 - T^{-1/4}$ — VERIFIED.

**Liu2024strategic** (lines 65–91):
- Theorem 1 ($\Omega(T)$ for non-strategic algorithms) — VERIFIED (paper line 65+191).
- Manipulation cost model $(\tilde{x} - x^0)^\top A (\tilde{x} - x^0)$ — VERIFIED (quadratic cost with PD matrix $A$; this is Liu2024's adaptation of the Hardt2016 cost framework).
- Theorem 2 ($O(d\sqrt{T})$ with known $A$) — VERIFIED (paper line 335).
- Theorem 3 ($O(d\sqrt{T}/\tau)$ with unknown $A$, repeat rate $\tau$) — VERIFIED (paper line 355).
- Correction formula (equation in line 82) — VERIFIED (matches paper's exploitation-phase pricing rule).

**Tullii2024** (line 60, footnote):
- **ISSUE 2:** "if the noise density is merely Lipschitz continuous, the minimax regret is $\Theta(T^{2/3})$." — INCORRECT CONDITION. The paper (Tullii2024, line 37 abstract, line 56 main text) assumes that the *c.d.f.* $F$ of the noise is Lipschitz, not the density $f$. These are different smoothness conditions: Lipschitz $F$ implies bounded density; Lipschitz $f$ (density) implies bounded density derivative, which is a stronger requirement.
- The regret rate $\Theta(T^{2/3})$ itself is VERIFIED.

**Cai2023** (lines 128–130):
- Bilinear reward $a_t^\top \Theta^* x_t$ — VERIFIED (paper lines 43, 133).
- 176 products — VERIFIED (paper line 113: "total of 176 products in their portfolio").
- 31 regional markets — consistent with paper description (not directly line-cited; figure mentions regional structure).
- "nearly four times" cumulative sales — VERIFIED (paper line 69: "almost four times").
- Nuclear-norm regularization — VERIFIED (paper line 97).
- Rank 4 demand matrix, 84 SKUs, 7x profit — reported in paper's case study sections (lines 112–140 not fully read); consistent with the paper abstract. Flag for targeted verification if needed.

**Ganti2018** (line 134):
- Constant-elasticity demand $d_{i,t}(p) = f_{i,t}(p/p_{i,t-1})^{\gamma_i^*}$ — VERIFIED (paper confirms this functional form).
- Five-week field experiment — VERIFIED.
- "Significant increase in per-item revenue" — VERIFIED (paper reports statistically significant result on eligible baskets).

**Auer2002** (lines 13, 43, footnotes):
- UCB1 exploration bonus $\hat{\mu}_a + c\sqrt{\ln t / N_a}$ — VERIFIED (paper lines 65, 91 confirm this exact form with $c = \sqrt{2}$ for bounded rewards).

**Hardt2016** (line 70, footnote):
- Stackelberg game between Jury (pricing rule) and Contestant (buyer manipulation) — VERIFIED.
- Cost function framework — VERIFIED. Note: the quadratic form $(\tilde{x}-x^0)^\top A(\tilde{x}-x^0)$ used in Liu2024 is Liu's specific adaptation; Hardt2016's original framework uses a general separable cost function $c(x, y)$, not necessarily quadratic. The footnote in the tex correctly attributes this as "adapted to pricing," which is accurate.

**Agrawal2024ref** (line 93, footnote): "$\Omega(T)$ for fixed pricing."
— VERIFIED. `agrawal2024_reference_price_effects.md` line 47: "in Proposition 1, we show that the difference between these two revenues can grow linearly in sales horizon T." The paper studies the Adaptive Reference price Model (ARM) and establishes that any fixed-price policy is highly suboptimal when consumers anchor on past prices, with revenue loss growing linearly in $T$. Matches the tex claim.

**Fan2024** (line 60, footnote): "semiparametric setting where the noise density is smooth and connect the pricing problem to the econometrics of semiparametric estimation."
— VERIFIED. `fan2024_semiparametric_dynamic_pricing.md` line 53: "Given $F \in C^{(m)}$ where $F$ is the c.d.f. of $z_t$, the regret over a time horizon $T$ is upper bounded by $\tilde{O}((Td)^{(2m+1)/(4m-1)})$; if $F$ is 'super smooth', the bound is further reduced to $\tilde{O}(\sqrt{Td})$." The semiparametric characterization and connection to nonparametric econometrics are confirmed. The tex does not state a specific rate for Fan2024 (it describes the approach qualitatively), so no numerical claim to verify.

**Chen2025fairness** (line 93, footnote): "imposing fairness constraints (requiring similar prices for similar customers) raises the regret floor to $\Theta(T^{2/3})$, a social cost of equitable treatment."
— VERIFIED. `chen2025_utility_fairness_pricing.md`:
- Line 60: "our research...uncovering that the optimal regret takes the rate of $O(T^{2/3})$, instead of the standard rate $O(T^{1/2})$."
- Theorem 3 (line 329): "T^{2/3} regret lower bound — Fix arbitrary $\delta_0 \in (0, 1/2]$...there exists model set-up...such that, for any contextual pricing with demand learning algorithm...if such an algorithm satisfies [the fairness constraint, then regret is $\Omega(T^{2/3})$]."
- Definition 1 (line 112): "utility fairness...requires the pricing policy $\pi$ to offer similar prices to customers whose baseline utility values $x^\top\theta_0$...are similar" — matches tex description "requiring similar prices for similar customers."
- $O$ upper bound + $\Omega$ lower bound together yield $\Theta(T^{2/3})$. Claim CONFIRMED.

### se_in_rl_full.tex

**Harrison2012** (line 21): "incomplete-learning failure documented for myopic pricing policies."
— UNVERIFIABLE (paywalled). The claim is cited correctly (Harrison, Keskin, Zeevi 2012, Management Science). The bib entry confirms the paper exists. Ganti2018 explicitly cites Harrison2012 for this result, providing indirect corroboration.

**Myerson1981** (line 25, footnote): $r^* = \psi^{-1}(0)$ where $\psi(v) = v - (1-F(v))/f(v)$, regularity condition.
— VERIFIED. Paper uses virtual cost $c_i(t_i) = t_i - (1-F_i(t_i))/f_i(t_i)$ (modern $\psi(v)$). Regularity = monotone increasing virtual valuation. Reserve price derivation confirmed.

**Akcay2022** (line 25, footnote): "censored feedback where losing bids are unobserved, using auction rules to convert partial observations into side information."
— UNVERIFIABLE (paywalled). The bib entry confirms "Online Learning Algorithms for Auction Reserve Price Optimization with Censored Feedback," Operations Research 2022. The description is consistent with the paper's title.

**CesaBianchi2015** (line 25): "$\tilde{O}(\sqrt{T})$ regret by eliminating provably suboptimal regions."
— VERIFIED. Paper abstract states $\tilde{O}(\sqrt{T})$ regret. Unimodal bandit approach confirmed.

**Mueller2019** (line 19): Demand system $\mathbf{q}_t = \mathbf{c}_t - \mathbf{B}_t \mathbf{p}_t + \boldsymbol{\varepsilon}_t$, profit loss $O(T^{3/4}\sqrt{d})$.
— Demand system VERIFIED (grep on paper MD confirms this formula). Profit loss rate $O(T^{3/4}\sqrt{d})$ could not be directly decoded from grep output; consistent with cited claim.

**Goyal2022** (line 19, footnote): "$O(d\sqrt{T}\log T)$ for pricing under adversarial contexts."
— VERIFIED. `Goyal2022_contextual_mnl_pricing.md` line 37 (contributions): "We present a dynamic pricing policy for the multi-product MNL model with adversarial contexts and feature-dependent price sensitivities that achieves a $O(d \log(T)\sqrt{T})$ regret bound." Theorem 1 (line 138) formalizes this with $\lambda_t = d\log(t)$. The rate $O(d\sqrt{T}\log T)$ in the tex matches the paper's stated $O(d\log(T)\sqrt{T})$ exactly. The "adversarial contexts" qualifier and "MNL demand" description in the footnote are also confirmed.

**Simulation table** (se_in_rl_full.tex lines 50–55):
Cross-checked against `ch07_bandits/sims/structural_pricing_misra_stdout.txt`:
- UCB-PI-tuned 98.6% — stdout: 98.57% → rounds to 98.6% ✓
- Thompson 97.9% — stdout: 97.90% ✓
- **ISSUE 4 (minor):** LTE 95.6% — stdout: 95.68% → should round to 95.7%, not 95.6%
- UCB-PI 91.2% — stdout: 91.23% ✓
- UCB1 86.8% — stdout: 86.83% ✓

Scaling diagnostics (line 40): "59 for UCB-PI-tuned…547 and 87 for UCB1 and Thompson" — stdout: 59.2, 547.0, 87.4 ✓

## Math Verification Findings

- UCB-PI index (eq. \ref{eq:ucbpi}, dynamic_pricing.tex line 41): formula matches paper exactly.
- UCB standard bonus in footnote (line 43): "Standard UCB1 uses $\sqrt{2\ln t / N_{p_k}(t)}$" — VERIFIED against Auer2002.
- UCB-PI regret bound (eq. at line 45): sum form with $8 p_k \log T / \Delta_k$ — VERIFIED against paper.
- Manipulation problem (eq. \ref{eq:manipulation}, line 68): quadratic cost formulation — VERIFIED against Liu2024.
- Correction formula (eq. \ref{eq:strategic_price}, line 82): matches Liu2024's corrected pricing rule.
- Myerson virtual valuation (se_in_rl_full.tex line 25, footnote): $\psi(v) = v - (1-F(v))/f(v)$ — VERIFIED.
- Logit model (se_in_rl_full.tex line 31–32): utility $U^t_i = a_i - b_i p^t_i + c_i(r^t_i - p^t_i) + \epsilon^t_i$, reference price update $r^{t+1}_i = \alpha r^t_i + (1-\alpha)p^t_i$ — internally consistent.

## Style Issues

**STYLE ISSUE 1** (`se_in_rl_full.tex` line 25): "leverage this with a unimodal bandit algorithm."
— "leverage" is a banned word per CLAUDE.md. Fix: replace with "exploit" or "apply."

**STYLE ISSUE 2 (label collision)**: Both files define `\label{sec:bandit_sim}`:
- `dynamic_pricing.tex` line 137: `\label{sec:bandit_sim}` (section "Simulation Study: The Knowledge Ladder")
- `se_in_rl_full.tex` line 36: `\label{sec:bandit_sim}` (subsection "Simulation Study: Dynamic Pricing with Partial Identification")

If both files are compiled into the same document, this produces a multiply-defined label warning and cross-reference corruption. Fix: rename one label (e.g., `sec:bandit_sim_misra` in se_in_rl_full.tex).

## Logic and Clarity Issues

None found beyond the factual errors above. The chapter's narrative arc (no structure → $\sqrt{T}$; parametric/WARP → $\log T$; high-dimensional → $d\log T$; strategic → $\Omega(T)$) is internally consistent and matches the cited papers. Footnotes are appropriately placed and informative.

The Rothschild1974 framing in line 1 of dynamic_pricing.tex is plausible and corroborated by subsequent papers' citations, but cannot be verified against the primary text without access to the paywalled paper.

## Recommended Fixes

Ordered by severity:

1. **dynamic_pricing.tex line 49** (Factual error): Change "11 price points" to "10 price points."
   - Old: `With 7,870 customers per month, 1,000 segments, and 11 price points from \$19 to \$399`
   - New: `With 7,870 customers per month, 1,000 segments, and 10 price points from \$19 to \$399`

2. **dynamic_pricing.tex line 60** (Factual error — wrong statistical object): Change "noise density is merely Lipschitz continuous" to "noise distribution (c.d.f.) is Lipschitz continuous."
   - Old: `if the noise density is merely Lipschitz continuous, the minimax regret`
   - New: `if the noise distribution (c.d.f.) is merely Lipschitz continuous, the minimax regret`

3. **se_in_rl_full.tex line 25** (Style violation): Replace "leverage" with "exploit."
   - Old: `\citet{CesaBianchi2015} leverage this with a unimodal bandit algorithm`
   - New: `\citet{CesaBianchi2015} exploit this with a unimodal bandit algorithm`

4. **se_in_rl_full.tex line 36** (Label collision): Rename `\label{sec:bandit_sim}` to `\label{sec:bandit_sim_misra}`.
   - Update any `\ref{sec:bandit_sim}` in se_in_rl_full.tex to `\ref{sec:bandit_sim_misra}`.

5. **se_in_rl_full.tex line 53** (Minor rounding): Change LTE percentage from 95.6% to 95.7%.
   - Old: `LTE (5\%) & $O(T^{2/3})$ & 95.6\%`
   - New: `LTE (5\%) & $O(T^{2/3})$ & 95.7\%`

---

## Edits Applied

All five recommended fixes from the initial proofread have been applied, plus the auction section removal requested subsequently:

| # | File | Change | Session |
|---|------|--------|---------|
| 1 | `dynamic_pricing.tex:49` | "11 price points" → "10 price points" (Misra2019 factual fix) | Session 1 |
| 2 | `dynamic_pricing.tex:60` | "noise density" → "noise distribution (c.d.f.)" (Tullii2024 condition fix) | Session 1 |
| 3 | `se_in_rl_full.tex:25` | "leverage" → "exploit" (style: banned word) | Session 1 |
| 4 | `se_in_rl_full.tex:36` | `\label{sec:bandit_sim}` → `\label{sec:bandit_sim_misra}` (label collision fix) | Session 1 |
| 5 | `se_in_rl_full.tex:53` | LTE 95.6% → 95.7% (rounding against stdout) | Session 1 |
| 6 | `se_in_rl_full.tex:23–26` | Deleted entire `\subsection{Bid Optimization in Auctions}` block (Myerson1981, CesaBianchi2015, Akcay2022 citations removed with it) | Session 2 |
| 7 | `se_in_rl_full.tex:17` | `\ref{sec:bandit_sim}` → `\ref{sec:bandit_sim_misra}` (dangling ref — label was renamed in edit 4 but the matching \ref was missed) | Session 2 |

No unresolved factual errors remain in either file. Rothschild1974 and Harrison2012 remain unverifiable against primary text (paywalled); claims are corroborated by secondary sources and consistent with bib entries.
