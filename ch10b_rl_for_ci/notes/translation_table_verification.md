# Translation Table Verification — §1 of `rl_for_ci.tex`

Source table: `/Users/pranjal/Code/rl/ch10b_rl_for_ci/tex/rl_for_ci.tex` lines 51-66
(Table `tab:rl_ci_dictionary`). The surrounding prose claims:

> "The translation is exact in the column-by-column sense. Each entry on the left,
> treated as a procedure on the observed-data distribution, returns the same
> numerical object as the entry on the right when applied to the same data."

This file goes row-by-row, quoting the primary sources verbatim, to decide
whether that claim is defensible as written. Notation in the comparisons uses
the chapter's $\bar H_k = (\bar S_k, \bar A_{k-1})$ convention.

Verdict legend: **EXACT** (literal equality of objects on identical data) /
**EXACT-UNDER-AUGMENTATION** (equal after state := full history, finite
horizon, $\gamma=1$) / **STRUCTURAL** (related but not the same object) /
**MISLEADING** (claim conflates distinct ideas).

---

## Row 1 — $Q_k(\bar h_k, a_k)$ $\leftrightarrow$ $Q(s,a)$

**Claim as written.**
LHS: $Q_k(\bar h_k, a_k)$, conditional mean of $V_{k+1}$.
RHS: $Q(s,a)$, state-action value function.

**DTR / g-method source.** Murphy (2003), eq. (1)-(2) of §2, reproduced in the
chapter as eq. (1)-(2). Direct quote, Murphy §2 p. 5:

> "The equation in (1) is a finite time version of Bellman's equation
> (Bellman, 1957). The function $J_{K-j}$ is usually called the 'optimal
> cost-to-go' from the present state $(\bar S_j, \bar A_{j-1})$ over the
> future intervals of time (Bertsekas & Tsitsiklis, 1996); we call $J_{K-j}$
> the 'optimal benefit-to-go' as we wish to maximize the mean response rather
> than minimize the mean cost."

Schulte et al. (2014, §3, eq. 9, p. 5) state the observed-data version:

> "$Q_k(\bar s_k, \bar a_k)$ are referred to as '$Q$-functions,' viewed as
> measuring the 'quality' associated with using treatment $a_k$ at decision
> $k$ given the history up to that decision and then following the optimal
> regime thereafter."

**RL source.** Sutton & Barto (2018, eq. 3.13): under a deterministic optimal
policy and discount $\gamma$,
$Q^*(s,a) = \mathbb{E}[R_{t+1} + \gamma \max_{a'} Q^*(S_{t+1}, a') \mid S_t = s, A_t = a]$.

**Side-by-side.** With state $= \bar H_k$, horizon finite $K$, terminal reward
$R_{K+1} = Y$, $R_k = 0$ for $k \leq K$, and $\gamma = 1$, Murphy's
$Q_k(\bar h_k, a_k) = \mathbb{E}[V_{k+1} \mid \bar H_k = \bar h_k, A_k = a_k]$
becomes the Bellman expectation under the optimal continuation, which is
$Q^*$ in Sutton-Barto notation.

**Verdict.** **EXACT-UNDER-AUGMENTATION.** Equality holds when (a) the RL
state is taken to be the full history $\bar H_k$ rather than a minimal Markov
state, (b) the horizon is finite and indexed by $k$, and (c) all immediate
rewards are zero except the terminal $Y$. The chapter prose at lines 39-40
already states these as the "dual settings" — Murphy is finite-horizon
undiscounted history-state, Watkins is infinite-horizon discounted
Markov-state. The table row uses the unsubscripted $Q(s,a)$ which silently
collapses the time index; a footnote or stationarity caveat would tighten it.

**Action.** Keep; consider a one-line footnote pointing back to the
"finite-horizon, history-augmented, $\gamma = 1$" reading already established
in the text above the table.

---

## Row 2 — $V_k(\bar h_k) = \max_{a_k} Q_k$ $\leftrightarrow$ $V(s) = \max_a Q(s,a)$

**Claim as written.** LHS: $V_k(\bar h_k) = \max_{a_k} Q_k(\bar h_k, a_k)$.
RHS: $V(s) = \max_a Q(s,a)$, state value function.

**DTR / g-method source.** Murphy (2003) eq. (2) and the chapter's eq.
$\eqref{eq:dtr_bellman}$. Schulte et al. (2014, eq. 11, p. 5):

> "The 'value functions' $V_k(\bar s_k, \bar a_{k-1})$ in (11) and (14)
> reflect the 'value' of a patient's history $\bar s_k, \bar a_{k-1}$
> assuming that optimal decisions are made in the future."

**RL source.** Sutton & Barto (2018, eq. 3.14):
$V^*(s) = \max_a Q^*(s,a)$.

**Side-by-side.** Identical formula once history-augmentation and the
finite-horizon convention from Row 1 are accepted. No further structural
assumption is needed.

**Verdict.** **EXACT-UNDER-AUGMENTATION.** Same caveat as Row 1.

**Action.** Keep.

---

## Row 3 — $d_k^{\text{opt}} = \arg\max Q_k$ $\leftrightarrow$ Greedy policy $\pi^*$

**Claim as written.** LHS: $d_k^{\text{opt}}(\bar h_k) = \arg\max_{a_k}
Q_k(\bar h_k, a_k)$. RHS: greedy policy $\pi^*(s) = \arg\max_a Q(s,a)$.

**DTR / g-method source.** Murphy (2003, eq. 2, restated as
$\eqref{eq:dtr_bellman}$ in the chapter). Schulte et al. (2014, eq. 10, p. 5)
write the observed-data version. Schulte (§3, p. 6):

> "There may not be a unique $d^{\text{opt}}$. At any decision $k$, if there
> is more than one possible option $a_k$ maximizing the $Q$-function, then
> any rule $d_k^{\text{opt}}$ yielding one of these $a_k$ defines an optimal
> regime."

**RL source.** Sutton & Barto (2018, §3.6) define a deterministic greedy
policy as any tie-breaking $\arg\max_a Q^*(s, a)$.

**Side-by-side.** Same operator applied to the (already-established) same
$Q$-function. Non-uniqueness handled identically on both sides.

**Verdict.** **EXACT-UNDER-AUGMENTATION** (inherits Row 1's caveat).

**Action.** Keep.

---

## Row 4 — Propensity $e_k(\bar h_k, a_k)$ $\leftrightarrow$ Behavior policy $\mu_k$

**Claim as written.** LHS: propensity $e_k(\bar h_k, a_k) = \mathbb{P}(A_k =
a_k \mid \bar H_k = \bar h_k)$. RHS: behavior / logging policy $\mu_k(a_k
\mid \bar h_k)$.

**DTR / g-method source.** Robins-Hernan-Brumback (2000), p. 553, defines the
denominator of stabilized weights:

> "$\mathrm{sw}_i = \mathrm{pr}[A_0 = a_{0i}] / \mathrm{pr}[A_0 = a_{0i} \mid
> L_0 = l_{0i}]$... The denominator of $\mathrm{sw}_i$ is informally the
> conditional probability that a subject had his or her own observed
> treatment."

Schulte et al. (2014, §5.2, eq. preceding 30):

> "Let $\pi_K(\bar s_K, \bar a_{K-1}) = \mathrm{pr}(A_K = 1 \mid \bar S_K =
> \bar s_K, \bar A_{K-1} = \bar a_{K-1})$ be the propensity of receiving
> treatment 1 in the observed data as a function of past history."

**RL source.** Precup-Sutton-Singh (2000, §3) define the behavior policy as
$\mu(a \mid s) = \mathbb{P}(A_t = a \mid S_t = s)$ under the logging
distribution; Sutton & Barto (2018, §5.5) formalize this as $b(a \mid s)$.
The per-step importance ratio is $\rho_t = \pi(a_t \mid s_t) / \mu(a_t \mid
s_t)$.

**Side-by-side.** Both are conditional probabilities of the recorded action
given the observed history under the data-generating process. They are
literally the same object once "state" is taken as "history". The notation
$e_k$ (Rosenbaum-Rubin "propensity") emphasizes the binary-treatment scalar
$\mathbb{P}(A_k = 1 \mid \cdot)$; $\mu_k(\cdot \mid \bar h_k)$ emphasizes the
full distribution over the action space. They denote the same thing.

**Verdict.** **EXACT-UNDER-AUGMENTATION.**

**Action.** Keep. (Minor wording tightening optional: "behavior policy
$\mu_k(\cdot \mid \bar h_k)$, equal to $e_k$ in the binary-treatment case",
but unnecessary if Row 1's history-state caveat is footnoted.)

---

## Row 5 — G-computation $\leftrightarrow$ Fitted-$Q$ evaluation

**Claim as written.** LHS: G-computation (sequential conditional regression).
RHS: Fitted-$Q$ evaluation [Ernst et al. 2005].

**DTR / g-method source.** Murphy (2003, eq. 5, p. 8):

> "$\Pr(Y \leq y \mid \bar A_K = \bar a_K) = \int \ldots \int \Pr(Y \leq y
> \mid \bar S_K = \bar s_K, \bar A_K = \bar a_K) \prod_{j=1}^{K} f_j(s_j \mid
> \bar s_{j-1}, \bar a_{j-1})\, ds_j$.   This is Robins' G-computation
> formula."

The sequential-regression form (Bang-Robins 2005; sometimes called iterated
conditional expectation): for $k = K, K-1, \ldots, 1$,
$\widehat Q_k(\bar h_k, a_k) = \widehat{\mathbb{E}}[\widehat V_{k+1}(\bar H_{k+1}) \mid \bar H_k, A_k]$
with $\widehat V_{K+1} = Y$, then plug in $\pi$ to get $\widehat V_k(\bar h_k) = \widehat Q_k(\bar h_k, \pi(\bar h_k))$.

**RL source.** Ernst, Geurts, Wehenkel (2005), "Tree-based batch mode
reinforcement learning," *JMLR* 6:503-556. Their fitted-$Q$-iteration (FQI)
algorithm regresses $r_t + \gamma \max_{a'} \widehat Q^{(N-1)}(s_{t+1}, a')$
on $(s_t, a_t)$ at each iteration $N$. The *evaluation* counterpart (FQE,
Le-Voloshin-Yue 2019) replaces $\max_{a'}$ with the target policy
$\widehat Q^{(N-1)}(s_{t+1}, \pi(s_{t+1}))$.

**Side-by-side.** Both procedures recurse over a regression target whose
right-hand side is "next-step value evaluated under the target policy".
Sequential conditional regression in g-computation is FQE on a
history-augmented finite-horizon MDP. In the infinite-horizon case, FQE
solves a fixed-point equation (Bellman projection) rather than a backward
one-pass fit, but on the finite-horizon side they coincide.

**Verdict.** **EXACT-UNDER-AUGMENTATION.**

**Caveat.** Ernst et al. (2005) actually proposed FQI (optimization), not FQE
(evaluation). The standard FQE citation in the offline-RL chapter is
Le-Voloshin-Yue 2019 or Munos-Szepesvari 2008. The current citation
`Ernst2005` is defensible if read as "fitted-Q methods", but technically FQE
is a downstream variant. Optionally swap to a citation that explicitly
defines FQE, or footnote that g-computation $=$ FQE while $Q$-learning by
backward regression $=$ FQI.

**Action.** Refine wording (optional footnote distinguishing FQI from FQE).

---

## Row 6 — IPTW / MSM $\leftrightarrow$ Off-policy importance sampling

**Claim as written.** LHS: IPTW for marginal structural models
[Robins-Hernan-Brumback 2000]. RHS: off-policy importance sampling
[Precup 2000].

**DTR / g-method source.** Robins-Hernan-Brumback (2000, eq. 14, p. 553):
the stabilized weight for a longitudinal treatment history is
$$\mathrm{sw}_i = \prod_{k=0}^{K} \frac{\mathrm{pr}[A_k = a_{ki} \mid \bar A_{k-1} = \bar a_{k-1,i}]}{\mathrm{pr}[A_k = a_{ki} \mid \bar L_k = \bar l_{ki}, \bar A_{k-1} = \bar a_{k-1,i}]}.$$

Direct quote (p. 552):

> "The denominator of $\mathrm{sw}_i$ is informally the conditional
> probability that a subject had his or her own observed treatment history
> through time $K$. ... if there is no unmeasured confounder given the $L_k$,
> then one can still obtain unbiased estimates of the causal parameter
> $\psi_1$ of model 12 by fitting the logistic model 13 with the stabilized
> weights."

The MSM target $\mathbb{E}[Y^*(\bar a_K)]$ is recovered as $\mathbb{E}[\mathrm{sw} \cdot Y]$
under the IPTW reweighting.

**RL source.** Precup-Sutton-Singh (2000), "Eligibility Traces for Off-Policy
Policy Evaluation," ICML. The per-decision importance-sampling estimator of
$V^\pi$ from data collected under $\mu$:
$\widehat V^\pi = \mathbb{E}_\mu\!\left[\Big(\prod_{t=0}^{T} \frac{\pi(a_t \mid s_t)}{\mu(a_t \mid s_t)}\Big) \sum_t \gamma^t r_t\right]$.

**Side-by-side.** The trajectory weight $\prod_t \pi(a_t \mid s_t)/\mu(a_t \mid s_t)$
in RL is structurally identical to the IPTW weight $\prod_k 1 / e_k$ (with a
deterministic target regime: numerator becomes 1 on the followed path, 0
otherwise) or the *stabilized* weight (numerator $= \mathrm{pr}[A_k]$,
matching Precup's "fixed target policy = marginal of $A_k$" case). Both
estimators are unbiased under positivity and sequential ignorability /
known-behavior-policy.

**Verdict.** **EXACT-UNDER-AUGMENTATION.**

**Action.** Keep.

---

## Row 7 — G-estimation for SNMMs $\leftrightarrow$ Orthogonal-score OPE

**Claim as written.** LHS: g-estimation for structural nested mean models
[`robins1994estimation`]. RHS: orthogonal-score off-policy evaluation
[`lewisSyrgkanis2021dynamicDML`].

**DTR / g-method source — citation problem first.** The chapter cites
`robins1994estimation`, which in `/Users/pranjal/Code/rl/docs/refs.bib`
lines 4132-4140 is:

> Robins, James M. and Rotnitzky, Andrea and Zhao, Lue Ping. "Estimation of
> Regression Coefficients When Some Regressors are Not Always Observed."
> *JASA* 89(427):846-866, 1994.

That is the Robins-Rotnitzky-Zhao IPW-for-missing-data paper, **not** the
SNMM g-estimation paper. The correct SNMM/g-estimation citations are Robins
1994 *Communications in Statistics* ("Correcting for non-compliance...") or
the more comprehensive Robins 2004 ("Optimal Structural Nested Models for
Optimal Sequential Decisions," in Lin-Heagerty eds., Springer). Both appear
in Lewis-Syrgkanis 2021's own reference list (`lewis2021dml.md` lines
805-806). Neither has a BibTeX entry in `refs.bib`; one must be added.

**RL source / Lewis-Syrgkanis 2021 §1.** From `lewis2021dml.md` line 31
(the abstract continuation, Lewis-Syrgkanis §1, p. 2):

> "Our approach addresses these challenges by providing a Neyman orthogonal
> (aka locally robust) g-estimation algorithm for linear structural nested
> mean models (Robins, 1994, 2004), that allows for both continuous and
> discrete treatments in each time period."

And from the abstract (line 15):

> "We propose an extension of the double/debiased machine learning framework
> to estimate the dynamic effects of treatments, which can be viewed as a
> Neyman orthogonal (locally robust) cross-fitted version of g-estimation in
> the dynamic treatment regime."

These two sentences are the canonical paper *self-identifying* its method as
the locally-robust dynamic-DML version of Robins's g-estimation for SNMMs.
The equivalence in the table is therefore claimed by the RL side's primary
source itself.

**Side-by-side.** Both procedures (i) target the blip / structural parameter
in a linear SNMM, (ii) use a moment whose first-order bias vanishes at the
truth, (iii) recurse backward through time peeling off later-period effects.
Lewis-Syrgkanis 2021's contribution over classical g-estimation is the
orthogonalization step (subtracting $\mathbb{E}[H_t(\psi) \mid \bar X_t, \bar T_{t-1}]$
in dynamic dependence on $\psi$) and cross-fitting — i.e., they are not
*identical*, but the LHS is the LHS-direction of a documented bridge that
Lewis-Syrgkanis builds explicitly.

**Verdict.** **EXACT-UNDER-AUGMENTATION** (the augmentation is the
"orthogonalized + cross-fitted" decoration Lewis-Syrgkanis add on top of
Robins's g-estimation).

**Action.** Change citation. Replace `robins1994estimation` with a proper
SNMM/g-estimation key (add Robins 2004 or Robins 1994 *Comm. Stat.* to
refs.bib). Optionally add a footnote with the verbatim Lewis-Syrgkanis line
quoted above.

---

## Row 8 — A-learning / contrast estimation $\leftrightarrow$ Advantage learning

**Claim as written.** LHS: A-learning / contrast estimation
[`schulte2014qlearning`]. RHS: Advantage learning [`Baird1995`].

### 8(a) Citation issue

`Baird1995` in `refs.bib` (lines 2752-2759) is:

> Baird, Leemon. "Residual Algorithms: Reinforcement Learning with Function
> Approximation." *ICML* 1995, pp. 30-37.

The 1995 ICML paper is on **residual gradient algorithms**, not on advantage
updating. The originator of "advantage updating" is Baird's 1993 Wright
Laboratory technical report:

> Baird, Leemon C. "Advantage Updating." Tech. Rep. WL-TR-93-1146, Wright
> Laboratory, Wright-Patterson AFB, OH, 1993.

(Closely related: Harmon, Baird & Klopf 1995 "Advantage Updating Applied to
a Differential Game," NIPS, and Baird 1994 "Reinforcement learning in
continuous time: advantage updating," IEEE WCCI.) The 1995 ICML paper does
*not* introduce advantage learning. The current citation is wrong by
provenance, even though both papers are by Baird.

**Recommendation.** Replace `Baird1995` with a new entry `Baird1993` (Wright
Lab WL-TR-93-1146). If a peer-reviewed alternative is preferred, use
Harmon-Baird-Klopf 1995 NIPS, which is the first archival publication
attaching the term "advantage" to a learning rule. Both need to be added to
`refs.bib`.

### 8(b) Math equivalence

**DTR / g-method source.** Schulte et al. (2014), §5.2 (contrast / blip /
regret). Direct quote from `schulte2014qlearning.md` lines 185-188:

> "$d_2^{\text{opt}}(\bar s_2, a_1; \xi_2)$ depends only on $H_2^T \psi_2 =
> Q_2(\bar s_2, a_1, 1; \xi_2) - Q_2(\bar s_2, a_1, 0; \xi_2)$. This reflects
> the general result that, for purposes of deducing the optimal regime, for
> each $k = 1, \ldots, K$, it suffices to know the contrast function
> $C_k(\bar s_k, \bar a_{k-1}) = Q_k(\bar s_k, \bar a_{k-1}, 1) - Q_k(\bar s_k,
> \bar a_{k-1}, 0)$."

And lines 187-188:

> "In the case of two treatment options we consider here, the contrast
> function is also referred to as the optimal-blip-to-zero function (Robins,
> 2004; Moodie, Richardson and Stephens, 2007). Murphy (2003) considers the
> expression $C_k(\bar S_k, \bar A_{k-1})[I\{C_k(\bar S_k, \bar A_{k-1}) > 0\}
> - A_k]$, referred to as the advantage or regret function, as it represents
> the 'advantage' in response incurred if the optimal treatment at the $k$th
> decision were given relative to that actually received (or, equivalently,
> the 'regret' incurred by not using the optimal treatment)."

So Schulte explicitly *uses the word "advantage"* for the Murphy regret
$C_k[I\{C_k > 0\} - A_k]$.

**RL source.** Baird (1993, "Advantage Updating") and Sutton & Barto (2018,
§13.5): the advantage function is
$A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s) = Q^\pi(s, a) - \mathbb{E}_{a' \sim \pi}[Q^\pi(s, a')]$.

**Side-by-side.** In the chapter's $\bar H_k$ notation:

| Quantity | DTR / g-method (binary $A_k$) | RL (general $A_k$) |
|---|---|---|
| Baseline | $Q_k(\bar h_k, 0)$ (reference action) | $V^\pi(\bar h_k) = \mathbb{E}_\pi[Q_k(\bar h_k, A_k)]$ (policy mean) |
| Contrast | $C_k = Q_k(\bar h_k, 1) - Q_k(\bar h_k, 0)$ | $A_k^\pi = Q_k(\bar h_k, a_k) - V_k^\pi(\bar h_k)$ |
| Anchored at | A fixed reference $a_k = 0$ | The policy itself |

The Schulte/Murphy "advantage" Schulte quotes ($C_k [I\{C_k > 0\} - A_k]$)
is *not* $Q_k - V_k$; it is the regret, the value lost by deviating from
$d^{\text{opt}}$. The blip / contrast $C_k$ is $Q_k(\cdot, 1) - Q_k(\cdot,
0)$, anchored at the reference action.

The two objects coincide when (i) the action set is binary, (ii) the
reference action $0$ is chosen so that $V^\pi(\bar h_k) = Q_k(\bar h_k, 0)$
(e.g. a deterministic policy that always plays $0$, or a baseline-zero
convention). For a stochastic $\pi$ over $\{0,1\}$, $V^\pi = (1-\pi)
Q_k(\cdot,0) + \pi Q_k(\cdot,1)$, so $A^\pi(\bar h_k, 1) = (1-\pi) C_k$ and
$A^\pi(\bar h_k, 0) = -\pi C_k$. These are scalar multiples of $C_k$, not
equal to it.

**Verdict.** **STRUCTURAL.** A-learning's contrast and RL's advantage are
both action-conditional deviations from a baseline, but the baseline differs
(reference action vs. policy mean). They coincide only in the binary,
deterministic-target, zero-baseline corner case. The chapter's table
implies stronger equivalence than the math supports.

**Action.** Refine wording AND change citation. Suggested table row:
"A-learning / contrast (blip) estimation [add proper SNMM cite] $\leftrightarrow$
Advantage function [Baird 1993]". Add a footnote: "The DTR contrast
$C_k(\bar h_k) = Q_k(\bar h_k, 1) - Q_k(\bar h_k, 0)$ is anchored at the
reference action $a_k = 0$, whereas the RL advantage
$A_k^\pi(\bar h_k, a_k) = Q_k(\bar h_k, a_k) - V_k^\pi(\bar h_k)$ is anchored
at the policy mean; the two coincide up to a $\pi$-dependent scalar in the
binary-action case."

---

## Row 9 — Sequential ignorability + positivity + consistency $\leftrightarrow$ Markov property + full support of $\pi^*$

**Claim as written.** LHS: sequential ignorability + positivity +
consistency. RHS: Markov property + full support of $\pi^*$.

**DTR / g-method source.** From the chapter itself, line 26 (consolidating
Murphy 2003 §2 and Schulte 2014 §2):

> "*Consistency* states that $Y = Y^*(\bar A_K)$ and $S_k = S_k^*(\bar
> A_{k-1})$. *Positivity* requires that every action of interest receives
> strictly positive probability conditional on every history that arises in
> the population. *Sequential ignorability* requires $A_k \perp \{Y^*,
> S_{k+1}^*, \ldots\} \mid \bar H_k$ for every $k$, the assumption that the
> treating physician ... makes choices that, conditional on observed history,
> are independent of unobserved future state and outcome."

Murphy (2003) §2 (`murphy2003dtr.md` line 85):

> "No Unmeasured Confounders: For each $j = 1, \ldots, K$, $A_j$ is
> independent of $O^*$ given $\{S_1, A_1, S_2, A_2, \ldots, S_j\}$. This
> assumption is also called sequential ignorability."

**RL source.** Sutton & Barto (2018, §3.1, eq. 3.1):

> Markov property: $\mathbb{P}(S_{t+1} = s', R_{t+1} = r \mid S_0, A_0, R_1,
> \ldots, S_t, A_t) = \mathbb{P}(S_{t+1} = s', R_{t+1} = r \mid S_t, A_t)$.

Full support of the *target* policy $\pi^*$ (or the *behavior* policy in
off-policy evaluation) is the "coverage" condition needed for IS / FQI to be
unbiased.

**Side-by-side.**

| Assumption | What it constrains | Type |
|---|---|---|
| Sequential ignorability ($A_k \perp$ counterfactuals $\mid \bar H_k$) | The mechanism by which $A_k$ is selected (no unobserved confounder) | Epistemic / identification |
| Markov property ($S_{t+1} \perp$ history $\mid S_t, A_t$) | The structure of the data-generating process (state is sufficient) | Structural / DGP |
| Positivity ($e_k(\bar h_k, a) > 0$ for every action of interest) | Coverage of the action space by the logging distribution | Identification |
| Full support of $\pi^*$ | Coverage by the target policy (relevant for on-policy estimation) | Identification |
| Consistency ($Y = Y^*(\bar A_K)$) | Linking observed outcomes to potential outcomes | Bridging |

These are **not the same set of assumptions**. Specifically:

1. Sequential ignorability is a no-unmeasured-confounders condition about the
   *true treatment-assignment mechanism*. It has no Markov-property
   counterpart on the RL side. In offline RL the equivalent is "the behavior
   policy is known and depends only on the observed state/history".
2. The Markov property is a *state-sufficiency* condition. It has no
   sequential-ignorability counterpart on the DTR side; in DTR one always
   conditions on the full history $\bar H_k$ precisely to avoid relying on a
   Markov property.
3. Positivity (DTR) and full support (RL) align reasonably well, but
   positivity is on the *behavior* (observed-data) distribution, while "full
   support of $\pi^*$" as written conflates target and behavior coverage.
4. Consistency (potential-outcomes $\leftrightarrow$ observed-outcomes) has
   no direct RL counterpart at all; in RL there is no potential-outcomes
   notation, so the question never arises.

**Verdict.** **MISLEADING.** The row pairs two different assumption types
(epistemic identification vs. structural state-sufficiency) and drops
consistency entirely on the right. A reader who took the table at face value
would conclude that "the Markov property buys you sequential ignorability";
it does not.

**Action.** Split into two rows (and drop one). Proposed replacement:

| DTR / g-methods | RL |
|---|---|
| Sequential ignorability ($A_k \perp$ counterfactuals $\mid \bar H_k$) | Known behavior policy $\mu_k(\cdot \mid \bar h_k)$ (offline RL) |
| Positivity: $e_k(\bar h_k, a) > 0$ for all $a$ of interest | Coverage: $\mu_k(a \mid \bar h_k) > 0$ for all $a$ in $\mathrm{supp}\,\pi^*$ |

Drop "Markov property" from the table entirely (it is a structural choice on
the RL side, not an identification assumption, and the chapter's setup at
line 39 already explains that history-augmentation is what removes the need
for it on the DTR side). Drop "consistency" from the equivalence as well
(it has no RL analog) but keep the assumption explicit in the surrounding
text.

---

## Additional issues flagged in unflagged rows

- **Row 5 / `Ernst2005` citation.** Ernst-Geurts-Wehenkel 2005 introduces
  fitted-$Q$ *iteration* (FQI, control). The chapter uses it to gloss
  fitted-$Q$ *evaluation* (FQE, policy evaluation). These are different
  algorithms. Acceptable as a stand-in if read loosely; tighter would be to
  cite Le-Voloshin-Yue 2019 "Batch Policy Learning under Constraints" or
  Munos-Szepesvari 2008.

- **Row 1-3 / unsubscripted $Q(s,a)$ on the RHS.** The chapter prose at line
  39 explains that Murphy is finite-horizon undiscounted with state $= \bar H_k$
  and Watkins is infinite-horizon discounted with Markov state. The table
  loses this. A single footnote attached to Row 1 would carry the load for
  Rows 1-3 together.

- **Prose claim "exact in the column-by-column sense" (line 69).** Given the
  Row 8 (STRUCTURAL) and Row 9 (MISLEADING) verdicts, the prose claim is
  stronger than the table supports. The sentence should be softened to
  something like "the translation is procedural in the column-by-column
  sense: each entry on the left, applied to history-augmented data and
  finite-horizon undiscounted accounting, returns the same numerical object
  as the entry on the right, modulo the binary-action / baseline-anchoring
  caveats noted below".

---

## Revisions Required for §1

The following diffs target `/Users/pranjal/Code/rl/ch10b_rl_for_ci/tex/rl_for_ci.tex`.

### R1. Row 7 citation (line 61)

```diff
- G-estimation for structural nested mean models \citep{robins1994estimation} & Orthogonal-score off-policy evaluation \citep{lewisSyrgkanis2021dynamicDML} \\
+ G-estimation for structural nested mean models \citep{robins2004snmm} & Orthogonal-score off-policy evaluation \citep{lewisSyrgkanis2021dynamicDML} \\
```

Add to `/Users/pranjal/Code/rl/docs/refs.bib`:

```bibtex
@incollection{robins2004snmm,
  author    = {Robins, James M.},
  title     = {Optimal Structural Nested Models for Optimal Sequential Decisions},
  booktitle = {Proceedings of the Second Seattle Symposium in Biostatistics},
  editor    = {Lin, D. Y. and Heagerty, P. J.},
  series    = {Lecture Notes in Statistics},
  volume    = {179},
  pages     = {189--326},
  publisher = {Springer},
  address   = {New York},
  year      = {2004}
}
```

### R2. Row 8 citation (line 62)

```diff
- A-learning / contrast estimation \citep{schulte2014qlearning} & Advantage learning \citep{Baird1995} \\
+ A-learning / contrast (blip) estimation \citep{schulte2014qlearning, robins2004snmm} & Advantage function \citep{Baird1993advantage} \\
```

Add to `refs.bib`:

```bibtex
@techreport{Baird1993advantage,
  author      = {Baird, Leemon C.},
  title       = {Advantage Updating},
  institution = {Wright Laboratory, Wright-Patterson AFB},
  number      = {WL-TR-93-1146},
  year        = {1993}
}
```

### R3. Row 8 footnote

Insert immediately after the table (between line 67 `\end{table}` and line
69 `The translation is exact ...`):

```latex
\footnote{The DTR contrast $C_k(\bar h_k) = Q_k(\bar h_k, 1) - Q_k(\bar h_k, 0)$
is anchored at the reference action $a_k = 0$, whereas the RL advantage
$A_k^\pi(\bar h_k, a_k) = Q_k(\bar h_k, a_k) - V_k^\pi(\bar h_k)$ is anchored
at the policy mean. The two coincide up to a $\pi$-dependent scalar in the
binary-action case and agree exactly when the baseline policy is the constant
$a_k = 0$.}
```

### R4. Row 9 split (line 63)

```diff
- Sequential ignorability + positivity + consistency & Markov property + full support of $\pi^*$ \\
+ Sequential ignorability: $A_k \perp \{Y^*, S^*_{k+1}, \ldots\} \mid \bar H_k$ & Known behavior policy $\mu_k(\cdot \mid \bar h_k)$ (offline RL setting) \\
+ Positivity: $e_k(\bar h_k, a) > 0$ for all $a$ of interest & Coverage: $\mu_k(a \mid \bar h_k) > 0$ for all $a \in \mathrm{supp}\,\pi^*$ \\
```

Drop "Markov property" from the table. Drop "consistency" from the table
(it remains stated in the prose at line 26).

### R5. Soften the "exact" claim (line 69)

```diff
- The translation is exact in the column-by-column sense. Each entry on the left, treated as a procedure on the observed-data distribution, returns the same numerical object as the entry on the right when applied to the same data.
+ The translation is procedural in the column-by-column sense. Each entry on the left, applied to history-augmented finite-horizon undiscounted data, returns the same numerical object as the entry on the right, with two caveats. The A-learning contrast and the RL advantage agree only up to a baseline-anchoring rescaling (footnote above). Sequential ignorability and "known behavior policy" play the same identification role but are not literally the same statement, the former being a property of the unobserved confounder structure and the latter a property of the data-collection protocol.
```

### R6. Optional Row 1 footnote (line 55)

Attach to the first table row:

```latex
\footnote{Throughout the table, the RL state $s$ is read as the full history
$\bar h_k = (\bar s_k, \bar a_{k-1})$ at decision $k$, the horizon is finite
and equal to $K$, all immediate rewards are zero except $R_{K+1} = Y$, and
$\gamma = 1$. The Markov property used in standard RL is replaced by
history-augmentation; the dual setting was discussed in the paragraph
preceding the table.}
```

### R7. Optional Row 5 footnote (line 59)

```latex
\footnote{Strictly, \citet{Ernst2005} introduced fitted-$Q$ iteration (FQI),
the control version. The evaluation counterpart (FQE) was formalized later
(see, e.g., \citealp{munosSzepesvari2008, leVoloshinYue2019}). G-computation
by sequential conditional regression equals FQE on history-augmented data;
$Q$-learning by backward regression equals FQI.}
```

(Skip if `munosSzepesvari2008` / `leVoloshinYue2019` keys do not exist;
otherwise add the entries.)

---

## Summary of verdicts

| Row | Verdict | Action |
|---|---|---|
| 1 $Q_k \leftrightarrow Q(s,a)$ | EXACT-UNDER-AUGMENTATION | Keep; optional footnote (R6) |
| 2 $V_k \leftrightarrow V(s)$ | EXACT-UNDER-AUGMENTATION | Keep |
| 3 $d_k^{\text{opt}} \leftrightarrow \pi^*$ | EXACT-UNDER-AUGMENTATION | Keep |
| 4 propensity $\leftrightarrow$ behavior policy | EXACT-UNDER-AUGMENTATION | Keep |
| 5 g-computation $\leftrightarrow$ FQE | EXACT-UNDER-AUGMENTATION | Refine wording; optional footnote (R7) |
| 6 IPTW $\leftrightarrow$ off-policy IS | EXACT-UNDER-AUGMENTATION | Keep |
| 7 g-estimation SNMM $\leftrightarrow$ orthogonal-score OPE | EXACT-UNDER-AUGMENTATION | Change citation (R1) |
| 8 A-learning $\leftrightarrow$ advantage learning | STRUCTURAL | Change citation + add footnote (R2, R3) |
| 9 sequential ignorability $\leftrightarrow$ Markov + full support | MISLEADING | Split into two rows, drop one (R4) |

Prose "exact" claim: softened to "procedural" (R5).
