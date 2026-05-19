# Audit: ch03b_deeprl_practice/sims/overestimation_bias.py

**Date:** 2026-05-19
**Diagram-only:** yes — script computes closed-form / numerical-quadrature properties of $\max$ over iid normals and renders one figure. No Q-learning rollout, no environment, no Double-Q baseline. Per CLAUDE.md, diagram-only sims cap at 25% unless the figure visually contradicts the caption.
**Cited tex file(s):** `ch03b_deeprl_practice/tex/deeprl_practice.tex` §`sec:overestimation` (lines 26–45). Figure `\label{fig:overestimation_bias}` at line 44.
**Cited paper PDFs read:** present in `papers/`: `vanHasselt2016_double_qlearning.pdf` (+ `.md`), `fujimoto2018_td3.pdf`, `vanHasselt2018_deadly_triad.pdf`, `Ciosek2019` etc. The script itself does not implement any algorithm from these; it depicts Jensen's inequality on iid normals. Thrun-Schwartz 1993 and Van Hasselt 2010 (original Double-Q NeurIPS) are not in `papers/` as PDFs but are cited in the tex.

## 1. Algorithm Identity

The script does not implement Q-learning or Double Q-learning. It computes the analytical density of $\max_{i=1,2} \hat Q_i$ where $\hat Q_i \stackrel{\mathrm{iid}}{\sim} \mathcal N(\mu, \sigma^2)$ and plots it against the marginal density. For $n$ iid normals:

- Marginal PDF: $\phi_{\mu,\sigma}(x)$. Correctly coded.
- Max PDF (n=2): $f_{\max}(x) = 2\,\phi(x)\,\Phi(x)$. Correctly coded (line 37, 41).
- $\mathbb E[\max] = \mu + \sigma/\sqrt{\pi}$ for $n=2$. Correctly coded (line 29) and matches numerical quadrature to 6 d.p. (`E_max analytical = 2.564190`, `E_max numerical = 2.564190`).
- For general $n$: PDF $= n\,\phi(z)\,\Phi(z)^{n-1}$ on standard normal scale. Correctly coded (line 80).
- Reported $\mathbb E[\max_n] - \mu$ scaling (0.564, 1.163, 1.539, 1.867, 2.249, 2.508 for $n=2,5,10,20,50,100$) matches standard order-statistics tables.

This is not an implementation of Q-learning or Double Q-learning. It is a single static illustration of Jensen's inequality, which is what the figure caption claims it is. The tex prose surrounding the figure correctly attributes the bias mechanism (Thrun-Schwartz, Van Hasselt 2010) and discusses Double DQN / TD3 separately as remedies, without claiming this figure shows any algorithm. No identity-mismatch.

Minor: the figure caption says "the bias exceeds $2.5\sigma$" at $n=100$; the stdout reports $2.5076\sigma$. Technically true but only just; "exceeds 2.5σ" is fair.

## 2. Environment / MDP Fidelity

No environment / MDP. The script studies a non-sequential distributional fact: $\mathbb E[\max_i X_i] \ge \max_i \mathbb E[X_i]$. This is the iid-normals reduction of the Thrun-Schwartz argument. The tex never claims a specific MDP is being simulated; the figure is positioned strictly as an illustration of the Jensen-inequality mechanism.

A more ambitious sim (Sutton-Barto Example 6.7 left/right MDP with $N(-0.1, 1)$ rewards, or the Van Hasselt 2010 multi-action bandit) would show Q-learning vs. Double-Q empirically. This script does not attempt that. The tex does not promise that, so there is no fidelity violation, but it is a missed opportunity: the chapter's central empirical claim ("Double DQN reduces overestimation by a factor of 3-5") is supported only by citation, not by any sim in this repo. That is a chapter-content gap, not an audit failure of this script.

## 3. Data Integrity

`generate_outputs()` computes everything fresh on each run via `scipy.stats.norm` and `scipy.integrate.quad`. No cache, no hardcoded numbers reported as results. The numerical-quadrature integral of the max-PDF returns 1.0000000000 (correct sanity check that the density integrates to 1), and the numerical mean matches the analytical $\mu + \sigma/\sqrt\pi$ to 6 d.p. The bias annotation in the figure (`f'bias = sigma/sqrt(pi) ≈ {E_max_analytical - mu:.3f}'`) is computed from the same variable, not hardcoded. Stdout file matches what the script prints. Clean.

## 4. Comparison Fairness

No comparison is performed. There is no Double Q-learning curve, no Q-learning curve, no oracle. The figure compares the marginal density to the max density at fixed $(\mu, \sigma, n=2)$, plus a stdout table over $n \in \{2,5,10,20,50,100\}$. Since no two algorithms are pitted against each other, fairness is vacuous. A hostile reviewer would note: a chapter section titled "Value Overestimation" with three paragraphs naming Double DQN, Clipped Double Q, and the deadly triad gets only a Jensen-inequality cartoon as its figure. The reviewer might say "Where is your DQN-vs-DDQN learning curve?" — that is a chapter-level critique, but it is *not* a misrepresentation by this script.

## 5. Theoretical Sanity Checks

- Closed form for $n=2$: $\mathbb E[\max] - \mu = \sigma/\sqrt\pi \approx 0.5642$. Script: 0.564190. ✓
- Density integration: 1.0000000000. ✓ (scipy `quad` over $[\mu-10\sigma, \mu+10\sigma]$ is more than enough for a normal mixture.)
- Order-statistics asymptotic: $\mathbb E[\max_n] - \mu \sim \sigma\sqrt{2\ln n}$ for large $n$. At $n=100$: $\sqrt{2\ln 100} = \sqrt{9.21} \approx 3.035$. Reported: 2.508. Order statistics converge slowly to that asymptotic; 2.508 is consistent with finite-$n$ values from standard tables (e.g. Harter 1961: $E[X_{(100:100)}] = 2.5076$). ✓
- Sign: positive bias for max over noisy estimates, as Thrun-Schwartz 1993 and Van Hasselt 2010 Theorem 1 require. ✓

Van Hasselt 2010 Theorem 1 is more general (any distribution with mean 0 noise gives non-negative bias), but the iid-normal special case is the canonical pedagogical instance and is what the figure caption claims.

## 6. Information Leakage

There is nothing to peek at. The true $\mu$ is set as a fixed parameter for the illustration; there is no "learned" estimator that could cheat. No leakage possible.

## 7. Seed & Reproducibility

No randomness. Closed-form + deterministic numerical quadrature. Reproducible by definition. The $N \ge 10$ seeds requirement does not apply (and stdout/figure are byte-identical on re-runs given the same scipy version).

## Hostile-Reviewer Summary

Mechanical content is correct: PDFs are right, expectations match to 6 d.p., the scaling table is consistent with standard order-statistics references, the figure annotations are computed from the same variables they label, and the caption's claims (n=2 shift of $\sigma/\sqrt\pi \approx 0.56$, $n=100$ exceeding 2.5σ) survive checking. The tex prose around the figure correctly attributes the bias mechanism (Thrun-Schwartz, Van Hasselt 2010) and never claims this figure is anything more than the Jensen-inequality illustration that it is.

The hostile reviewer's complaints are at the chapter level, not the script level: a "Value Overestimation and Spikes" section that names DQN, Double DQN, Clipped Double Q (TD3), and soft divergence is supported by exactly one analytic Gaussian density plot and a stdout table. No empirical DQN-vs-DDQN curve, no soft-divergence trace, no Sutton-Barto §6.7 left/right MDP demonstration. The Reviewer 2 line is "fine math, but where is the simulation?" — that bumps it past 0% but is not a misrepresentation; it is a thin demonstration that delivers what its caption promises.

One nit: the caption says "the bias exceeds $2.5\sigma$" at $n=100$ when the actual value is $2.5076\sigma$; "approaches 2.5σ" or "slightly above 2.5σ" would be more honest, though "exceeds 2.5σ" is technically true.

Diagram-only cap applies (25% per CLAUDE.md); no visual-vs-caption contradiction, no other code-level issue.

**Bullshit score: 15%** — Reviewer 2 grumbles about thin empirical support for a chapter section that names DQN/DDQN/TD3 and would prefer a real Q-learning-vs-Double-Q curve, but the math, code, and caption are mutually consistent and theoretically clean. Diagram-only cap applies (25%); landing below it.
