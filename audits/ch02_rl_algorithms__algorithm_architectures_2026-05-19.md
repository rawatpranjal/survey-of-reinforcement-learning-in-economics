# Audit: ch02_rl_algorithms/sims/algorithm_architectures.py

**Date:** 2026-05-19
**Diagram-only:** yes (score capped at 25% per CLAUDE.md)
**Cited tex file(s):** `/Users/pranjal/Code/rl/ch02_rl_algorithms/tex/rl_algorithms.tex` — figure `\includegraphics` at line 197, caption at line 198, `\label{fig:algorithm_architectures}` at line 199. Surrounding prose: §"Williams (1992)" (line 49+), §"Actor-Critic Methods (2000)" (line 93+), with the TD-error definition $\delta_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)$ on line 33, and the actor update using $\delta_t$ as advantage estimate on line 103.
**Cited paper PDFs read:** none in `papers/` for the three algorithms depicted (DQN/REINFORCE/Actor-Critic). The four PDFs present (`andrychowicz2021`, `engstrom2020`, `huang2024`, `tesauro1995`) are tangential implementation/history papers, not the originating papers for the three architectures. The schematic content is, however, fully fixed by standard textbook material (Sutton & Barto Ch. 13) and by the chapter's own prose, which is what the diagram must match.

## 1. Algorithm Identity

The script draws three schematic flowcharts. The hostile-reviewer question for a diagram is: does each panel name and depict the algorithm it claims to?

- **Panel (a) DQN.** Boxes: $s_t \to Q(s,\cdot;\theta) \to \arg\max_a \to a_t^*$. This is the standard value-based decision pipeline. Matches the chapter prose on DQN (line 161) and the caption ("maps states to Q-values for all actions, selecting the argmax"). Strictly, DQN as defined in Mnih2015 uses an $\varepsilon$-greedy behavior policy, not a pure argmax, but the diagram is the *exploitation* path and the caption says exactly that. No issue.
- **Panel (b) REINFORCE.** Boxes: $s_t \to \pi_\theta(a|s) \to a \sim \pi_\theta(\cdot|s) \to a_t$. Correct depiction of stochastic policy + sampling. Matches the chapter's introduction of policy gradients (line 57). The diagram does not show the return-weighted gradient update, but neither does the caption claim to — it depicts forward decision-making, with the environment loop carrying the reward. Acceptable abstraction level.
- **Panel (c) Actor-Critic.** Shows separate $\pi_\theta(a|s)$ actor and $V_w(s)$ critic, both fed by $s_t$, actor producing $a_t$, critic producing $\delta_t$, and dashed feedback arrows from $\delta_t$ back into both actor and critic. The TD-error formula $\delta_t = r_t + \gamma V_w(s_{t+1}) - V_w(s_t)$ is printed at the bottom. This is the canonical advantage-actor-critic schematic.

Two nits worth recording:

- The TD-error formula shown ($\delta_t = r_t + \gamma V_w(s_{t+1}) - V_w(s_t)$) uses $r_t$ as the immediate reward. The chapter prose at line 33 writes the same expression with $r_{t+1}$ (Sutton & Barto convention). The discrepancy is a notation drift between the figure and the prose in the same chapter. Reviewer 2 would note this in 5 seconds.
- The env-loop labels in panel (c) place "$a_t$" at $(4.6, 0.5)$ and "$r_{t+1}, s_{t+1}$" at $(0.4, 0.5)$, but the panel-(c) arrows from $a \to \text{env}$ and $\text{env} \to s$ are drawn with `curve=-0.55` (sweeping outward), whereas panels (a) and (b) use `curve=+0.4` (sweeping inward/downward). The labels in (a) and (b) sit at $y=0.35$; in (c) the action goes through $y_\text{actor}=2.5$ first, so the geometry is asymmetric across panels even though the caption says "Each panel includes the environment feedback loop." This is a visual-consistency nit, not a content error.

No placeholder implementations, no missing key components for diagram purposes. Algorithm identities match what the caption says.

## 2. Environment / MDP Fidelity

N/A in the standard sense — there is no MDP being simulated. The "environment" appears only as a gray box labeled "Environment" feeding a reward and next state back to $s_t$. The depiction is generic and consistent with the standard MDP loop. No fidelity claim is made or violated.

## 3. Data Integrity

N/A — no `compute_data()` exists by design. The script is purely a Matplotlib drawing routine. `--data-only` correctly exits with "No computation to cache (diagram-only script)." per the CLAUDE.md interface convention. No hardcoded "results" are reported; the script prints only the output path and a confirmation line.

## 4. Comparison Fairness

N/A for diagram-only — no algorithms are being numerically compared. The caption asserts that actor-critic "yields lower-variance policy updates than REINFORCE's Monte-Carlo return," but this is a theoretical statement supported in the surrounding prose (line 95+, "address the high variance of REINFORCE by replacing Monte Carlo returns with bootstrapped TD targets") and does not require empirical comparison in the figure.

## 5. Theoretical Sanity Checks

N/A — no numerical results. The captions' theoretical claims (DQN selects argmax of Q; REINFORCE samples from a stochastic policy; actor-critic feeds back TD error to lower variance) are textbook-correct and aligned with Sutton & Barto Ch. 13 and Williams (1992).

## 6. Information Leakage

N/A — no agent is learning anything. There is no training data, no held-out evaluation, no oracle.

## 7. Seed & Reproducibility

N/A in the Monte-Carlo sense — the script is deterministic. Re-running produces the identical PNG. `np.random` is imported but never seeded because never used. Reproducibility is trivially satisfied.

## Hostile-Reviewer Summary

A diagram-only schematic that does what its caption says: three panels for value-based, policy-gradient, and actor-critic decision flows, plus a generic environment loop. The forward decision pipelines are correctly depicted, the actor-critic feedback structure (dashed arrows from $\delta_t$ back into both networks) is the canonical advantage-actor-critic picture, and the TD-error formula is rendered in the figure. Two nits Reviewer 2 will spot: (i) the figure writes $\delta_t = r_t + \gamma V_w(s_{t+1}) - V_w(s_t)$ while the chapter prose at line 33 writes the same formula with $r_{t+1}$, a small intra-chapter notation drift; (ii) the panel-(c) env-loop geometry sweeps outward while panels (a) and (b) sweep inward, breaking the visual symmetry the caption implies ("Each panel includes the environment feedback loop"). Neither nit changes the substance of what the figure asserts. The diagram does not visually contradict its caption, so the 25% cap binds; the actual score sits below the cap because both flaws are cosmetic / typographical, not conceptual.

**Bullshit score: 15%** — Reviewer 2 catches the $r_t$ vs $r_{t+1}$ drift between figure and prose and the asymmetric env-loop geometry in panel (c), but the three architectures are accurately depicted and the caption holds.
