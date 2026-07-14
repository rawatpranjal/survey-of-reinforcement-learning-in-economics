# Audit: ch02_rl_algorithms/sims/algorithm_architectures.py

**Sim:** `ch02_rl_algorithms/sims/algorithm_architectures.py` (three-panel schematic: DQN / REINFORCE / Actor-Critic)
**Date:** 2026-07-14
**Type:** FULL calibration re-audit (treated as never-audited through step 5; prior audits read only at step 6)
**Diagram-only:** yes — 25% score cap applies per CLAUDE.md unless the diagram visually contradicts its caption
**Files read this pass:**
- `/Users/pranjal/Code/rl/ch02_rl_algorithms/sims/algorithm_architectures.py` (end to end)
- `/Users/pranjal/Code/rl/ch02_rl_algorithms/sims/algorithm_architectures.png` (viewed, rendered)
- `/Users/pranjal/Code/rl/ch02_rl_algorithms/sims/algorithm_architectures_stdout.txt`
- `/Users/pranjal/Code/rl/ch02_rl_algorithms/tex/rl_algorithms.tex` (figure at 195-200, caption 198; Actor-Critic §93-105; policy-gradient/SAC context 49-193)
- `/Users/pranjal/Code/rl/sims/plot_style.py` (constant verification)
- Prior audits (step 6 only): `audits/ch02_rl_algorithms__algorithm_architectures_2026-05-19.md`; `ch02_rl_algorithms/sims/bullshit-detector_algorithm_architectures_2026-05-18_v3.md`

**Provenance note:** the script mtime is 2026-05-18 17:39; the v3 detector was written 17:45, i.e. after the script's last edit. `git log` shows no commit touching the script since. The caption in the live tex (line 198) is byte-identical to what both prior audits quote. Consequence: every code-level and caption-level item the prior audits left open is necessarily still open, because nothing was edited.

---

## Step-3 statement (written before judging)

**(i) Claim the chapter advances here.** The three canonical model-free RL algorithm families are distinguished by their architectural wiring, i.e. by how a state is turned into an action and where the learning signal enters. Value-based methods select an action by taking the argmax over a Q-network; policy-gradient methods sample from a parameterized stochastic policy; actor-critic methods pair a policy network with a value network whose temporal-difference error supplies a lower-variance learning signal than a Monte-Carlo return.

**(ii) What the diagram is evidence FOR.** The figure is a summary schematic (placed after the SAC subsection, floated, not `\ref{}`-ed in prose) used as evidence for the *structural* contrast among the three families: the forward state-to-action pipeline in each, plus the specific point that actor-critic's distinguishing feature is the critic's TD-error δ_t feeding back into both the actor and the critic during training. It is a taxonomy picture, not a numerical result.

---

## Criteria verdicts

### (a) CORRECTNESS — PASS (one notation inconsistency)

Every arrow encodes the algorithm it names; no arrow points the wrong way.

- Panel (a) DQN: `s_t → Q(s,·;θ) → arg max_a → a_t*` (`.py:181-190`). The `·` denotes Q over all actions; argmax selects greedily. This is the exploitation path, which is exactly what the caption describes. DQN acts ε-greedy while training, but the caption does not claim otherwise. Correct.
- Panel (b) REINFORCE: `s_t → π_θ(a|s) → a ∼ π_θ(·|s) → a_t` (`.py:234-243`). Correct stochastic-policy + sampling flow.
- Panel (c) Actor-Critic: separate `Actor π_θ(a|s)` and `Critic V_w(s)`, both fed by `s_t`; actor → `a_t`; critic → `δ_t` (solid); dashed δ_t → actor and δ_t → critic feedback (`.py:295-315`). This is the canonical advantage-actor-critic schematic. The two arcs between Critic and δ_t both use `curve=0.30`, but because direction is reversed the forward (solid) bows up and the feedback (dashed) bows down, so they render distinct — verified in the PNG.

One genuine flaw, at notation level: the TD formula printed in panel (c) is `δ_t = r_t + γ V_w(s_{t+1}) − V_w(s_t)` (`.py:343`), using `r_t`. The chapter's own TD-error equation at `rl_algorithms.tex:97` writes `δ_t = r_{t+1} + γ V(s_{t+1}) − V(s_t)`, using `r_{t+1}` (Sutton-Barto convention). Sharper than the prior framing: the inconsistency is also *intra-panel* — the same panel's environment-loop label (`.py:338`) reads `r_{t+1}, s_{t+1}`, so within one panel the reward is indexed `r_{t+1}` on the loop and `r_t` in the formula. Neither index is wrong in the absolute, but the document contradicts itself. This is a real, catchable flaw; it does not change the depicted structure.

The critic → δ_t solid arrow is a standard simplification: δ_t also depends on the environment's reward and next state, which the diagram does not wire into δ_t. Acceptable schematic abstraction; consistent with how the caption labels δ_t "the critic's TD error."

### (b) PRESENTATION — PASS with caveats

Caption-to-panel coverage: all promised elements are present. Each panel has the state node, the family-specific operation nodes, the action node, and an Environment box with the `a→env` and `env→s` gray loop and `a_t` / `r_{t+1}, s_{t+1}` labels. Panel (c) additionally shows the δ_t feedback and the TD formula. Nothing promised is missing; nothing shown contradicts the caption.

Caveats (all cosmetic, none conceptual):
- "yielding lower-variance policy updates than REINFORCE's Monte-Carlo return" (caption) has no visual referent — panel (b) has no `G_t` / return node, so the variance comparison lives in the caption and prose only, not the figure.
- "three core algorithm families" reads, under a hostile parse, as a closed partition; model-based, distributional, and evolutionary methods are excluded.
- Panel (c) is roughly twice the element density of (a)/(b); its env-loop curvature (`curve=-0.55`) differs in sign and magnitude from (a)/(b) (`curve=+0.4`), and its loop labels sit at y=0.5 (vs 0.35) in the neighborhood of the dashed feedback arcs. In the rendered PNG these remain legible but visibly busier. The TD formula floats at (4.2, -0.75), below and right of the Environment box (centered x=2.5), weakening its visual tie to δ_t.

### (c) CHAPTER FIT — PASS

The figure + caption alone teach what the surrounding prose teaches: the forward decision pipeline of each family and actor-critic's TD feedback. This aligns with §"Actor-Critic Methods" (`.tex:93-105`, "address the high variance of REINFORCE by replacing Monte Carlo returns with bootstrapped TD targets") and the policy-gradient framing at `.tex:57`. A reader who sees only the figure and caption comes away with the correct structural taxonomy. Good fit.

### (d) EFFICIENCY / STANDARDS — PASS

- `plot_style` used correctly: imports `apply_style, COLORS, ALGO_COLORS, FIG_WIDE` and calls `apply_style()` (`.py:8-10`). Algorithm fills drawn from `ALGO_COLORS['DQN'|'REINFORCE'|'Actor-Critic']` and `COLORS['gray']`; all keys verified present in `sims/plot_style.py` (lines 38, 42, 44, 19; `FIG_WIDE` line 97). No hardcoded hex or `'C0'`-style shorthand for algorithm traces. `black` is used only for structural node outlines and default arrows, which is acceptable (not an algorithm trace).
- Flags interface for a diagram-only script matches CLAUDE.md exactly: `--data-only` exits with "No computation to cache (diagram-only script)." and `--plots-only` runs normally (`.py:379-388`). No caching, correct for a diagram.
- `_stdout.txt` present and matches the two `print` calls (`.py:374-375`); two lines, appropriate for a diagram-only script (nothing to tabulate).
- Deterministic: `np.random` imported but never called; prior audits confirmed a stable PNG MD5 across reruns.

---

## 7-point checklist

1. **Algorithm Identity** — PASS (not N/A). Each panel depicts the family it names; forward pipelines and the actor-critic feedback are the canonical schematics. One notation drift (`r_t` vs `r_{t+1}`), not an identity error.
2. **Environment / MDP Fidelity** — N/A. No MDP is simulated; "Environment" is a generic box closing the agent loop. No fidelity claim made or broken.
3. **Data Integrity** — N/A. No `compute_data()` by design; the script only draws. No numbers are computed or reported; stdout prints the output path and a confirmation line.
4. **Comparison Fairness** — N/A. No numerical comparison. The caption's variance claim is a theoretical statement grounded in prose, not an empirical result.
5. **Theoretical Sanity** — N/A (no numbers). The captions' qualitative claims (argmax selection; stochastic sampling; TD feedback lowers variance) are textbook-correct.
6. **Information Leakage** — N/A. No agent learns; no data, oracle, or held-out set exists.
7. **Seed / Reproducibility** — N/A in the Monte-Carlo sense; deterministic drawing, no RNG calls, PNG reproducible.

---

## Prior-audit comparison

Two prior audits: the 2026-05-19 audit (`audits/…_2026-05-19.md`, score 15%) and the v3 bullshit detector (`…_v3.md`, score 25% = cap-pinned). The v3 detector explicitly recommended STOP local editing (each round closed one issue and opened another of equal severity) and left six items open. Because the script and caption are unchanged since, resolution status is:

| Prior open item | Source | Status now |
|---|---|---|
| #4 "lower-variance vs REINFORCE's Monte-Carlo return" caption unanchored in panel (b) | v3 | STILL OPEN (caption byte-identical, `.tex:198`) |
| #5 "three core algorithm families" exhaustiveness over-read | v3 | STILL OPEN (caption unchanged) |
| #13 panel (c) env-loop curvature/label differ from (a)/(b) | v3 | STILL OPEN (`.py:324-339` unchanged, `curve=-0.55`) |
| #15 TD formula floats at y=-0.75, disassociated from δ_t | v3 | STILL OPEN (`.py:342-345` unchanged) |
| #16 env-loop labels at (4.6,0.5)/(0.4,0.5) crowd the dashed arcs | v3 | STILL OPEN (`.py:336-339` unchanged) |
| #17 panel (c) denser than (a)/(b) at print scale | v3 | STILL OPEN (edge count unchanged) |
| `r_t` vs `r_{t+1}` figure-vs-prose notation drift | both | STILL OPEN (`.py:343` unchanged) |

None RESOLVED, none REGRESSED. This is the expected state: no edits were made after the v3 detector advised stopping, so the artifact sits at the fixed point both prior audits describe — literal caption claims all HOLD, residual dilutions are all cosmetic/notational.

**Did this fresh pass find anything ≥25% severity beyond the cap that the prior audits missed?** NO. Every item found is cosmetic (env-loop asymmetry, formula float, label crowding, print density), caption-level (variance referent, "core" exhaustiveness), or notational (`r_t`/`r_{t+1}`). None is a conceptual/structural error that would push past 25% if the cap were lifted. The one place I sharpen the prior framing — the `r_t`/`r_{t+1}` clash is *intra-panel* (formula vs env label), not merely figure-vs-prose — is a tighter description of an already-flagged flaw, not a new defect and not ≥25%.

---

## Findings, severity-ordered

1. **[LOW, correctness/notation] `r_t` vs `r_{t+1}` inconsistency.** Panel (c) formula `δ_t = r_t + γ V_w(s_{t+1}) − V_w(s_t)` (`.py:343`) disagrees with the chapter's TD equation `δ_t = r_{t+1} + …` (`.tex:97`) and with the same panel's own env-loop label `r_{t+1}, s_{t+1}` (`.py:338`). A hostile reviewer writes one snarky line: "your figure contradicts your Eq. for δ_t." Substance survives a one-character revision.
2. **[LOW, presentation] Caption variance claim has no figure referent.** "lower-variance … than REINFORCE's Monte-Carlo return" (`.tex:198`); panel (b) shows no return/`G_t` node (`.py:224-266`). Claim is prose-grounded, figure-ungrounded.
3. **[LOW, presentation] Panel (c) cross-panel asymmetry and crowding.** Env-loop curvature `-0.55` vs `+0.4` in (a)/(b); labels raised to y=0.5 into the dashed-arc region; TD formula floats below/right of the Environment box. Legible in the PNG but visibly busier than (a)/(b). (v3 Findings 13/15/16/17.)
4. **[LOW, presentation] "three core algorithm families" reads as exhaustive.** Excludes model-based, distributional, evolutionary families under a hostile parse of "core."

All four are LOW and non-result-changing. The diagram does not visually contradict its caption: every element the caption promises is present and correctly labeled.

**Bullshit score: 25%** — Diagram-only cap applies as a ceiling and the content legitimately reaches it: a hostile reviewer has a concrete, specific flaw to write up (the `r_t`/`r_{t+1}` clash between the figure's δ_t formula, the chapter's δ_t equation, and the panel's own env label), plus cross-panel env-loop asymmetry — the textbook Reviewer-2 (25%) anchor. No conceptual error would carry it past the cap; the three architectures are faithfully depicted and every caption claim holds. Rounds up from the 2026-05-19 audit's 15% per the round-up / under-flagging-is-worse rule; matches the v3 detector's cap-pinned 25%.
