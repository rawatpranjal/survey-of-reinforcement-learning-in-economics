# Theory-Rigor for the RL Chapter (research dossier)

_Workstream 1. Give `ch03_theory` explicit Theorem + Proof rigor in Enoch Kang's style.
Status: research/planning. No tex written yet._

## The target style

State the result, then show the proof, cite the latest-and-greatest source, and visually
highlight the proof so a reader sees the chapter lays the rigorous foundation of the
monograph. Radical transparency. This is stronger than the in-repo precedent
(`ORE_main/appendix_a.tex` uses named theorem environments with bracketed titles, e.g.
`\begin{thm}[Equivalence of estimation methods]`, but carries no `\begin{proof}` blocks;
justification is woven into prose). We go further: explicit, highlighted proof blocks.

Scope note: this is about RL, not inverse RL. Kang's write-up is the presentation model
plus a proof source for the deep-RL convergence results.

## Current state of `ch03_theory/tex/planning_learning_v3.tex`

- 94k, the live theory chapter (`docs/main.tex:162`). Subsections: Geometry of DP; Value
  Learning; The Deadly Triad; Policy Learning; Hybrid Methods; Fundamental Tradeoffs;
  Conclusion.
- Exactly one `\begin{theorem}` (Policy Improvement, `\citet{howard1960}`, stated with no
  proof) and zero `proof`/`lemma`/`proposition`/`definition`/`assumption` envs.
- Style today is state-and-cite prose, e.g. the contraction bound is asserted inline
  ("Since $T$ is a $\gamma$-contraction ... Banach's fixed-point theorem guarantees
  $\|V_k - V^*\|_\infty \le \gamma^k\|V_0-V^*\|_\infty$") and TD error bounds are cited
  (`\citep[Theorem~1]{tsitsiklis1997}`) rather than derived.

## Sources to acquire (chunk T0)

Fetch full text (arXiv LaTeX source preferred), land as markdown in `docs/`, add bib keys.

| Source | id / location | bib key | what we take from it |
|---|---|---|---|
| Enoch Kang, RL lecture notes | `EK_RL_note.pdf`, gtown email msg `19efb63bbcb7528f` | none | 4.3 ODE stability, 4.5 semi-gradient / deadly triad, 4.6 positive results (Zhang online-SGD; van der Laan and Kallus). The proof-presentation model. |
| Kang, "Gradients can train reward models" | arXiv 2502.14131 (2025) | `Kang2025` (salvaged) | secondary style reference (ERM framing). |
| van der Laan and Kallus, FQE without Bellman completeness | arXiv 2512.23805 (2026) | to add | the FQE-via-stationary-weighting result and the open sup-norm gap. |
| Park et al. (Kang's CS-theory working paper) | `ParkParkJangKang_paper.pdf`, gtown msg `19efb63bbcb7528f` | to add | L2 Bellman-residual solvable under standard regularization. |
| Zhang et al. 2023 | already cited in the article; confirm key in `docs/refs.bib` | check | online-SGD DQN local convergence (realizability plus near-optimal init). |
| Antos, Szepesvari, Munos 2008 | to fetch | to add if missing | earliest source for minimizing the Bellman-error objective. |

The 0-byte Dropbox copies of the Kang paper
(`~/Dropbox/oxford_reinforcement_learning/literature/ddc_using_rl/2025_Kang_*.pdf`) are broken
and must be replaced by a fresh fetch.

## Proof inventory (candidate theorems for T2..Tn)

Each is a result to present as Theorem plus highlighted Proof. Final list fixed in T1.

1. **Contraction of the Bellman operator plus Banach fixed point.** $T$ is a $\gamma$-contraction
   in $\|\cdot\|_\infty$, so it has a unique fixed point $V^*$ with geometric convergence. Proof
   is short and self-contained; today it is asserted inline. Source: `denardo1967`, standard.
2. **Policy improvement (with proof).** The chapter already states it (`howard1960`) but omits
   the proof. Add the one-step-improvement telescoping argument.
3. **TD(0) / linear TD convergence plus projected-Bellman error bound.** The
   $\|\Phi\theta^*-V^\pi\|_{d^\pi}$ bound (`tsitsiklis1997`, currently cited as Theorem 1).
   Present the ODE / projection argument.
4. **The deadly triad plus Baird counterexample.** Off-policy plus bootstrapping plus function
   approximation can diverge. Present Baird's star as the explicit counterexample, and Kang's
   precise semi-gradient framing (semi-gradient TD is not gradient descent on Bellman error;
   the associated ODE can be unstable).
5. **Zhang et al. 2023 online-SGD DQN local convergence.** Under realizability (their Assumption
   1) and a near-optimal initialization (radius grows with replay-buffer size $N$), online SGD
   DQN converges locally to $Q$ in sup norm at rate about $1/\sqrt N$. Present as a local
   guarantee, with the initialization condition named as the load-bearing assumption.
6. **van der Laan and Kallus FQE plus the open sup-norm gap** (see below). Present the positive
   result, then state the gap as a labelled Open Problem.

## The open problem (label it explicitly in the chapter)

FQE without Bellman completeness (van der Laan and Kallus 2026) restores the projected operator's
contraction by reweighting to the target policy's stationary distribution, but secures
convergence only in a weighted $L^2$ norm and only to the projected fixed point $Q_{\text{proj}}$,
not the true $Q$. Converting such a bound into a counterfactual pays a factor growing with the
ratio of counterfactual occupancy to the stationary occupancy the bound is stated in, and
re-solving under perturbed primitives can make that ratio arbitrarily large. A sup-norm bound
between the true $Q$ and the projected pseudo-$Q$ is what is missing. Rust (gtown msg
`19eb4439fa172e18`) flags this needs functional analysis or a Sobolev-type structure; Kang's
notes "point some way towards such a bound." This is genuinely open and worth stating as such.

## Presentation decisions to make in T1

- Highlighted-proof environment: shaded `tcolorbox` or `mdframed` "Proof" box (define once).
- Theorem / lemma / definition / assumption env set (mirror `ORE_main/main.tex:21-26` naming).
- House rule (project CLAUDE.md): original author notation first, then modern RL notation.
- Keep it RL. IRL identification proofs are out of scope for this chapter.
